import random                          # 用于随机采样（经验回放时随机抽取数据）
import sys                             # 用于修改模块搜索路径
import os                              # 用于获取文件路径
import gymnasium as gym               # OpenAI Gymnasium，提供强化学习标准环境（如CartPole）
import numpy as np                    # 数值计算库，处理数组和矩阵
import collections                    # 提供 deque 双端队列，用作经验回放缓冲区
from tqdm import tqdm                 # 进度条库，显示训练进度
import torch                          # PyTorch 深度学习框架
import torch.nn.functional as F       # PyTorch 中的函数式接口，包含 relu、mse_loss 等
import matplotlib.pyplot as plt       # 绘图库，用于可视化训练曲线

sys.path.insert(0, os.path.dirname(__file__))  # 将脚本所在目录加入搜索路径，确保能找到同目录下的 rl_utils
import rl_utils                        # 导入自定义工具函数（如 moving_average 滑动平均）

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 用双端队列（队列先进先出，在固定容量下自动删除）存储经验，maxlen 限制最大容量，超出后自动丢弃最旧数据（先进先出）

    def add(self, state, action, reward, next_state, done):  # 将一条交互经验 (s, a, r, s', done) 存入回放池
        self.buffer.append((state, action, reward, next_state, done))  # 打包成元组后追加到队列末尾

    def sample(self, batch_size):  # 从回放池中随机采样 batch_size 条经验，用于训练，buffer是缓冲区的意思
        transitions = random.sample(self.buffer, batch_size)  # 不放回随机采样 batch_size 条经验元组
        state, action, reward, next_state, done = zip(*transitions)  # 解压：将列表[(s,a,r,s',d),...]转置为分组元组（从每一次每一组（行），转化为了同一个状态，同一个动作（列））
        return np.array(state), action, reward, np.array(next_state), done  # state/next_state 转为 numpy 数组便于批量处理

    def size(self):  # 返回回放池当前存储的经验条数
        return len(self.buffer)

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()          # 调用父类 nn.Module 的初始化，注册网络参数
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)   # 第一层全连接：输入维度=状态维度，输出维度=隐藏层维度
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)  # 第二层全连接：输入=隐藏层维度，输出=每个动作对应的 Q 值

    def forward(self, x):        #前向传播
        x = F.relu(self.fc1(x))  # 输入先经过第一层线性变换，再用 ReLU 激活（将负值截断为0，引入非线性）
        return self.fc2(x)       # 输出层不加激活函数，直接输出各动作的 Q 值（可为任意实数）

class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim  # 保存动作空间维度，供 epsilon-贪婪策略随机选动作时使用
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # 创建在线 Q 网络（用于选动作和计算当前 Q 值），并移至指定设备（CPU/GPU）
        # 目标网络：结构与在线网络相同，但参数更新滞后，用于计算更稳定的 TD 目标
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 使用 Adam 优化器更新在线 Q 网络的参数，lr 为学习率
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma                   # 折扣因子 γ，控制未来奖励的权重（越接近1越重视长期回报）
        self.epsilon = epsilon               # ε-贪婪策略中随机探索的概率（小值表示主要利用，少量探索）
        self.target_update = target_update   # 每隔多少次 update 调用，才将在线网络参数同步到目标网络
        self.count = 0                       # 记录 update 被调用的总次数，用于判断何时更新目标网络
        self.device = device                 # 训练所用设备（"cuda" 或 "cpu"）

    def take_action(self, state):  # 根据 ε-贪婪策略选择动作
        if np.random.random() < self.epsilon:           # 以 ε 的概率随机探索
            action = np.random.randint(self.action_dim) # 在所有合法动作中均匀随机选一个
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)  # 将状态转为 shape=(1, state_dim) 的 float 张量并移至设备，因为神经网络的输入至少是一个二维矩阵
            action = self.q_net(state).argmax().item()  # 前向传播得到各动作 Q 值，取最大 Q 值对应的动作索引（贪婪）
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)         # 将一批状态转为 float 张量，shape=(batch, state_dim)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)                                                  # 动作转为列向量，shape=(batch, 1)，便于 gather 操作
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)  # 奖励转为列向量，shape=(batch, 1)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)   # 下一状态，shape=(batch, state_dim)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)  # 终止标志转为列向量，shape=(batch, 1)，值为 0 或 1

        q_values = self.q_net(states).gather(1, actions)  # 在线网络前向传播得到所有动作的Q值，再用 gather 按 actions 索引取出实际执行动作的 Q 值，shape=(batch,1)
        # 用目标网络计算下一状态所有动作的 Q 值，.max(1)[0] 取每行最大值，.view(-1,1) 变为列向量
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        # Bellman 目标：r + γ * max_a' Q_target(s', a') * (1 - done)
        # (1 - dones) 保证终止状态后无后续 Q 值（done=1 时该项为0）
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = F.mse_loss(q_values, q_targets)  # 计算在线Q值与TD目标之间的均方误差损失（即 Bellman 误差）
        self.optimizer.zero_grad()   # 清空上一步积累的梯度（PyTorch 默认梯度累加，每次反传前必须清零）
        dqn_loss.backward()          # 对损失函数做反向传播，计算在线Q网络各参数的梯度
        self.optimizer.step()        # Adam 优化器根据梯度更新在线Q网络的参数（更新本质是改进后的随机梯度下降）

        if self.count % self.target_update == 0:  # 每隔 target_update 次更新，才同步一次目标网络（硬更新）（跳变可能训练抖动）
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 将在线网络的全部参数直接复制到目标网络（硬更新，区别于软更新的加权平均）
        self.count += 1  # 更新计数器，下次判断是否需要同步目标网络

# ===================== 超参数设置 =====================
lr = 2e-3           # 学习率，Adam 优化器的步长，控制每次参数更新的幅度
num_episodes = 500  # 总训练回合数，每个 episode 是一局完整的游戏
hidden_dim = 128    # Q 网络隐藏层的神经元数量，越大表达能力越强但计算更慢
gamma = 0.98        # 折扣因子，接近1说明智能体重视长期奖励
epsilon = 0.01      # ε-贪婪策略的探索率，1% 的概率随机行动以避免局部最优
target_update = 10  # 每训练10次（调用10次 update），才将在线网络参数同步到目标网络
buffer_size = 10000 # 经验回放池最大容量，超出后丢弃最旧经验
minimal_size = 500  # 回放池至少积累 500 条经验后才开始训练，确保样本多样性
batch_size = 64     # 每次从回放池随机采样的经验条数，用于一次梯度更新
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")  # 优先使用 GPU（cuda）加速训练，不可用时退回 CPU

# ===================== 环境与智能体初始化 =====================
env_name = 'CartPole-v1'      # 使用 CartPole-v1 环境：控制小车保持竖杆不倒，最多坚持 500 步
env = gym.make(env_name)      # 创建 Gymnasium 环境实例
random.seed(0)                # 固定 Python random 的随机种子，保证实验可复现
np.random.seed(0)             # 固定 NumPy 的随机种子
torch.manual_seed(0)          # 固定 PyTorch 的随机种子（CPU 上的随机操作）
replay_buffer = ReplayBuffer(buffer_size)              # 创建经验回放池，容量为 buffer_size
state_dim = env.observation_space.shape[0]  # type: ignore[index]   # 获取状态维度（CartPole 为 4：小车位置、速度、杆角度、角速度）
action_dim = env.action_space.n  # type: ignore[union-attr]          # 获取动作维度（CartPole 为 2：向左/向右施力）
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)  # 创建 DQN 智能体，传入所有超参数

# ===================== 训练主循环 =====================
return_list = []               # 记录每个 episode 的累计奖励，用于绘制学习曲线
for i in range(10):            # 将 500 个 episode 分成 10 轮（每轮 50 个 episode），方便显示进度
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:  # 创建进度条，每轮显示 50 步进度
        for i_episode in range(int(num_episodes / 10)):  # 每轮训练 50 个 episode
            episode_return = 0.0          # 初始化本 episode 的累计奖励为 0
            state, _ = env.reset()        # 重置环境，获取初始状态（第二个返回值为 info，用 _ 忽略）
            done = False                  # 初始化终止标志为 False
            while not done:               # 在 episode 未结束前持续交互
                action = agent.take_action(state)  # 智能体根据当前状态用 ε-贪婪策略选择动作
                next_state, reward, terminated, truncated, _ = env.step(action)  # 执行动作，获取下一状态、奖励、是否自然终止、是否超时截断
                done = terminated or truncated     # 自然终止（杆倒下）或超时截断都算 episode 结束
                replay_buffer.add(state, action, reward, next_state, done)  # 将这条经验 (s,a,r,s',done) 存入回放池
                state = next_state                 # 更新当前状态为下一状态，继续下一步
                episode_return += float(reward)    # 累加本步奖励到本 episode 总回报
                # 当回放池数据量超过 minimal_size 后才开始训练（确保初始样本足够多样）
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)  # 从回放池随机采样一个 batch 的经验
                    transition_dict = {
                        'states': b_s,        # 批量状态，shape=(batch_size, state_dim)
                        'actions': b_a,       # 批量动作
                        'next_states': b_ns,  # 批量下一状态
                        'rewards': b_r,       # 批量奖励
                        'dones': b_d          # 批量终止标志
                    }
                    agent.update(transition_dict)  # 用这个 batch 更新 Q 网络参数（一步梯度下降）
            return_list.append(episode_return)     # 本 episode 结束，记录总回报
            if (i_episode + 1) % 10 == 0:         # 每完成 10 个 episode，刷新一次进度条显示
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),  # 显示当前是第几个总 episode
                    'return':
                    '%.3f' % np.mean(return_list[-10:])               # 显示最近 10 个 episode 的平均回报
                })
            pbar.update(1)  # 进度条前进一步（代表完成了一个 episode）

# ===================== 绘图：原始回报曲线 =====================
episodes_list = list(range(len(return_list)))  # 生成 x 轴：[0, 1, 2, ..., 499]
plt.plot(episodes_list, return_list)           # 绘制每个 episode 的原始累计回报
plt.xlabel('Episodes')                         # x 轴标签
plt.ylabel('Returns')                          # y 轴标签
plt.title('DQN on {}'.format(env_name))        # 图标题
plt.show()                                     # 显示图像

# ===================== 绘图：滑动平均平滑曲线 =====================
mv_return = rl_utils.moving_average(return_list, 9)  # 对回报序列做窗口为9的滑动平均，消除噪声使曲线更平滑
plt.plot(episodes_list, mv_return)             # 绘制平滑后的回报曲线
plt.xlabel('Episodes')                         # x 轴标签
plt.ylabel('Returns')                          # y 轴标签
plt.title('DQN on {}'.format(env_name))        # 图标题
plt.show()                                     # 显示图像


# ===================== 进阶：带卷积层的 Q 网络（用于图像输入，如 Atari 游戏）=====================
class ConvolutionalQnet(torch.nn.Module):
    ''' 加入卷积层的Q网络 '''
    def __init__(self, action_dim, in_channels=4):  # in_channels=4 对应将连续4帧灰度图叠加作为输入（捕捉运动信息）
        super(ConvolutionalQnet, self).__init__()   # 调用父类初始化
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)  # 第1个卷积层：4通道→32通道，8×8卷积核，步长4（大幅缩小特征图）
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)           # 第2个卷积层：32→64通道，4×4卷积核，步长2（继续缩小）
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)           # 第3个卷积层：64→64通道，3×3卷积核，步长1（精细特征提取）
        self.fc4 = torch.nn.Linear(7 * 7 * 64, 512)  # 全连接层：将卷积输出的 7×7×64 个特征展平后映射到 512 维（适用于 84×84 输入）
        self.head = torch.nn.Linear(512, action_dim)  # 输出层：512 维特征映射到各动作的 Q 值

    def forward(self, x):
        x = x / 255                      # 将像素值从 [0, 255] 归一化到 [0, 1]，加速收敛、防止梯度爆炸
        x = F.relu(self.conv1(x))        # 第1卷积 + ReLU 激活，提取低层视觉特征（边缘、纹理）
        x = F.relu(self.conv2(x))        # 第2卷积 + ReLU 激活，提取中层特征
        x = F.relu(self.conv3(x))        # 第3卷积 + ReLU 激活，提取高层语义特征
        x = x.reshape(x.size(0), -1)     # 将 (batch, 64, 7, 7) 的张量展平为 (batch, 7*7*64)，准备输入全连接层
        x = F.relu(self.fc4(x))          # 全连接层 + ReLU，整合空间特征为抽象表示
        return self.head(x)              # 输出各动作的 Q 值，shape=(batch, action_dim)，不加激活函数
