import random                          # 用于随机采样（经验回放时随机抽取数据）
import gymnasium as gym               # OpenAI Gymnasium，提供强化学习标准环境（如 Pendulum）
import numpy as np                    # 数值计算库，处理数组和矩阵
from tqdm import tqdm                 # 进度条库，显示训练进度
import torch                          # PyTorch 深度学习框架
import torch.nn.functional as F       # PyTorch 中的函数式接口，包含 relu、mse_loss 等
import matplotlib.pyplot as plt       # 绘图库，用于可视化训练曲线

import rl_utils  # 导入同目录下的自定义工具函数（如 ReplayBuffer 经验回放池、moving_average 滑动平均）；脚本所在目录已自动加入 sys.path，无需额外操作


class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络（普通DQN使用） '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()                             # 调用父类 nn.Module 的初始化，注册网络参数
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)       # 第一层全连接：输入维度=状态维度，输出维度=隐藏层维度
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)      # 第二层全连接：输入=隐藏层维度，输出=每个动作对应的 Q 值

    def forward(self, x):             # 前向传播：定义数据从输入到输出的计算流程
        x = F.relu(self.fc1(x))       # 输入先经过第一层线性变换，再用 ReLU 激活（将负值截断为0，引入非线性）
        return self.fc2(x)            # 输出层不加激活函数，直接输出各动作的 Q 值（可为任意实数）


class VAnet(torch.nn.Module):
    ''' 只有一层隐藏层的 A 网络和 V 网络（Dueling DQN 专用） '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()                            # 调用父类 nn.Module 的初始化，注册网络参数
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)       # 共享特征层：将状态映射到隐藏表示，A 流和 V 流共同复用该层
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)     # 优势流（Advantage stream）：输出每个动作相对于平均水平的优势值 A(s,a)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)              # 价值流（Value stream）：输出当前状态的整体价值 V(s)，与具体动作无关

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))          # 共享层经 ReLU 激活后，送入优势流，得到各动作的优势值 A(s,a)
        V = self.fc_V(F.relu(self.fc1(x)))          # 共享层经 ReLU 激活后，送入价值流，得到状态价值 V(s)，shape=(batch,1)
        Q = V + A - A.mean(1).view(-1, 1)           # Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]；减去均值是为了保证可识别性（让 V 和 A 的分解唯一）
        return Q                                     # 返回各动作的 Q 值，shape=(batch, action_dim)


class DQN:
    ''' DQN算法，包括 Double DQN 和 Dueling DQN '''
    def __init__(self,
                 state_dim,
                 hidden_dim,
                 action_dim,
                 learning_rate,
                 gamma,
                 epsilon,
                 target_update,
                 device,
                 dqn_type='VanillaDQN'):  # dqn_type 区分普通DQN、Double DQN、Dueling DQN，默认为普通DQN
        self.action_dim = action_dim      # 保存动作空间维度，供 epsilon-贪婪策略随机选动作时使用
        if dqn_type == 'DuelingDQN':      # Dueling DQN 使用 VAnet（双流网络）代替普通 Qnet
            self.q_net = VAnet(state_dim, hidden_dim, self.action_dim).to(device)          # 在线网络：VAnet 同时估计 V(s) 和 A(s,a)，合并得到 Q 值
            self.target_q_net = VAnet(state_dim, hidden_dim, self.action_dim).to(device)  # 目标网络：结构与在线网络相同，参数更新滞后，用于计算更稳定的 TD 目标
        else:                             # 普通 DQN 或 Double DQN 使用标准 Qnet
            self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)          # 在线网络：标准两层全连接 Q 网络
            self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)  # 目标网络：结构与在线网络相同，参数更新滞后
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)  # 使用 Adam 优化器更新在线 Q 网络的参数，lr 为学习率
        self.gamma = gamma                   # 折扣因子 γ，控制未来奖励的权重（越接近1越重视长期回报）
        self.epsilon = epsilon               # ε-贪婪策略中随机探索的概率（小值表示主要利用，少量探索）
        self.target_update = target_update   # 每隔多少次 update 调用，才将在线网络参数同步到目标网络
        self.count = 0                       # 记录 update 被调用的总次数，用于判断何时更新目标网络
        self.dqn_type = dqn_type             # 保存算法类型，update 中用于区分 DQN 与 Double DQN 的目标值计算方式
        self.device = device                 # 训练所用设备（"cuda" 或 "cpu"）

    def take_action(self, state):  # 根据 ε-贪婪策略选择动作
        if np.random.random() < self.epsilon:            # 以 ε 的概率随机探索
            action = np.random.randint(self.action_dim)  # 在所有合法动作中均匀随机选一个
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)  # 将状态转为 shape=(1, state_dim) 的 float 张量并移至设备（神经网络输入至少是二维矩阵）
            action = self.q_net(state).argmax().item()   # 前向传播得到各动作 Q 值，取最大 Q 值对应的动作索引（贪婪）
        return action

    def max_q_value(self, state):  # 获取当前状态下在线网络输出的最大 Q 值（用于监控训练过程中 Q 值的变化趋势）
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)  # 状态转为张量并移至设备
        return self.q_net(state).max().item()  # 前向传播后取所有动作 Q 值中的最大值并转为 Python 标量

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)          # 将一批状态转为 float 张量，shape=(batch, state_dim)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)                                                   # 动作转为列向量，shape=(batch, 1)，便于 gather 操作
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)   # 奖励转为列向量，shape=(batch, 1)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)    # 下一状态，shape=(batch, state_dim)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)   # 终止标志转为列向量，shape=(batch, 1)，值为 0 或 1

        q_values = self.q_net(states).gather(1, actions)  # 在线网络前向传播得到所有动作的Q值，再用 gather 按 actions 索引取出实际执行动作的 Q 值，shape=(batch,1)
        # Double DQN 与普通 DQN 的核心区别：选动作和评估动作使用不同的网络
        if self.dqn_type == 'DoubleDQN':  # Double DQN 的目标值计算方式（Dueling DQN 也可与此结合使用）
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)                # 用在线网络选出下一状态中 Q 值最大的动作索引（"选择"步骤）
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)  # 用目标网络评估该动作的 Q 值（"评估"步骤，解耦降低过估计偏差）
        else:  # 普通 DQN 或 Dueling DQN 的目标值计算方式
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)  # 目标网络直接选最大 Q 值（选择与评估都用目标网络）
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # Bellman 目标：r + γ * Q_target * (1 - done)，done=1 时终止状态无后续价值
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 计算在线Q值与TD目标之间的均方误差损失（即 Bellman 误差）
        self.optimizer.zero_grad()   # 清空上一步积累的梯度（PyTorch 默认梯度累加，每次反传前必须清零）
        dqn_loss.backward()          # 对损失函数做反向传播，计算在线Q网络各参数的梯度
        self.optimizer.step()        # Adam 优化器根据梯度更新在线Q网络的参数

        if self.count % self.target_update == 0:  # 每隔 target_update 次更新，才同步一次目标网络（硬更新）
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 将在线网络的全部参数直接复制到目标网络（硬更新，区别于软更新的加权平均）
        self.count += 1  # 更新计数器，用于下次判断是否需要同步目标网络


# ===================== 超参数设置 =====================
lr = 1e-2            # 学习率，Adam 优化器的步长，Pendulum 任务收敛较慢需要相对较大的学习率
num_episodes = 200   # 总训练回合数，每个 episode 是一局完整的游戏
hidden_dim = 128     # Q 网络隐藏层的神经元数量，越大表达能力越强但计算更慢
gamma = 0.98         # 折扣因子，接近1说明智能体重视长期奖励
epsilon = 0.01       # ε-贪婪策略的探索率，1% 的概率随机行动以避免局部最优
target_update = 50   # 每训练50次（调用50次 update）才将在线网络参数同步到目标网络
buffer_size = 5000   # 经验回放池最大容量，超出后丢弃最旧经验
minimal_size = 1000  # 回放池至少积累 1000 条经验后才开始训练，确保样本多样性
batch_size = 64      # 每次从回放池随机采样的经验条数，用于一次梯度更新
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")  # 优先使用 GPU（cuda）加速训练，不可用时退回 CPU

# ===================== 环境初始化 =====================
env_name = 'Pendulum-v1'                    # Pendulum-v1：控制钟摆保持直立，原生为连续动作空间，这里将其离散化
env = gym.make(env_name)                    # 创建 Gymnasium 环境实例
state_dim = env.observation_space.shape[0]  # type: ignore[index]  # 获取状态维度（Pendulum 为 3：cos θ、sin θ、角速度）
action_dim = 11                             # 将连续动作空间（力矩）离散化为 11 个等间距的动作


def dis_to_con(discrete_action, env, action_dim):  # 将离散动作编号转换回对应的连续力矩值
    action_lowbound = env.action_space.low[0]      # 连续动作的下界（Pendulum 为 -2.0）
    action_upbound = env.action_space.high[0]      # 连续动作的上界（Pendulum 为 +2.0）
    # 将 [0, action_dim-1] 的整数动作线性映射到 [low, high] 的连续值
    return action_lowbound + (discrete_action / (action_dim - 1)) * (action_upbound - action_lowbound)


def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []        # 记录每个 episode 的累计奖励，用于绘制学习曲线
    max_q_value_list = []   # 记录每个时间步的平滑 Q 值，用于监控 Q 值过估计现象
    max_q_value = 0         # 初始化指数移动平均 Q 值为 0（用于平滑 Q 值曲线，防止剧烈抖动）
    for i in range(10):     # 将总 episode 分成 10 轮，方便进度条分段显示
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:  # 每轮创建一个进度条，显示本轮进度
            for i_episode in range(int(num_episodes / 10)):  # 每轮训练 num_episodes/10 个 episode
                episode_return = 0          # 初始化本 episode 的累计奖励为 0
                state, _ = env.reset()      # 重置环境，获取初始状态（gymnasium 返回 (state, info) 元组，用 _ 忽略 info）
                done = False                # 初始化终止标志为 False
                while not done:             # 在 episode 未结束前持续与环境交互
                    action = agent.take_action(state)  # 智能体根据当前状态用 ε-贪婪策略选择离散动作编号
                    max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995  # 对最大Q值做指数移动平均（系数0.005），平滑监控曲线
                    max_q_value_list.append(max_q_value)  # 记录当前步的平滑最大 Q 值
                    action_continuous = dis_to_con(action, env, agent.action_dim)          # 将离散动作编号转换为 Pendulum 所需的连续力矩值
                    next_state, reward, terminated, truncated, _ = env.step([action_continuous])  # 执行连续动作，获取下一状态、奖励、是否自然终止、是否超时截断
                    done = terminated or truncated  # 自然终止或超时截断都算 episode 结束
                    replay_buffer.add(state, action, reward, next_state, done)  # 将经验 (s,a,r,s',done) 存入回放池
                    state = next_state      # 更新当前状态为下一状态，继续下一步
                    episode_return += reward  # 累加本步奖励到本 episode 总回报
                    if replay_buffer.size() > minimal_size:  # 回放池数据量超过阈值后才开始训练（确保初始样本足够多样）
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
    return return_list, max_q_value_list  # 返回回报列表和 Q 值列表，供后续绘图使用


# ===================== 随机种子与智能体初始化 =====================
random.seed(0)         # 固定 Python random 的随机种子，保证实验可复现
np.random.seed(0)      # 固定 NumPy 的随机种子
torch.manual_seed(0)   # 固定 PyTorch 的随机种子（CPU 上的随机操作）
replay_buffer = rl_utils.ReplayBuffer(buffer_size)  # 创建经验回放池，容量为 buffer_size
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device, dqn_type='DuelingDQN')  # 创建 Dueling DQN 智能体（指定 dqn_type='DuelingDQN' 以启用 VAnet 双流网络）
return_list, max_q_value_list = train_DQN(agent, env, num_episodes,
                                          replay_buffer, minimal_size,
                                          batch_size)  # 训练智能体，返回每回合总回报与每步最大Q值

# ===================== 绘图：回报曲线 =====================
episodes_list = list(range(len(return_list)))         # 生成 x 轴：[0, 1, 2, ..., num_episodes-1]
mv_return = rl_utils.moving_average(return_list, 5)   # 对回报序列做窗口为5的滑动平均，消除噪声使曲线更平滑
plt.plot(episodes_list, mv_return)                    # 绘制平滑后的回报曲线
plt.xlabel('Episodes')                                # x 轴标签
plt.ylabel('Returns')                                 # y 轴标签
plt.title('Dueling DQN on {}'.format(env_name))       # 图标题
plt.show()                                            # 显示图像

# ===================== 绘图：Q值曲线（监控过估计问题）=====================
frames_list = list(range(len(max_q_value_list)))      # 生成 x 轴：时间步编号
plt.plot(frames_list, max_q_value_list)               # 绘制每步的平滑最大 Q 值曲线
plt.axhline(0, c='orange', ls='--')                  # 橙色虚线标注 Q=0（参考基准线）
plt.axhline(10, c='red', ls='--')                    # 红色虚线标注 Q=10（参考基准线，可对比其他算法的过估计程度）
plt.xlabel('Frames')                                  # x 轴标签：时间步
plt.ylabel('Q value')                                 # y 轴标签
plt.title('Dueling DQN on {}'.format(env_name))       # 图标题
plt.show()                                            # 显示图像
