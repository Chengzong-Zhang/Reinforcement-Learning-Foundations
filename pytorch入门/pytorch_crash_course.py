# ============================================================
#   PyTorch 速成教程 —— 面向强化学习
#   涵盖：张量基础 → 神经网络 → 自动微分 →
#         经验回放 → DQN → Policy Gradient → Actor-Critic
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

# ============================================================
# 第一部分：张量基础 —— 用 RL 场景理解
# ============================================================

# ---------- 1.1 RL 中的常见张量 ----------

# 状态 (state)：CartPole 的状态是4个浮点数
state = torch.tensor([0.01, -0.02, 0.03, -0.04], dtype=torch.float32)
print(state.shape)   # torch.Size([4])

# 批量状态：从经验回放中采样一批，shape = (batch, state_dim)
batch_states = torch.randn(32, 4)    # batch=32，每个状态4维

# 动作 (action)：离散动作，整数类型
action = torch.tensor(1, dtype=torch.long)           # 单个动作
batch_actions = torch.randint(0, 2, (32,))           # 32个动作，值为 0 或 1

# 奖励 (reward)：标量浮点
reward = torch.tensor(1.0)
batch_rewards = torch.randn(32)                      # 32个奖励

# done 标志：episode 是否结束
done = torch.tensor(False)
batch_dones = torch.zeros(32, dtype=torch.bool)      # 全未结束

# ---------- 1.2 设备管理（CPU / GPU）----------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# RL 中所有张量和模型都要放到同一个设备
state = state.to(device)
batch_states = batch_states.to(device)

# ---------- 1.3 张量创建 ----------

# RL 中常见的初始化场景
q_values  = torch.zeros(4)              # 初始 Q 值，4个动作
returns_g = torch.zeros(200)            # 一条 episode 的累积回报，最多200步
advantage = torch.empty(32)            # 优势函数，之后会填充数据

x = torch.randn(32, 4)                 # 随机初始化一批状态（调试用）
x = torch.ones(32, 1)                  # 全1，常用于构造基线
x = torch.arange(0, 200)              # [0,1,...,199]，用于折扣因子计算

# ---------- 1.4 张量属性 ----------

t = torch.randn(32, 4)
print(t.shape)      # torch.Size([32, 4])
print(t.ndim)       # 2（batch维 + 状态维）
print(t.dtype)      # torch.float32
print(t.device)     # cpu 或 cuda:0
print(t.numel())    # 128（32×4）

# ---------- 1.5 形状变换 ----------

# Q 网络输出 (batch, actions)，取单条时需要 squeeze
q_out = torch.randn(1, 2)              # 单条状态的 Q 值，shape (1,2)
q_out = q_out.squeeze(0)              # -> (2,)，去掉 batch 维

# Policy 网络输出 logits，需要 unsqueeze 配合其他操作
logits = torch.randn(4)               # 单步 (4个动作的 logit)
logits = logits.unsqueeze(0)          # -> (1,4)，加上 batch 维才能送入网络

# reshape：将卷积特征展平后送入全连接层（Atari 像素输入场景）
feature_map = torch.randn(32, 64, 7, 7)          # (batch, C, H, W)
flat = feature_map.reshape(32, -1)               # (32, 3136)，-1 自动推断

# ---------- 1.6 索引与切片 ----------

batch_states = torch.randn(32, 4)
batch_actions = torch.randint(0, 2, (32,))

# 从 Q 值矩阵中取出每个样本实际执行的动作对应的 Q 值（DQN 核心操作）
q_all = torch.randn(32, 2)                       # 所有动作的 Q 值，(32,2)
q_taken = q_all.gather(1, batch_actions.unsqueeze(1))   # (32,1)
# gather(dim, index)：在 dim=1（动作维）上，按 index 选值
q_taken = q_taken.squeeze(1)                     # -> (32,)

# 布尔掩码：只更新非终止状态的 TD 目标（done=False 才有下一步）
batch_dones = torch.zeros(32, dtype=torch.bool)
batch_dones[5] = True                            # 第5个样本 episode 结束
mask = ~batch_dones                              # 取反：True 表示未结束
next_q = torch.randn(32)
next_q[batch_dones] = 0.0                        # 终止状态的下一步 Q 值置0

# ---------- 1.7 数学运算 ----------

rewards  = torch.randn(32)
next_q   = torch.randn(32)
gamma    = 0.99

# Bellman 方程：TD 目标 = r + γ * max Q(s',a')
td_target = rewards + gamma * next_q             # 逐元素加法和标量乘法

# 归约：计算一批损失的均值
losses = (td_target - torch.randn(32)) ** 2
mean_loss = losses.mean()                        # 标量，用于 backward()

# 计算折扣累积回报（从后往前遍历）
T = 10
ep_rewards = torch.tensor([1.0] * T)
returns = torch.zeros(T)
G = 0.0
for t in reversed(range(T)):
    G = ep_rewards[t].item() + gamma * G        # G_t = r_t + γ * G_{t+1}
    returns[t] = G

# 优势函数标准化（减均值除标准差，稳定训练）
advantages = returns - returns.mean()
advantages = advantages / (advantages.std() + 1e-8)  # +eps 防止除0

# ---------- 1.8 与 NumPy 互转 ----------

# RL 环境（如 gym）返回 numpy array，需要转成 tensor
np_state = np.array([0.01, -0.02, 0.03, -0.04], dtype=np.float32)
state_t = torch.from_numpy(np_state)            # 共享内存，零拷贝
state_t = torch.tensor(np_state)               # 拷贝（更安全）

# 网络输出 tensor，执行动作前需要转回 numpy/python int
q_values = torch.tensor([0.1, 0.9])
action = q_values.argmax().item()               # .item() 转为 Python int

# detach() + numpy()：推理时不需要梯度
probs = torch.softmax(torch.randn(4), dim=0)
probs_np = probs.detach().numpy()               # 先 detach，再转 numpy
action = np.random.choice(4, p=probs_np)       # 按概率采样动作

# ============================================================
# 第二部分：RL 中的神经网络
# ============================================================

# ---------- 2.1 Q 网络（DQN 用）----------
# 输入：状态 s，输出：每个动作的 Q(s,a)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),   # 状态 -> 隐层
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),  # 隐层 -> 每个动作的 Q 值
            # 输出层没有激活函数，Q 值可以是任意实数
        )

    def forward(self, state):
        return self.net(state)                  # shape: (batch, action_dim)

# ---------- 2.2 策略网络（Policy Network，离散动作）----------
# 输入：状态，输出：各动作的概率分布（用于 REINFORCE / Actor-Critic）

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            # 输出层不加 softmax，用 Categorical 分布采样（更稳定）
        )

    def forward(self, state):
        logits = self.net(state)               # 未归一化的对数概率
        return logits

    def get_action(self, state):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()                 # 按概率采样一个动作
        log_prob = dist.log_prob(action)       # 该动作的对数概率，用于计算梯度
        return action.item(), log_prob

# ---------- 2.3 价值网络（Value Network，用于 Actor-Critic）----------
# 输入：状态，输出：状态价值 V(s)（标量）

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),          # 输出标量 V(s)
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)     # (batch,1) -> (batch,)

# ---------- 2.4 Actor-Critic 合并网络（共享特征提取）----------
# 前几层共享，最后分叉为 actor head 和 critic head

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # 共享的特征提取主干
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        # Actor head：输出动作 logits
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        # Critic head：输出状态价值
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        feat = self.backbone(state)            # 共享特征
        logits = self.actor_head(feat)         # 动作概率（未归一化）
        value  = self.critic_head(feat).squeeze(-1)  # 状态价值 V(s)
        return logits, value

    def get_action(self, state):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()               # 策略熵，用于鼓励探索
        return action.item(), log_prob, value, entropy

# ---------- 2.5 Dueling DQN 网络 ----------
# 将 Q(s,a) 分解为 V(s) + A(s,a)（优势），提升学习效率

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream    = nn.Linear(hidden_dim, 1)          # V(s)
        self.advantage_stream = nn.Linear(hidden_dim, action_dim) # A(s,a)

    def forward(self, state):
        feat = self.backbone(state)
        V = self.value_stream(feat)                              # (batch,1)
        A = self.advantage_stream(feat)                          # (batch, actions)
        # Q = V + (A - mean(A))，减均值保证唯一性
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q

# ---------- 2.6 查看模型 ----------

q_net = QNetwork(state_dim=4, action_dim=2)
print(q_net)
total_params = sum(p.numel() for p in q_net.parameters())
print(f"参数量: {total_params}")

# 用假输入测试输出形状（调试必备）
dummy_state = torch.randn(1, 4)
with torch.no_grad():
    out = q_net(dummy_state)
print(f"Q网络输出形状: {out.shape}")    # torch.Size([1, 2])

# ============================================================
# 第三部分：自动微分 —— RL 中的损失计算
# ============================================================

# ---------- 3.1 DQN 的 TD 损失 ----------

# 模拟一批经验：(s, a, r, s', done)
state_dim, action_dim, batch_size = 4, 2, 32
gamma = 0.99

q_net    = QNetwork(state_dim, action_dim)
q_target = QNetwork(state_dim, action_dim)  # Target 网络（权重延迟更新）
q_target.load_state_dict(q_net.state_dict())  # 初始时同步权重

optimizer = optim.Adam(q_net.parameters(), lr=1e-3)

s  = torch.randn(batch_size, state_dim)
a  = torch.randint(0, action_dim, (batch_size,))
r  = torch.randn(batch_size)
s2 = torch.randn(batch_size, state_dim)
d  = torch.zeros(batch_size, dtype=torch.bool)

# 前向传播：计算当前 Q 值
q_pred = q_net(s)                              # (32,2)
q_pred = q_pred.gather(1, a.unsqueeze(1)).squeeze(1)  # 取实际动作的 Q 值 (32,)

# 用 target 网络计算 TD 目标（不需要梯度）
with torch.no_grad():
    q_next = q_target(s2).max(dim=1).values   # max Q(s',a')，shape (32,)
    q_next[d] = 0.0                           # 终止状态无下一步
    td_target = r + gamma * q_next            # Bellman 方程

# 计算损失并反向传播
loss = F.mse_loss(q_pred, td_target)          # TD 误差的均方
# 也可以用 Huber Loss，对异常值更鲁棒：
# loss = F.smooth_l1_loss(q_pred, td_target)

optimizer.zero_grad()    # 清零梯度（每次 backward 前必须清零）
loss.backward()          # 反向传播，计算所有参数的梯度
torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)  # 梯度裁剪
optimizer.step()         # 更新参数

print(f"DQN TD Loss: {loss.item():.4f}")

# ---------- 3.2 REINFORCE 的策略梯度损失 ----------

policy_net = PolicyNetwork(state_dim=4, action_dim=2)
optimizer_p = optim.Adam(policy_net.parameters(), lr=1e-3)

# 假设跑完一条 episode，收集了 log_probs 和 returns
log_probs = [torch.tensor(-0.5, requires_grad=True),
             torch.tensor(-0.3, requires_grad=True)]   # 每步动作的 log π(a|s)
returns_list = [1.98, 1.0]                              # 折扣回报 G_t

log_probs_t = torch.stack(log_probs)                   # (T,)
returns_t   = torch.tensor(returns_list)                # (T,)

# 策略梯度损失：- E[G_t * log π(a_t|s_t)]
# 负号：因为我们用梯度"下降"，但策略梯度是要"上升"
pg_loss = -(log_probs_t * returns_t).mean()

optimizer_p.zero_grad()
pg_loss.backward()
optimizer_p.step()

print(f"Policy Gradient Loss: {pg_loss.item():.4f}")

# ---------- 3.3 Actor-Critic 的组合损失 ----------

ac_net = ActorCritic(state_dim=4, action_dim=2)
optimizer_ac = optim.Adam(ac_net.parameters(), lr=1e-3)

# 一批 transition
states   = torch.randn(32, 4)
actions  = torch.randint(0, 2, (32,))
rewards  = torch.randn(32)
dones    = torch.zeros(32)
next_states = torch.randn(32, 4)

logits, values = ac_net(states)

# Critic 损失：TD 误差（价值网络学习 V(s)）
with torch.no_grad():
    _, next_values = ac_net(next_states)
    td_targets = rewards + gamma * next_values * (1 - dones)

critic_loss = F.mse_loss(values, td_targets)

# Actor 损失：用 TD 误差作为优势函数
advantages = (td_targets - values).detach()           # detach()！不让梯度流回 critic
dist = torch.distributions.Categorical(logits=logits)
log_probs = dist.log_prob(actions)
entropy   = dist.entropy().mean()                      # 熵正则项，鼓励探索

actor_loss = -(log_probs * advantages).mean()

# 总损失 = actor损失 + critic损失 - 熵奖励
# 系数可调：critic_coef 和 entropy_coef 是超参数
total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

optimizer_ac.zero_grad()
total_loss.backward()
torch.nn.utils.clip_grad_norm_(ac_net.parameters(), max_norm=0.5)
optimizer_ac.step()

print(f"Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")

# ---------- 3.4 no_grad 和 detach 的使用场景 ----------

# no_grad：推理时（与环境交互时）不需要梯度，节省内存
state = torch.randn(1, 4)
with torch.no_grad():
    q_values = q_net(state)                # 不建立计算图
    action = q_values.argmax(dim=1).item() # 选择最优动作

# detach()：计算 TD 目标时，target 网络的输出不应该参与梯度计算
# 已在上面 DQN 部分演示（with torch.no_grad() 包裹 target 网络即可）

# ============================================================
# 第四部分：经验回放缓冲区 (Replay Buffer)
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity):
        # deque 是双端队列，超出容量时自动从左侧删除旧数据
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # 存储一条 transition
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 随机采样一批 transition（打破时序相关性）
        batch = random.sample(self.buffer, batch_size)
        # zip(*batch) 将 list of tuples 转置为 tuple of lists
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(np.array(actions),     dtype=torch.long),
            torch.tensor(np.array(rewards),     dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones),       dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

    @property
    def is_ready(self, min_size=1000):
        # 缓冲区足够大才开始训练
        return len(self.buffer) >= min_size

# ============================================================
# 第五部分：DQN 完整实现
# ============================================================

def train_dqn(
    state_dim   = 4,
    action_dim  = 2,
    num_episodes= 200,
    gamma       = 0.99,
    lr          = 1e-3,
    batch_size  = 64,
    buffer_size = 10000,
    target_update_freq = 10,   # 每隔多少个 episode 同步 target 网络
    eps_start   = 1.0,
    eps_end     = 0.05,
    eps_decay   = 0.995,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 在线网络（持续更新）和目标网络（延迟更新，提供稳定的 TD 目标）
    online_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()                         # target 网络不需要训练模式

    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    buffer    = ReplayBuffer(buffer_size)
    epsilon   = eps_start

    for episode in range(num_episodes):
        # 模拟环境交互（实际用 gym.make('CartPole-v1') 等）
        state_np = np.random.randn(state_dim).astype(np.float32)
        ep_reward = 0.0

        for step in range(200):                         # 最多200步
            # ε-贪婪策略：以 ε 概率随机探索，否则选最优动作
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)   # 随机动作
            else:
                state_t = torch.tensor(state_np).unsqueeze(0).to(device)  # (1,4)
                with torch.no_grad():
                    action = online_net(state_t).argmax(dim=1).item()      # 贪婪动作

            # 执行动作，获得下一状态和奖励（模拟）
            next_state_np = np.random.randn(state_dim).astype(np.float32)
            reward = 1.0
            done   = (step == 199)

            # 存入回放缓冲区
            buffer.push(state_np, action, reward, next_state_np, float(done))
            state_np = next_state_np
            ep_reward += reward

            # 等缓冲区积累足够多的数据再开始训练
            if len(buffer) < batch_size:
                continue

            # 从缓冲区采样并训练
            s, a, r, s2, d = buffer.sample(batch_size)
            s, a, r, s2, d = s.to(device), a.to(device), r.to(device), s2.to(device), d.to(device)

            # 计算当前 Q 值
            q_pred = online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

            # 计算 TD 目标（用 target 网络）
            with torch.no_grad():
                q_next = target_net(s2).max(dim=1).values
                td_target = r + gamma * q_next * (1 - d)

            loss = F.smooth_l1_loss(q_pred, td_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
            optimizer.step()

            if done:
                break

        # ε 衰减：随时间减少探索
        epsilon = max(eps_end, epsilon * eps_decay)

        # 定期将在线网络权重复制到目标网络
        if episode % target_update_freq == 0:
            target_net.load_state_dict(online_net.state_dict())

        if (episode + 1) % 20 == 0:
            print(f"Episode {episode+1:4d} | Reward {ep_reward:.1f} | ε {epsilon:.3f}")

    return online_net

# ============================================================
# 第六部分：REINFORCE（蒙特卡洛策略梯度）
# ============================================================

def train_reinforce(
    state_dim   = 4,
    action_dim  = 2,
    num_episodes= 300,
    gamma       = 0.99,
    lr          = 1e-3,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = PolicyNetwork(state_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for episode in range(num_episodes):
        # 收集一整条 episode 的数据（蒙特卡洛方法，必须跑完才能更新）
        log_probs_ep = []
        rewards_ep   = []

        state_np = np.random.randn(state_dim).astype(np.float32)

        for step in range(200):
            state_t = torch.tensor(state_np).unsqueeze(0).to(device)  # (1,4)

            # 从策略网络采样动作
            logits = policy(state_t)                                    # (1,2)
            dist   = torch.distributions.Categorical(logits=logits)
            action = dist.sample()                                      # 采样
            log_prob = dist.log_prob(action)                           # log π(a|s)

            # 执行动作（模拟）
            next_state_np = np.random.randn(state_dim).astype(np.float32)
            reward = 1.0
            done   = (step == 199)

            log_probs_ep.append(log_prob)
            rewards_ep.append(reward)
            state_np = next_state_np

            if done:
                break

        # 计算折扣累积回报 G_t（从后往前）
        returns = []
        G = 0.0
        for r in reversed(rewards_ep):
            G = r + gamma * G
            returns.insert(0, G)               # 插到列表开头

        returns_t = torch.tensor(returns, dtype=torch.float32).to(device)

        # 标准化回报（减均值除标准差，稳定训练）
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # 拼接所有步的 log_prob
        log_probs_t = torch.cat(log_probs_ep)   # (T,)

        # REINFORCE 损失：- 1/T * Σ G_t * log π(a_t|s_t)
        loss = -(log_probs_t * returns_t).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if (episode + 1) % 50 == 0:
            ep_reward = sum(rewards_ep)
            print(f"Episode {episode+1:4d} | Reward {ep_reward:.1f} | Loss {loss.item():.4f}")

    return policy

# ============================================================
# 第七部分：模型保存与加载
# ============================================================

def save_agent(model, optimizer, episode, path='rl_agent.pth'):
    checkpoint = {
        'episode':         episode,
        'model_state':     model.state_dict(),     # 网络权重
        'optimizer_state': optimizer.state_dict(), # 优化器状态（含动量等）
    }
    torch.save(checkpoint, path)
    print(f"模型已保存至 {path}")

def load_agent(model, optimizer, path='rl_agent.pth', device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    start_episode = ckpt['episode']
    print(f"模型已加载，从第 {start_episode} 个 episode 继续训练")
    return start_episode

# Target 网络的权重同步方式：
def hard_update(online_net, target_net):
    # 硬更新：直接复制权重（DQN 标准做法，每隔 N 步同步一次）
    target_net.load_state_dict(online_net.state_dict())

def soft_update(online_net, target_net, tau=0.005):
    # 软更新：θ_target = τ*θ_online + (1-τ)*θ_target（DDPG/SAC 常用）
    for p_online, p_target in zip(online_net.parameters(), target_net.parameters()):
        p_target.data.copy_(tau * p_online.data + (1 - tau) * p_target.data)

# ============================================================
# 第八部分：实用技巧
# ============================================================

# ---------- 8.1 随机种子（可复现实验）----------
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# ---------- 8.2 ε-贪婪的常见写法 ----------
def epsilon_greedy(q_values, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)     # 随机探索
    return q_values.argmax().item()                  # 贪婪利用

# ---------- 8.3 折扣因子对 returns 的影响 ----------
# gamma=1.0：看重远期奖励（适合有限 episode）
# gamma=0.99：稍微偏向近期（CartPole 常用）
# gamma=0.9：更偏短视（奖励密集时可用）

# ---------- 8.4 检查 NaN（训练崩溃的常见原因）----------
def check_nan(model, loss):
    if torch.isnan(loss):
        print("警告：loss 为 NaN，检查学习率或梯度裁剪")
        return True
    for name, p in model.named_parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            print(f"警告：{name} 的梯度为 NaN")
            return True
    return False

# ---------- 8.5 GPU 加速要点 ----------
# 1. 模型和张量必须在同一设备
# 2. 环境交互用 numpy（gym 不支持 tensor），存回放用 numpy，采样后才转 tensor
# 3. 推理时用 torch.no_grad()，避免占用显存
# 4. pin_memory=True 可加速 CPU->GPU 的数据传输（DataLoader 场景）

# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    set_seed(42)

    print("=" * 50)
    print("训练 DQN")
    print("=" * 50)
    trained_q = train_dqn(num_episodes=60)

    print("\n" + "=" * 50)
    print("训练 REINFORCE")
    print("=" * 50)
    trained_policy = train_reinforce(num_episodes=150)
