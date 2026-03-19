# ============================================================
#   PyTorch 速成教程 —— 面向强化学习
#   涵盖：张量基础 → 神经网络 → 自动微分 →
#         经验回放 → DQN → Policy Gradient → Actor-Critic
# ============================================================

# import 语句：导入外部库，让当前文件可以使用它们
import torch                        # Python语法：import <包名>，导入 PyTorch 主库
import torch.nn as nn               # Python语法：import <包名> as <别名>，之后用 nn 代替 torch.nn
import torch.nn.functional as F     # 导入函数式 API（不含参数的层，如 relu/mse_loss）
import torch.optim as optim         # 导入优化器模块（Adam、SGD 等）
import numpy as np                  # 导入 NumPy，别名 np，用于数组操作（gym 环境返回 numpy）
import random                       # Python 标准库：随机数，用于 ε-贪婪策略
from collections import deque       # Python语法：from <包> import <类>，只导入 deque（双端队列）

# ============================================================
# 第一部分：张量基础 —— 用 RL 场景理解
# ============================================================

# ---------- 1.1 RL 中的常见张量 ----------

# torch.tensor(数据, dtype=类型)：把 Python 列表/数组转成 PyTorch 张量
# dtype=torch.float32：指定数据类型为32位浮点（神经网络默认用 float32）
state = torch.tensor([0.01, -0.02, 0.03, -0.04], dtype=torch.float32)

# .shape：张量的属性，返回各维度大小，shape=[4] 表示一维、4个元素
print(state.shape)   # torch.Size([4])

# torch.randn(行, 列)：生成服从标准正态分布的随机张量
# 这里生成 32 行 4 列的矩阵，代表一批32个状态，每个状态4维
batch_states = torch.randn(32, 4)

# dtype=torch.long：64位整数类型，动作索引必须是整数才能用于 gather/index
action = torch.tensor(1, dtype=torch.long)           # 单个标量动作

# torch.randint(最小值, 最大值不含, (形状,))：生成随机整数张量
# 值域 [0, 2)，即 0 或 1，生成32个随机动作
batch_actions = torch.randint(0, 2, (32,))

# torch.tensor(标量)：没有指定 dtype，默认 float32
reward = torch.tensor(1.0)

# torch.randn(32)：生成一维、32个元素的随机浮点张量，代表一批奖励
batch_rewards = torch.randn(32)

# torch.tensor(布尔值)：创建布尔类型标量张量
done = torch.tensor(False)

# dtype=torch.bool：布尔类型，torch.zeros 生成全0张量，全0布尔即全 False
batch_dones = torch.zeros(32, dtype=torch.bool)

# ---------- 1.2 设备管理（CPU / GPU）----------

# torch.cuda.is_available()：检测当前机器是否有可用的 NVIDIA GPU
# Python三元表达式：值A if 条件 else 值B
# 如果有 GPU 用 'cuda'，否则用 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# f-string（格式化字符串）：Python语法，f"...{变量}..."，把变量嵌入字符串
print(f"使用设备: {device}")

# .to(device)：把张量移动到指定设备（CPU 或 GPU），返回新张量
# RL 中所有张量和模型都要放到同一个设备，否则运算会报错
state = state.to(device)
batch_states = batch_states.to(device)

# ---------- 1.3 张量创建 ----------

# torch.zeros(n)：创建全0的一维张量，n 个元素
q_values  = torch.zeros(4)              # 4个动作的初始 Q 值，全部为0
returns_g = torch.zeros(200)            # 存储一条 episode 最多200步的累积回报
advantage = torch.empty(32)            # torch.empty：创建未初始化张量（之后会填充，不需要初始化）

# torch.randn：正态分布随机张量，常用于调试时模拟一批假数据
x = torch.randn(32, 4)

# torch.ones(行, 列)：全1张量
x = torch.ones(32, 1)

# torch.arange(start, end)：类似 Python range，生成 [0,1,...,199] 的整数张量
x = torch.arange(0, 200)

# ---------- 1.4 张量属性 ----------

t = torch.randn(32, 4)
print(t.shape)      # .shape 属性：各维度大小，torch.Size([32, 4])
print(t.ndim)       # .ndim 属性：维度数量，这里是 2（二维矩阵）
print(t.dtype)      # .dtype 属性：数据类型，torch.float32
print(t.device)     # .device 属性：所在设备，cpu 或 cuda:0
print(t.numel())    # .numel() 方法：元素总数，32×4=128

# ---------- 1.5 形状变换 ----------

q_out = torch.randn(1, 2)              # shape: (1, 2)，第0维是 batch（只有1个样本）

# .squeeze(dim)：去掉指定维度（该维度大小必须为1），(1,2) -> (2,)
q_out = q_out.squeeze(0)

logits = torch.randn(4)               # shape: (4,)，一维张量

# .unsqueeze(dim)：在指定位置插入一个大小为1的维度，(4,) -> (1,4)
# 神经网络通常要求输入有 batch 维，所以单个样本要先 unsqueeze
logits = logits.unsqueeze(0)

# Atari 像素输入场景：卷积层输出 (batch, C, H, W)，送入全连接层前要展平
feature_map = torch.randn(32, 64, 7, 7)   # shape: (32, 64, 7, 7)

# .reshape(新形状)：重塑张量维度，-1 表示"自动计算这一维的大小"
# 64*7*7=3136，所以结果是 (32, 3136)
flat = feature_map.reshape(32, -1)

# ---------- 1.6 索引与切片 ----------

batch_states = torch.randn(32, 4)
batch_actions = torch.randint(0, 2, (32,))

q_all = torch.randn(32, 2)   # 所有动作的 Q 值，shape: (32, 2)

# .gather(dim, index)：沿指定维度，按 index 中的值逐行取元素
# dim=1 表示在动作维上取，batch_actions.unsqueeze(1) 把 (32,) 变成 (32,1) 作为 index
# 结果 shape: (32, 1)，每行取出实际执行的那个动作的 Q 值
q_taken = q_all.gather(1, batch_actions.unsqueeze(1))
q_taken = q_taken.squeeze(1)   # (32,1) -> (32,)，去掉多余的维度

batch_dones = torch.zeros(32, dtype=torch.bool)

# 整数索引赋值：Python语法，tensor[索引] = 值，修改指定位置的元素
batch_dones[5] = True        # 把第5个元素设为 True（该 episode 结束）

# ~ 是按位取反运算符，对布尔张量相当于逻辑 NOT：True->False, False->True
mask = ~batch_dones

next_q = torch.randn(32)

# 布尔掩码赋值：tensor[布尔张量] = 值，把所有 True 位置的元素改为指定值
# 终止状态没有"下一步"，所以下一步的 Q 值置0
next_q[batch_dones] = 0.0

# ---------- 1.7 数学运算 ----------

rewards  = torch.randn(32)
next_q   = torch.randn(32)
gamma    = 0.99             # Python 普通变量赋值，gamma 是折扣因子（超参数）

# 张量运算支持广播：标量 * 张量 = 每个元素都乘以标量
# + 是逐元素加法（两个 shape 相同的张量对应位置相加）
td_target = rewards + gamma * next_q

# ** 是幂运算符，(a-b)**2 逐元素求平方差
losses = (td_target - torch.randn(32)) ** 2

# .mean()：对张量所有元素求均值，返回标量张量（才能调用 .backward()）
mean_loss = losses.mean()

# ---------- 计算折扣累积回报（从后往前遍历）----------

T = 10                              # 普通 Python 整数变量
ep_rewards = torch.tensor([1.0] * T)  # [1.0]*T：Python列表重复，生成长度为T的全1列表

returns = torch.zeros(T)            # 存储每步的折扣回报 G_t
G = 0.0                             # 累积回报初始值

# reversed(range(T))：Python内置函数，反向遍历 [T-1, T-2, ..., 0]
# for 循环：Python语法，for 变量 in 可迭代对象: 循环体
for t in reversed(range(T)):
    # .item()：把单元素张量转为 Python 标量（float/int），方便做普通数学运算
    G = ep_rewards[t].item() + gamma * G   # 贝尔曼方程：G_t = r_t + γ * G_{t+1}
    returns[t] = G                         # 把结果存回张量对应位置

# 优势函数标准化：减均值再除标准差，让梯度更新更稳定
advantages = returns - returns.mean()      # .mean() 返回均值标量张量，广播相减

# .std()：标准差；+1e-8：加极小值防止标准差为0时除以0（1e-8 = 0.00000001）
advantages = advantages / (advantages.std() + 1e-8)

# ---------- 1.8 与 NumPy 互转 ----------

# np.array(列表, dtype=类型)：创建 NumPy 数组，gym 环境返回的就是这种格式
np_state = np.array([0.01, -0.02, 0.03, -0.04], dtype=np.float32)

# torch.from_numpy(numpy数组)：零拷贝转换，tensor 和 numpy 共享同一块内存
state_t = torch.from_numpy(np_state)

# torch.tensor(numpy数组)：拷贝数据，两者互不影响（更安全）
state_t = torch.tensor(np_state)

q_values = torch.tensor([0.1, 0.9])

# .argmax()：返回最大值的索引（张量）；.item()：转为 Python int
# 链式调用：Python 中可以连续用点号调用方法
action = q_values.argmax().item()

# torch.softmax(张量, dim=0)：对指定维度做 softmax，把任意实数变成概率分布（和为1）
probs = torch.softmax(torch.randn(4), dim=0)

# .detach()：从计算图中分离，返回不参与梯度计算的张量
# .numpy()：把 tensor 转为 numpy array（必须先 detach，且必须在 CPU 上）
probs_np = probs.detach().numpy()

# np.random.choice(范围, p=概率数组)：按指定概率分布采样一个整数
action = np.random.choice(4, p=probs_np)

# ============================================================
# 第二部分：RL 中的神经网络
# ============================================================

# ---------- 2.1 Q 网络（DQN 用）----------
# 输入：状态 s，输出：每个动作的 Q(s,a)

# class 语句：Python语法，定义一个类；(nn.Module) 表示继承自 nn.Module
# nn.Module 是所有 PyTorch 神经网络的基类，必须继承它
class QNetwork(nn.Module):

    # __init__ 是构造方法，创建类的实例时自动调用
    # self 指向当前对象本身（Python 的约定，必须是第一个参数）
    # 默认参数：hidden_dim=128，调用时不传该参数则使用默认值
    def __init__(self, state_dim, action_dim, hidden_dim=128):

        # super().__init__()：调用父类（nn.Module）的构造方法
        # 必须在 __init__ 开头调用，否则 PyTorch 内部机制无法正确初始化
        super().__init__()

        # self.net：给对象添加一个属性，名字叫 net，存储神经网络层
        # nn.Sequential(层1, 层2, ...)：容器，把多个层按顺序串联，forward 时依次执行
        self.net = nn.Sequential(
            # nn.Linear(输入维度, 输出维度)：全连接线性层，y = xW^T + b
            nn.Linear(state_dim, hidden_dim),   # 输入层：state_dim -> hidden_dim
            # nn.ReLU()：激活函数层，ReLU(x) = max(0, x)，引入非线性
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # 隐藏层
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),  # 输出层：hidden_dim -> action_dim
            # Q 值是任意实数，所以输出层不加激活函数
        )

    # def 定义方法；forward 是 nn.Module 的特殊方法，定义前向传播计算过程
    # 调用 net(输入) 时 PyTorch 会自动调用 forward
    def forward(self, state):
        # self.net(state)：调用 Sequential 容器，数据依次经过所有层
        # 返回 shape: (batch_size, action_dim)，即每个样本的每个动作的 Q 值
        return self.net(state)

# ---------- 2.2 策略网络（Policy Network，离散动作）----------
# 输入：状态，输出：各动作的概率分布（用于 REINFORCE / Actor-Critic）

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            # 输出层不加 softmax，因为 Categorical(logits=...) 内部会用数值稳定的方式处理
        )

    def forward(self, state):
        # 把状态送入神经网络，得到每个动作的 logit（原始分数，未归一化）
        logits = self.net(state)   # shape: (batch, action_dim)
        return logits              # return 语句：从函数返回一个值

    # 这是类的另一个方法，与 forward 并列，用于与环境交互时采样动作
    def get_action(self, state):
        # 调用本对象的 forward 方法（self.forward 也可以直接写 self(state)）
        logits = self.forward(state)

        # torch.distributions.Categorical(logits=...)：创建离散概率分布对象
        # logits= 关键字参数：内部会自动做 log_softmax，数值更稳定
        dist = torch.distributions.Categorical(logits=logits)

        # dist.sample()：按概率分布随机采样一个动作（整数张量）
        action = dist.sample()

        # dist.log_prob(action)：计算该动作在当前分布下的对数概率 log π(a|s)
        # 策略梯度更新时需要这个值来计算梯度
        log_prob = dist.log_prob(action)

        # 返回多个值：Python语法，用逗号分隔，调用方用 a, b = func() 接收
        # action.item()：把整数张量转为 Python int，方便传给 gym 环境
        return action.item(), log_prob

# ---------- 2.3 价值网络（Value Network，用于 Actor-Critic）----------
# 输入：状态，输出：状态价值 V(s)（标量）

class ValueNetwork(nn.Module):
    # 价值网络不需要 action_dim，因为输出是单个标量 V(s)
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),    # 输出维度为1，代表标量状态价值
        )

    def forward(self, state):
        # self.net(state) 输出 shape: (batch, 1)
        # .squeeze(-1)：去掉最后一个维度（大小为1），(batch,1) -> (batch,)
        # -1 表示"最后一个维度"，等价于 .squeeze(1)
        return self.net(state).squeeze(-1)

# ---------- 2.4 Actor-Critic 合并网络（共享特征提取）----------
# 前几层共享，最后分叉为 actor head 和 critic head

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # self.backbone：共享的特征提取网络，actor 和 critic 都用这个提取特征
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # actor_head 和 critic_head 是两个独立的线性层，各自有自己的参数
        # self.actor_head：Actor 的输出头，输出各动作的 logits
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # self.critic_head：Critic 的输出头，输出状态价值标量
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        # 先过共享主干，得到特征向量
        feat = self.backbone(state)

        # 特征向量同时送入两个头，分别计算 logits 和 value
        logits = self.actor_head(feat)
        value  = self.critic_head(feat).squeeze(-1)  # (batch,1) -> (batch,)

        # 同时返回两个值：元组返回，调用方用 logits, value = net(state) 解包
        return logits, value

    def get_action(self, state):
        # 解包赋值：Python语法，左边多个变量同时接收右边返回的多个值
        logits, value = self.forward(state)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # dist.entropy()：计算当前策略分布的熵，熵越大表示越"随机"（探索性强）
        # 训练时把熵加入损失可以防止策略过早收敛
        entropy = dist.entropy()

        # 返回4个值：动作整数、log概率张量、状态价值张量、策略熵张量
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
        # value_stream：只输出1个数（状态价值 V(s)）
        self.value_stream    = nn.Linear(hidden_dim, 1)

        # advantage_stream：输出每个动作的优势 A(s,a)
        self.advantage_stream = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        feat = self.backbone(state)
        V = self.value_stream(feat)      # shape: (batch, 1)
        A = self.advantage_stream(feat)  # shape: (batch, action_dim)

        # A.mean(dim=1, keepdim=True)：沿 dim=1（动作维）求均值
        # keepdim=True：保持维度不变，结果是 (batch,1)，方便广播减法
        # Q = V + A - mean(A)：减去均值是为了保证分解的唯一性
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q

# ---------- 2.6 查看模型 ----------

# 实例化：类名(参数)，调用 __init__，创建一个 QNetwork 对象
q_net = QNetwork(state_dim=4, action_dim=2)   # 关键字参数：参数名=值，顺序可以不同

# print(模型)：nn.Module 重载了 __repr__，会打印出网络结构
print(q_net)

# sum(生成器表达式)：Python内置，对可迭代对象求和
# for p in q_net.parameters()：遍历网络所有参数（权重和偏置的张量）
# p.numel()：每个参数张量的元素数量
# 合起来就是统计网络总参数量
total_params = sum(p.numel() for p in q_net.parameters())

print(f"参数量: {total_params}")   # f-string 格式化输出

# torch.randn(1, 4)：batch=1 的假状态，用于测试网络输出形状
dummy_state = torch.randn(1, 4)

# with torch.no_grad(): 上下文管理器：进入这个块后，所有操作都不建立计算图
# 推理时（不需要梯度）必须加，否则浪费内存
with torch.no_grad():
    out = q_net(dummy_state)   # 调用模型：Python 中 对象(参数) 等价于 对象.__call__(参数)

print(f"Q网络输出形状: {out.shape}")   # 应该是 torch.Size([1, 2])

# ============================================================
# 第三部分：自动微分 —— RL 中的损失计算
# ============================================================

# ---------- 3.1 DQN 的 TD 损失 ----------

# 多变量同行赋值：Python语法，用逗号分隔，右边是普通整数
state_dim, action_dim, batch_size = 4, 2, 32
gamma = 0.99

q_net    = QNetwork(state_dim, action_dim)

# Target 网络：结构和 online 网络相同，但权重延迟更新，提供稳定的 TD 目标
q_target = QNetwork(state_dim, action_dim)

# .state_dict()：返回模型所有参数的字典（参数名 -> 张量）
# .load_state_dict(字典)：把字典中的参数加载到模型中
# 这两句合起来：把 q_net 的权重完整复制到 q_target
q_target.load_state_dict(q_net.state_dict())

# optim.Adam(参数迭代器, lr=学习率)：创建 Adam 优化器
# lr=1e-3 即 lr=0.001，科学计数法
optimizer = optim.Adam(q_net.parameters(), lr=1e-3)

# 准备一批模拟数据（实际中来自 ReplayBuffer）
s  = torch.randn(batch_size, state_dim)              # 当前状态批量
a  = torch.randint(0, action_dim, (batch_size,))     # 执行的动作
r  = torch.randn(batch_size)                         # 获得的奖励
s2 = torch.randn(batch_size, state_dim)              # 下一状态批量
d  = torch.zeros(batch_size, dtype=torch.bool)       # done 标志

# 前向传播：q_net(s) 得到所有动作的 Q 值，再用 gather 取出实际动作的 Q 值
# q_net(s) shape: (32, 2)；a.unsqueeze(1) shape: (32,1)；gather 结果 (32,1)
q_pred = q_net(s)
q_pred = q_pred.gather(1, a.unsqueeze(1)).squeeze(1)   # 链式调用：先 gather 再 squeeze，(32,)

# with torch.no_grad()：target 网络的输出不需要梯度（它不是训练目标）
with torch.no_grad():
    # .max(dim=1)：沿 dim=1 求最大值，返回命名元组 (values, indices)
    # .values：取出最大值部分，shape: (32,)
    q_next = q_target(s2).max(dim=1).values

    # 布尔索引赋值：终止状态没有下一步，Q 值置0
    q_next[d] = 0.0

    # Bellman 方程：TD 目标 = r + γ * max_a' Q(s', a')
    td_target = r + gamma * q_next

# F.mse_loss(预测值, 目标值)：均方误差损失，(pred - target)^2 的均值
# TD 误差的均方，衡量 Q 网络预测有多准
loss = F.mse_loss(q_pred, td_target)
# 也可用 Huber Loss（对大误差更鲁棒）：
# loss = F.smooth_l1_loss(q_pred, td_target)

# optimizer.zero_grad()：清空所有参数的 .grad 属性
# PyTorch 默认会累积梯度，每次 backward 前必须先清零
optimizer.zero_grad()

# loss.backward()：反向传播，自动计算 loss 对所有参数的梯度（存在 .grad 属性中）
loss.backward()

# clip_grad_norm_(参数, max_norm)：梯度裁剪，防止梯度爆炸
# 如果所有参数梯度的 L2 范数超过 max_norm，则等比例缩小
torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)

# optimizer.step()：用梯度更新参数，即 θ = θ - lr * ∇L
optimizer.step()

# .item()：把标量张量转为 Python float，:.4f 是格式化，保留4位小数
print(f"DQN TD Loss: {loss.item():.4f}")

# ---------- 3.2 REINFORCE 的策略梯度损失 ----------

policy_net = PolicyNetwork(state_dim=4, action_dim=2)
optimizer_p = optim.Adam(policy_net.parameters(), lr=1e-3)

# requires_grad=True：告诉 PyTorch 这个张量需要计算梯度
# 模拟收集到的 log π(a_t|s_t)，实际中由网络采样动作时自动计算
log_probs = [torch.tensor(-0.5, requires_grad=True),
             torch.tensor(-0.3, requires_grad=True)]

# 普通 Python 列表，存储每步的折扣回报 G_t
returns_list = [1.98, 1.0]

# torch.stack(列表)：把多个张量沿新维度拼合成一个张量
# 列表中每个是标量张量，stack 后变成 (T,) 的一维张量
log_probs_t = torch.stack(log_probs)

# torch.tensor(Python列表)：把列表转为张量
returns_t   = torch.tensor(returns_list)

# REINFORCE 策略梯度损失：-E[G_t * log π(a_t|s_t)]
# * 是逐元素乘法，.mean() 对所有步求均值，负号是因为优化器做梯度下降（我们想最大化回报）
pg_loss = -(log_probs_t * returns_t).mean()

optimizer_p.zero_grad()  # 清零梯度
pg_loss.backward()       # 反向传播
optimizer_p.step()       # 更新参数

print(f"Policy Gradient Loss: {pg_loss.item():.4f}")

# ---------- 3.3 Actor-Critic 的组合损失 ----------

ac_net = ActorCritic(state_dim=4, action_dim=2)
optimizer_ac = optim.Adam(ac_net.parameters(), lr=1e-3)

# 准备一批 transition 数据（实际中来自环境交互或 ReplayBuffer）
states   = torch.randn(32, 4)
actions  = torch.randint(0, 2, (32,))
rewards  = torch.randn(32)
dones    = torch.zeros(32)           # float 类型的 done，用于后面的 (1-dones) 乘法
next_states = torch.randn(32, 4)

# 解包赋值：forward 返回两个值，用逗号同时接收
logits, values = ac_net(states)

# Critic 损失：让价值网络的预测 V(s) 逼近 TD 目标
with torch.no_grad():
    # 用下一状态的价值估计 TD 目标（暂时只用1步 TD）
    _, next_values = ac_net(next_states)   # _ 是惯例，表示不关心第一个返回值（logits）

    # (1 - dones)：终止状态 done=1，则 (1-1)=0，屏蔽下一步的价值
    td_targets = rewards + gamma * next_values * (1 - dones)

# F.mse_loss：均方误差，让 values（网络预测）靠近 td_targets（目标）
critic_loss = F.mse_loss(values, td_targets)

# .detach()：把张量从计算图中分离，梯度不会通过它反向传播
# 优势函数是常数（标签），不应该让 actor 的梯度影响到 critic 的参数
advantages = (td_targets - values).detach()

# 重新计算 log π(a|s)（这次的 logits 有梯度，用于更新 actor）
dist = torch.distributions.Categorical(logits=logits)

# dist.log_prob(actions)：计算每个样本的动作在当前策略下的 log 概率
log_probs = dist.log_prob(actions)

# dist.entropy()：计算策略熵；.mean()：对 batch 求均值
entropy   = dist.entropy().mean()

# Actor 损失：-E[A * log π(a|s)]
# 负号：优化器做梯度下降，等效于对 E[A * log π] 做梯度上升
actor_loss = -(log_probs * advantages).mean()

# 总损失 = actor损失 + 0.5*critic损失 - 0.01*熵
# 0.5 和 0.01 是超参数（手动设定的系数，不通过梯度学习）
# 减熵：鼓励策略保持多样性，防止过早收敛到次优策略
total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

optimizer_ac.zero_grad()
total_loss.backward()
torch.nn.utils.clip_grad_norm_(ac_net.parameters(), max_norm=0.5)
optimizer_ac.step()

print(f"Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")

# ---------- 3.4 no_grad 和 detach 的使用场景 ----------

state = torch.randn(1, 4)

# with torch.no_grad(): 进入推理模式，不建立计算图，节省内存和计算
with torch.no_grad():
    q_values = q_net(state)                    # 正常前向传播，但不记录梯度
    action = q_values.argmax(dim=1).item()     # argmax 取最大值索引，.item() 转 int

# ============================================================
# 第四部分：经验回放缓冲区 (Replay Buffer)
# ============================================================

# 经验回放：把 (s,a,r,s',done) 存入缓冲区，训练时随机采样，打破时序相关性
class ReplayBuffer:
    # capacity：缓冲区最大容量（超出时自动覆盖最旧的数据）
    def __init__(self, capacity):
        # deque(maxlen=n)：双端队列，超出容量时自动从左侧（最旧）删除元素
        self.buffer = deque(maxlen=capacity)

    # push 方法：存入一条 transition（五元组）
    def push(self, state, action, reward, next_state, done):
        # .append(元素)：在队列右侧添加一个元素（五元组）
        # (state, action, reward, next_state, done) 是 Python 元组（用逗号创建）
        self.buffer.append((state, action, reward, next_state, done))

    # sample 方法：随机采样一批数据，返回分类好的张量
    def sample(self, batch_size):
        # random.sample(序列, k)：不放回地随机抽取 k 个元素，返回列表
        batch = random.sample(self.buffer, batch_size)

        # zip(*batch)：* 是解包运算符，把列表展开为多个参数
        # zip 把多个元组"转置"：[(s1,a1,r1,...), (s2,a2,r2,...)] -> ([s1,s2,...], [a1,a2,...], ...)
        # 解包赋值：同时给5个变量赋值
        states, actions, rewards, next_states, dones = zip(*batch)

        # np.array(列表)：把 Python 列表（内含 numpy 数组）堆叠成一个大 numpy 数组
        # torch.tensor(numpy数组, dtype=类型)：转为指定类型的张量
        # 整个 return 返回一个元组（5个张量），调用方用 s,a,r,s2,d = buffer.sample(n) 接收
        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(np.array(actions),     dtype=torch.long),
            torch.tensor(np.array(rewards),     dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones),       dtype=torch.float32),
        )

    # __len__ 是特殊方法（魔术方法）：当对 ReplayBuffer 对象调用 len() 时触发
    def __len__(self):
        return len(self.buffer)   # 直接返回内部 deque 的长度

    # @property 装饰器：把方法变成属性，调用时不用加括号（写 buf.is_ready 而非 buf.is_ready()）
    @property
    def is_ready(self, min_size=1000):
        # >= 是大于等于运算符；返回布尔值 True 或 False
        return len(self.buffer) >= min_size

# ============================================================
# 第五部分：DQN 完整实现
# ============================================================

# def 定义函数；参数可以有默认值，调用时可以只传部分参数
def train_dqn(
    state_dim   = 4,
    action_dim  = 2,
    num_episodes= 200,
    gamma       = 0.99,
    lr          = 1e-3,          # 1e-3 = 0.001，科学计数法
    batch_size  = 64,
    buffer_size = 10000,
    target_update_freq = 10,     # 每隔多少个 episode 同步 target 网络
    eps_start   = 1.0,           # ε-贪婪的初始探索率（100%随机）
    eps_end     = 0.05,          # ε 的最小值（保留5%随机探索）
    eps_decay   = 0.995,         # 每个 episode 后 ε 乘以这个衰减系数
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 在线网络（online_net）：持续用梯度更新
    # .to(device)：把模型的所有参数张量移动到指定设备
    online_net = QNetwork(state_dim, action_dim).to(device)

    # 目标网络（target_net）：结构相同，但权重延迟更新，提供稳定的 TD 目标
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(online_net.state_dict())   # 初始时权重相同

    # .eval()：切换到评估模式（关闭 Dropout/BatchNorm 的训练行为）
    # target 网络只做推理，不需要训练模式
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    buffer    = ReplayBuffer(buffer_size)   # 创建 ReplayBuffer 实例
    epsilon   = eps_start                  # epsilon 初始值

    # for 循环：range(num_episodes) 生成 [0, 1, ..., num_episodes-1]
    for episode in range(num_episodes):

        # .astype(np.float32)：NumPy 方法，转换数组的数据类型
        state_np = np.random.randn(state_dim).astype(np.float32)
        ep_reward = 0.0

        for step in range(200):   # 每个 episode 最多200步

            # random.random()：生成 [0.0, 1.0) 之间的随机浮点数
            # < epsilon：如果小于 ε，执行随机动作（探索）
            if random.random() < epsilon:
                # random.randint(a, b)：注意！Python 的 randint 是闭区间 [a,b]
                action = random.randint(0, action_dim - 1)
            else:
                # torch.tensor(numpy数组)：转张量；.unsqueeze(0)：加 batch 维 (4,)->(1,4)
                state_t = torch.tensor(state_np).unsqueeze(0).to(device)
                with torch.no_grad():
                    # .argmax(dim=1)：沿 dim=1 取最大值索引，shape (1,)
                    # .item()：转为 Python int，才能传给环境
                    action = online_net(state_t).argmax(dim=1).item()

            # 执行动作，获得转移数据（真实场景用 env.step(action) 代替）
            next_state_np = np.random.randn(state_dim).astype(np.float32)
            reward = 1.0
            done   = (step == 199)   # 布尔表达式：最后一步时 done=True

            # float(done)：把布尔转为浮点数，True->1.0, False->0.0
            buffer.push(state_np, action, reward, next_state_np, float(done))
            state_np = next_state_np   # 更新当前状态
            ep_reward += reward        # += 是累加赋值，等价于 ep_reward = ep_reward + reward

            # len(buffer) < batch_size：缓冲区不够一批，跳过训练
            # continue：跳过本次循环剩余代码，直接进入下一次迭代
            if len(buffer) < batch_size:
                continue

            # 从缓冲区采样一批数据
            s, a, r, s2, d = buffer.sample(batch_size)

            # 把所有张量移动到同一设备（GPU 或 CPU）
            # 多重赋值：右边先全部求值，再赋给左边
            s, a, r, s2, d = s.to(device), a.to(device), r.to(device), s2.to(device), d.to(device)

            # 计算当前 Q 值：取实际执行动作对应的 Q 值
            q_pred = online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

            # 计算 TD 目标
            with torch.no_grad():
                # .max(dim=1).values：沿动作维取最大 Q 值
                q_next = target_net(s2).max(dim=1).values

                # (1 - d)：done=1 时屏蔽下一步价值
                td_target = r + gamma * q_next * (1 - d)

            # Huber Loss（smooth_l1_loss）：误差小时像 MSE，误差大时像 MAE，更鲁棒
            loss = F.smooth_l1_loss(q_pred, td_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
            optimizer.step()

            # if done: break：episode 结束时提前退出内层 for 循环
            if done:
                break

        # ε 衰减：max(最小值, 当前值 * 衰减系数)，确保 ε 不低于 eps_end
        epsilon = max(eps_end, epsilon * eps_decay)

        # % 是取模运算符；episode % target_update_freq == 0：每隔固定步数触发
        if episode % target_update_freq == 0:
            target_net.load_state_dict(online_net.state_dict())   # 同步权重

        # (episode + 1) % 20 == 0：每20个 episode 打印一次进度
        if (episode + 1) % 20 == 0:
            # :4d 格式化：右对齐，占4个字符宽度；.1f：保留1位小数；.3f：保留3位小数
            print(f"Episode {episode+1:4d} | Reward {ep_reward:.1f} | ε {epsilon:.3f}")

    # return 语句：函数结束，返回训练好的网络
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

        # 用列表存储一整条 episode 的数据（REINFORCE 是蒙特卡洛方法，必须跑完整条轨迹）
        log_probs_ep = []   # [] 创建空列表
        rewards_ep   = []

        state_np = np.random.randn(state_dim).astype(np.float32)

        for step in range(200):
            # 把 numpy 状态转为张量，加 batch 维，移到设备
            state_t = torch.tensor(state_np).unsqueeze(0).to(device)

            # 前向传播得到 logits，再创建分布，采样动作
            logits = policy(state_t)
            dist   = torch.distributions.Categorical(logits=logits)
            action = dist.sample()           # 采样动作（整数张量）
            log_prob = dist.log_prob(action) # log π(a|s)，是有梯度的张量

            # 模拟环境返回
            next_state_np = np.random.randn(state_dim).astype(np.float32)
            reward = 1.0
            done   = (step == 199)

            # .append(元素)：Python列表方法，在末尾添加元素
            log_probs_ep.append(log_prob)
            rewards_ep.append(reward)
            state_np = next_state_np

            if done:
                break   # break：立即退出当前 for 循环

        # ---------- episode 结束后，计算折扣回报并更新网络 ----------

        returns = []   # 存储每步的 G_t
        G = 0.0

        # reversed(rewards_ep)：反向遍历列表，从最后一步往前算
        for r in reversed(rewards_ep):
            G = r + gamma * G
            # .insert(0, 值)：在列表索引0处插入元素（相当于在头部插入）
            # 因为是反向遍历，所以每次插到最前面，最终顺序正确
            returns.insert(0, G)

        # torch.tensor(列表, dtype=...)：把 Python float 列表转为 float32 张量
        returns_t = torch.tensor(returns, dtype=torch.float32).to(device)

        # 标准化：(x - mean) / (std + eps)，让梯度更新更稳定
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # torch.cat(列表)：把列表中的张量沿 dim=0 拼接成一个大张量
        # log_probs_ep 中每个元素是 shape (1,) 的张量，cat 后变成 (T,)
        log_probs_t = torch.cat(log_probs_ep)

        # REINFORCE 损失：对所有时间步的 G_t * log π(a_t|s_t) 求均值，取负
        loss = -(log_probs_t * returns_t).mean()

        optimizer.zero_grad()
        loss.backward()
        # max_norm=1.0：梯度裁剪阈值，REINFORCE 通常用较小的值
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if (episode + 1) % 50 == 0:
            # sum(列表)：Python内置函数，对列表所有元素求和
            ep_reward = sum(rewards_ep)
            print(f"Episode {episode+1:4d} | Reward {ep_reward:.1f} | Loss {loss.item():.4f}")

    return policy

# ============================================================
# 第七部分：模型保存与加载
# ============================================================

# path='rl_agent.pth'：默认参数，.pth 是 PyTorch 模型文件的惯用后缀
def save_agent(model, optimizer, episode, path='rl_agent.pth'):
    # 用字典存储所有需要保存的内容：键是字符串，值是数据
    checkpoint = {
        'episode':         episode,
        'model_state':     model.state_dict(),      # 网络的所有权重（参数名->张量的字典）
        'optimizer_state': optimizer.state_dict(),  # 优化器状态（含动量、学习率等）
    }
    # torch.save(对象, 路径)：把 Python 对象序列化（用 pickle）保存到文件
    torch.save(checkpoint, path)
    print(f"模型已保存至 {path}")

# map_location=device：加载时把张量映射到指定设备（用于跨设备加载，如 GPU 训练的模型在 CPU 上加载）
def load_agent(model, optimizer, path='rl_agent.pth', device='cpu'):
    # torch.load(路径, map_location=设备)：从文件反序列化 Python 对象
    ckpt = torch.load(path, map_location=device)

    # 字典访问：字典['键'] 取对应的值
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    start_episode = ckpt['episode']

    print(f"模型已加载，从第 {start_episode} 个 episode 继续训练")
    return start_episode

# Target 网络权重同步的两种方式：

def hard_update(online_net, target_net):
    # 硬更新：直接完整复制权重（DQN 经典做法，每隔 N 步同步一次）
    target_net.load_state_dict(online_net.state_dict())

# tau=0.005：软更新系数，是一个很小的值（接近0），让 target 网络缓慢跟随 online 网络
def soft_update(online_net, target_net, tau=0.005):
    # zip(迭代器A, 迭代器B)：同时遍历两个迭代器，返回对应元素的元组
    for p_online, p_target in zip(online_net.parameters(), target_net.parameters()):
        # p_target.data：直接访问参数张量的数据（绕过梯度追踪）
        # .copy_(张量)：原地操作（in-place），用括号内的值覆盖 p_target.data
        # 软更新公式：θ_target = τ*θ_online + (1-τ)*θ_target
        p_target.data.copy_(tau * p_online.data + (1 - tau) * p_target.data)

# ============================================================
# 第八部分：实用技巧
# ============================================================

# ---------- 8.1 随机种子（可复现实验）----------

# seed=42：默认参数，42是惯例（实际上任意整数都行）
def set_seed(seed=42):
    # torch.manual_seed(n)：设置 PyTorch CPU 随机种子
    torch.manual_seed(seed)

    # torch.cuda.manual_seed_all(n)：设置所有 GPU 的随机种子
    torch.cuda.manual_seed_all(seed)

    # np.random.seed(n)：设置 NumPy 的随机种子
    np.random.seed(seed)

    # random.seed(n)：设置 Python 标准库 random 的种子
    random.seed(seed)

# ---------- 8.2 ε-贪婪的常见写法 ----------

def epsilon_greedy(q_values, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)   # 随机探索：返回随机整数动作
    return q_values.argmax().item()               # 贪婪利用：返回 Q 值最大的动作

# ---------- 8.3 折扣因子对 returns 的影响 ----------
# gamma=1.0：看重远期奖励（适合有限 episode）
# gamma=0.99：稍微偏向近期（CartPole 常用）
# gamma=0.9：更偏短视（奖励密集时可用）

# ---------- 8.4 检查 NaN（训练崩溃的常见原因）----------

def check_nan(model, loss):
    # torch.isnan(张量)：返回同形状的布尔张量，NaN 位置为 True
    if torch.isnan(loss):
        print("警告：loss 为 NaN，检查学习率或梯度裁剪")
        return True   # 提前返回 True

    # model.named_parameters()：迭代返回 (参数名字符串, 参数张量) 的元组
    for name, p in model.named_parameters():
        # p.grad：参数的梯度张量（未调用 backward 时为 None）
        # is not None：Python语法，检查对象不是 None
        if p.grad is not None and torch.isnan(p.grad).any():
            # .any()：只要有一个 True 就返回 True
            print(f"警告：{name} 的梯度为 NaN")
            return True
    return False   # 没有 NaN，返回 False

# ---------- 8.5 GPU 加速要点 ----------
# 1. 模型和张量必须在同一设备（都是 cpu 或都是 cuda:0）
# 2. 环境交互用 numpy（gym 不支持 tensor），存回放用 numpy，采样后才转 tensor
# 3. 推理时用 torch.no_grad()，避免占用显存
# 4. pin_memory=True 可加速 CPU->GPU 的数据传输（DataLoader 场景）

# ============================================================
# 主程序
# ============================================================

# if __name__ == '__main__': Python 惯用法
# 当这个文件被"直接运行"时（python xxx.py），__name__ 等于 '__main__'，条件为真执行
# 当这个文件被其他文件 import 时，__name__ 等于文件名，条件为假，不执行
if __name__ == '__main__':
    set_seed(42)   # 调用函数：函数名(参数)，设置所有随机种子

    # "=" * 50：字符串乘法，把 "=" 重复50次，生成分隔线
    print("=" * 50)
    print("训练 DQN")
    print("=" * 50)

    # 调用 train_dqn 函数，只传 num_episodes 参数，其余用默认值
    # 返回值（训练好的网络）赋给 trained_q
    trained_q = train_dqn(num_episodes=60)

    # "\n"：转义字符，代表换行；+ 是字符串拼接
    print("\n" + "=" * 50)
    print("训练 REINFORCE")
    print("=" * 50)

    trained_policy = train_reinforce(num_episodes=150)