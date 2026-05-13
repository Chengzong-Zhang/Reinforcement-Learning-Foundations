# =============================================================================
# PyTorch Tutorial — CS 185/285 Deep RL (Section 1)
# 覆盖内容：Tensors / Autograd / Modules / Training Loop /
#           Data+Eval / Practical Recipes / Modern PyTorch
# 运行环境：Python 3.10+，PyTorch 2.x，有无 GPU 均可
# =============================================================================

import torch                        # PyTorch 主库，几乎所有功能都从这里来
import torch.nn as nn               # nn = neural network，包含层、损失函数等
import torch.nn.functional as F     # 无状态的函数版本（relu、softmax 等）
import torch.optim as optim         # 优化器（SGD、Adam、AdamW 等）
from torch.utils.data import Dataset, DataLoader  # 数据加载工具


# =============================================================================
# Part 1 — Tensors（张量）
# 核心数据结构：带 GPU 支持 + autograd 元数据的 n 维数组
# 关键属性：shape / dtype / device / requires_grad / strides
# =============================================================================
print("=" * 60)
print("Part 1 — Tensors")
print("=" * 60)

# ── 1.1 随机种子（Randomness control）────────────────────────────────────────
# torch.manual_seed(n)：固定随机种子，保证每次运行结果相同，方便调试复现
torch.manual_seed(0)

# ── 1.2 常用构造函数（Common constructors）───────────────────────────────────

# torch.zeros(行, 列) → 全零张量，默认 dtype=float32，device=CPU
x = torch.zeros(3, 4)
print("torch.zeros(3,4):\n", x)          # shape=(3,4)，所有元素为 0.0

# torch.ones(行, 列) → 全一张量，同样默认 float32
o = torch.ones(3, 4)
print("torch.ones(3,4):\n", o)

# torch.randn(size) → 标准正态分布随机张量，均值0方差1
# device='cuda' 表示直接在 GPU 上创建，省去后续 .to(device) 的拷贝
# 注：若无 GPU 此行会报错，演示时用 CPU
r = torch.randn(10)                      # shape=(10,)，在 CPU 上
print("torch.randn(10):", r)

# torch.arange(start, end, step) → 等差序列，类似 Python range()
ar = torch.arange(0, 10, 2)             # [0, 2, 4, 6, 8]，dtype=int64
print("torch.arange(0,10,2):", ar)

# torch.linspace(start, end, steps) → 等间距浮点序列（包含两端）
ls = torch.linspace(0, 1, 5)            # [0.00, 0.25, 0.50, 0.75, 1.00]
print("torch.linspace(0,1,5):", ls)

# torch.eye(n) → n×n 单位矩阵（对角线为 1，其余为 0）
e = torch.eye(3)
print("torch.eye(3):\n", e)

# ── 1.3 从 Python / NumPy 创建张量 ──────────────────────────────────────────

# torch.tensor(data) → 总是复制数据（copy），data 可以是列表/元组/numpy array
a = torch.tensor([1, 2, 3])             # dtype 由数据自动推断：这里是 int64
print("torch.tensor([1,2,3]):", a, "dtype:", a.dtype)

# torch.tensor 也可以指定 dtype
a_f = torch.tensor([1, 2, 3], dtype=torch.float32)
print("torch.tensor with float32:", a_f, "dtype:", a_f.dtype)

# torch.as_tensor(data) → 尽量避免拷贝（zero-copy view，仅限 CPU numpy array）
# 如果 data 已经是 tensor，直接共享内存；适合性能敏感场景
b = torch.as_tensor(a)                  # 与 a 共享内存（CPU 且 dtype 相同时）
print("torch.as_tensor:", b)

# ── 1.4 控制 dtype（数据类型）───────────────────────────────────────────────
# 默认：浮点 → float32；整数 → int64
# 常用 dtype：torch.float32 / torch.float64 / torch.int64 / torch.bool

w = torch.ones(5, dtype=torch.float64)  # 显式指定 float64（双精度）
print("float64 ones:", w, "dtype:", w.dtype)

# .to(dtype) → 转换数据类型（返回新张量）
w32 = w.to(torch.float32)               # float64 → float32
print("converted to float32:", w32.dtype)

# ── 1.5 查看张量属性 ─────────────────────────────────────────────────────────
x = torch.randn(4, 6)                   # shape=(4,6) 的随机张量

print("\n--- Tensor attributes ---")
print("x.shape :", x.shape)             # torch.Size([4, 6])，表示形状
print("x.dtype :", x.dtype)             # torch.float32
print("x.device:", x.device)           # device(type='cpu')
print("x.ndim  :", x.ndim)             # 维度数 = 2
print("x.numel():", x.numel())         # 元素总数 = 4*6 = 24
print("x.requires_grad:", x.requires_grad)  # False（默认不追踪梯度）

# ── 1.6 索引与切片（Indexing & Slicing）─────────────────────────────────────
# 语法与 NumPy 完全一致

print("\n--- Indexing ---")
print("x[0]   :", x[0].shape)          # 第 0 行，shape=(6,)
print("x[-1]  :", x[-1].shape)         # 最后一行，shape=(6,)
print("x[0, 2]:", x[0, 2])             # 标量：第0行第2列的元素
print("x[:, 2:5]:", x[:, 2:5].shape)   # 所有行，第2~4列，shape=(4,3)
print("x[1:3, :]:", x[1:3, :].shape)   # 第1~2行，shape=(2,6)

# 布尔索引（Boolean indexing）
mask = x > 0                            # 生成同 shape 的 bool 张量
print("x[x>0] shape:", x[mask].shape)  # 取所有正数，结果是 1D 张量

# 花式索引（Fancy indexing）——用整数列表选行
rows = torch.tensor([0, 2])            # 选第 0 和第 2 行
print("x[[0,2]] shape:", x[rows].shape)  # shape=(2,6)

# ── 1.6b torch.gather：沿某维度按索引逐行/逐列取值 ──────────────────────────
# 问题背景：花式索引只能"整行/整列"选取。
# 但在 RL 中，我们的网络输出 Q(s, a_0), Q(s, a_1), ..., Q(s, a_n)（所有动作的Q值），
# 而我们只需要取出"当时执行的那个动作"对应的 Q 值，即逐行按不同列索引取一个元素。
# torch.gather 正是为这个场景设计的。
#
# 语法：torch.gather(input, dim, index)
#   input : 源张量，shape=(B, N)
#   dim   : 沿哪个维度取值（dim=1 表示沿列方向，即在每行内部选列）
#   index : 索引张量，shape 必须与输出 shape 完全一致
#           index[i][j] 表示：在 input 的第 i 行，取第 index[i][j] 列的元素
#   输出  : 与 index 同 shape
#
# 直觉理解：
#   output[i][j] = input[i][ index[i][j] ]   （dim=1 时）
#   output[i][j] = input[ index[i][j] ][j]   （dim=0 时）

print("\n--- torch.gather ---")

# 构造一个 Q 值矩阵：batch_size=4，num_actions=6
# q_values[i] 是第 i 个样本在所有动作上的 Q 值
q_values = torch.tensor([
    [0.1, 0.5, 0.3, 0.9, 0.2, 0.7],   # 样本 0：各动作 Q 值
    [0.4, 0.2, 0.8, 0.1, 0.6, 0.3],   # 样本 1
    [0.7, 0.1, 0.4, 0.5, 0.3, 0.9],   # 样本 2
    [0.2, 0.8, 0.6, 0.3, 0.9, 0.1],   # 样本 3
])  # shape=(4, 6)

# 每个样本实际执行的动作（从 replay buffer 取出）
actions = torch.tensor([3, 2, 5, 4])  # shape=(4,)，动作 id

# 目标：取出 q_values[0][3], q_values[1][2], q_values[2][5], q_values[3][4]
# 即 Q(s_0, a=3), Q(s_1, a=2), Q(s_2, a=5), Q(s_3, a=4)

# 第一步：index 必须与输出 shape 一致，这里输出是 (4,1)，所以 index 也要是 (4,1)
# actions 原本是 (4,)，需要 unsqueeze(1) 变成 (4,1)
index = actions.unsqueeze(1)           # (4,) → (4,1)
print("index shape:", index.shape)     # torch.Size([4, 1])

# 第二步：调用 gather，dim=1 表示在列方向索引（每行挑一个列元素）
q_selected = torch.gather(q_values, dim=1, index=index)  # shape=(4,1)
print("q_selected:\n", q_selected)
# 验证：q_values[0][3]=0.9, [1][2]=0.8, [2][5]=0.9, [3][4]=0.9

# 通常会 squeeze 掉多余的维度，得到 (B,) 形状
q_selected = q_selected.squeeze(1)    # (4,1) → (4,)
print("q_selected (squeezed):", q_selected)  # tensor([0.9, 0.8, 0.9, 0.9])

# 对比：不用 gather 的等价写法（仅适合 1D 情况，不推荐在批量训练中使用）
q_manual = q_values[torch.arange(4), actions]  # 花式索引：每行取对应列
print("等价花式索引结果:      ", q_manual)     # 两者结果相同

# ── 1.7 形状变换（Reshape & View）───────────────────────────────────────────

print("\n--- Reshape ---")
# .reshape(new_shape) → 返回相同数据的新形状张量
# 如果内存连续则是 view（不拷贝），否则拷贝
y = x.reshape(2, 2, 6)                 # (4,6) → (2,2,6)，元素总数不变
print("x.reshape(2,2,6):", y.shape)

# .view(new_shape) → 必须内存连续，否则报错；不拷贝数据
v = x.view(24)                          # (4,6) → (24,)，展平成 1D
print("x.view(24):", v.shape)

# .flatten() → 展平成 1D，等价于 reshape(-1)
flat = x.flatten()
print("x.flatten():", flat.shape)       # shape=(24,)

# .unsqueeze(dim) → 在指定维度插入大小为 1 的新维度
u = x.unsqueeze(0)                      # (4,6) → (1,4,6)
print("x.unsqueeze(0):", u.shape)

u2 = x.unsqueeze(-1)                    # (4,6) → (4,6,1)
print("x.unsqueeze(-1):", u2.shape)

# .squeeze(dim) → 删除大小为 1 的维度
s = u.squeeze(0)                        # (1,4,6) → (4,6)
print("u.squeeze(0):", s.shape)

# .permute(dims) → 维度重排（类似 NumPy transpose）
p = x.permute(1, 0)                     # (4,6) → (6,4)
print("x.permute(1,0):", p.shape)

# ── 1.8 广播（Broadcasting）& 形状检查 ──────────────────────────────────────
# 广播规则：从右到左对齐维度，大小为1或缺失的维度可自动扩展
# 广播很强大，但也是 #1 静默 bug 来源！

print("\n--- Broadcasting ---")
B, A = 4, 6
logits = torch.randn(B, A)              # shape=(4,6)，一个 batch 的 logits
bias   = torch.randn(A)                 # shape=(6,)，一个偏置向量

# 直接相加：bias 自动从 (6,) 广播到 (4,6)
result = logits + bias                  # shape=(4,6)
print("logits + bias:", result.shape)

# 更推荐的写法：先显式 reshape，避免隐式广播导致的 bug
bias_2d = bias.unsqueeze(0)             # (6,) → (1,6)，意图更清晰
result2 = logits + bias_2d             # (4,6) + (1,6) → (4,6)
print("logits + bias.unsqueeze(0):", result2.shape)

# 常见 bug：(B,) vs (B,1) 的区别
rewards = torch.randn(B)               # shape=(4,)  ← 1D 向量
values  = torch.randn(B, 1)           # shape=(4,1) ← 列向量
# rewards + values 会广播成 (4,4)！这通常不是你想要的
# 正确写法：rewards.unsqueeze(1) + values → (4,1)
print("rewards.unsqueeze(1) + values:", rewards.unsqueeze(1).shape)

# 建议：在关键位置用 assert 检查 shape，尽早发现 bug
assert logits.shape == (B, A), f"期望 ({B},{A})，实际 {logits.shape}"

# ── 1.9 Device 管理（CPU vs GPU）────────────────────────────────────────────
print("\n--- Device placement ---")

# 标准写法：自动检测 GPU，没有就用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("使用设备:", device)

# .to(device) → 把张量/模型移动到指定设备（返回新张量，原张量不变）
t = torch.randn(3, 4)                   # 默认在 CPU
t_dev = t.to(device)                   # 移到 device（有GPU就移到GPU）
print("tensor device:", t_dev.device)

# non_blocking=True → 异步传输，CPU 不等待 GPU 完成，可提升吞吐量
# 只有在 DataLoader 使用 pin_memory=True 时才有效
# t.to(device, non_blocking=True)

# ── 1.10 常用数学操作 ────────────────────────────────────────────────────────
print("\n--- Math ops ---")
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print("a + b   :", a + b)               # 逐元素加法（等价于 torch.add）
print("a * b   :", a * b)               # 逐元素乘法（等价于 torch.mul）
print("a @ b   :", a @ b)               # 向量点积（等价于 torch.dot）
print("a.sum() :", a.sum())             # 求和，返回标量 tensor
print("a.mean():", a.mean())            # 均值
print("a.max() :", a.max())             # 最大值

m1 = torch.randn(3, 4)
m2 = torch.randn(4, 5)
print("matmul:", (m1 @ m2).shape)       # 矩阵乘法，shape=(3,5)

# torch.stack vs torch.cat
t1 = torch.randn(3, 4)
t2 = torch.randn(3, 4)
stacked = torch.stack([t1, t2], dim=0)  # 新建维度，shape=(2,3,4)
catted  = torch.cat([t1, t2], dim=0)    # 在已有维度拼接，shape=(6,4)
print("stack shape:", stacked.shape)
print("cat shape  :", catted.shape)


# =============================================================================
# Part 2 — Autograd（自动微分）
# 核心概念：计算图 / requires_grad / .backward() / .grad
# PyTorch 在前向传播时动态构建计算图，反向传播时自动求导
# =============================================================================
print("\n" + "=" * 60)
print("Part 2 — Autograd")
print("=" * 60)

# ── 2.1 requires_grad：告诉 PyTorch "需要对这个张量求梯度" ──────────────────
# 只有 requires_grad=True 的叶子节点才会累积梯度到 .grad 属性

w = torch.tensor(2.0, requires_grad=True)   # 叶子节点，requires_grad=True
b = torch.tensor(1.0, requires_grad=True)   # 叶子节点
x = torch.tensor(3.0)                       # 输入数据，通常不求梯度

# ── 2.2 前向传播（forward pass）构建计算图 ──────────────────────────────────
y = w * x + b                               # y = wx + b = 2*3 + 1 = 7
# y 是中间节点，有 grad_fn（记录是哪个操作产生的）
print("y:", y.item(), "  grad_fn:", y.grad_fn)  # <AddBackward0>

loss = (y - 5.0) ** 2                      # loss = (7-5)^2 = 4
print("loss:", loss.item(), "  grad_fn:", loss.grad_fn)  # <PowBackward0>

# ── 2.3 反向传播（backward pass）自动计算梯度 ────────────────────────────────
# .backward() → 从 loss 出发，沿计算图反向传播，计算所有叶子节点的梯度
# 数学：d(loss)/dw = 2*(y-5)*x = 2*2*3 = 12
#       d(loss)/db = 2*(y-5)*1 = 2*2*1 = 4
loss.backward()

# .grad 属性保存累积的梯度（只有 requires_grad=True 的叶子节点才有）
print("w.grad:", w.grad)    # tensor(12.)
print("b.grad:", b.grad)    # tensor(4.)

# ── 2.4 梯度累积陷阱：.grad 是累积的！────────────────────────────────────────
# 如果不清零，下次 backward() 会把新梯度加到旧梯度上
# 在训练循环中必须手动清零（或用 optimizer.zero_grad()）
assert w.grad is not None and b.grad is not None  # backward 后 grad 必不为 None
w.grad.zero_()              # 就地清零（in-place，下划线结尾代表就地操作）
b.grad.zero_()

# ── 2.5 torch.no_grad()：关闭梯度追踪 ──────────────────────────────────────
# 推理/评估时不需要梯度，用 no_grad 可以：
# 1. 节省内存（不存储中间激活值）
# 2. 加快计算速度
with torch.no_grad():
    y_infer = w * x + b     # 在此块内的操作不建立计算图
    print("y_infer.requires_grad:", y_infer.requires_grad)  # False

# ── 2.6 .detach()：从计算图中分离张量 ──────────────────────────────────────
# .detach() → 返回与原张量共享数据但不追踪梯度的新张量
# 常用场景：把 tensor 转成 numpy，或者阻止梯度流过某个分支
w2 = w.detach()
print("detached requires_grad:", w2.requires_grad)  # False
# 转成 numpy 必须先 detach（并移到 CPU）
arr = w2.numpy()
print("numpy value:", arr)

# ── 2.7 torch.inference_mode()：更强的推理模式（PyTorch 1.9+）──────────────
# 比 no_grad 更激进的优化，彻底禁止梯度相关操作，推理时首选
with torch.inference_mode():
    y_fast = w * x + b
    print("inference_mode requires_grad:", y_fast.requires_grad)  # False

# ── 2.8 叶子节点 vs 非叶子节点 ──────────────────────────────────────────────
# 叶子节点：用户直接创建的张量（不是运算结果），梯度存在 .grad
# 非叶子节点：运算结果，梯度在 backward 后被释放（不存储）
print("w is leaf:", w.is_leaf)    # True
print("y is leaf:", y.is_leaf)    # False（y = w*x+b 是运算结果）

# retain_grad()：强制保留非叶子节点的梯度（调试用）
w3 = torch.tensor(2.0, requires_grad=True)
b3 = torch.tensor(1.0, requires_grad=True)
x3 = torch.tensor(3.0)
y3 = w3 * x3 + b3
y3.retain_grad()                  # 保留 y3 的梯度
loss3 = y3 ** 2
loss3.backward()
print("y3.grad:", y3.grad)        # 现在可以访问中间节点的梯度了


# =============================================================================
# Part 3 — Modules（模块/网络）
# nn.Module：PyTorch 神经网络的基类
# 封装参数（parameters）、缓冲区（buffers）和子模块
# =============================================================================
print("\n" + "=" * 60)
print("Part 3 — Modules")
print("=" * 60)

# ── 3.1 用 nn.Sequential 快速搭建网络 ───────────────────────────────────────
# nn.Sequential：按顺序组合多个层，适合简单的前馈网络

simple_net = nn.Sequential(
    nn.Linear(8, 64),       # 全连接层：输入维度 8，输出维度 64
                            # 自动创建权重矩阵 W(64×8) 和偏置 b(64,)
    nn.ReLU(),              # 激活函数：ReLU(x) = max(0, x)，不含可学习参数
    nn.Linear(64, 64),      # 第二层全连接：64 → 64
    nn.ReLU(),
    nn.Linear(64, 2),       # 输出层：64 → 2（比如 2 个动作的 Q 值）
)
print("simple_net:\n", simple_net)

# ── 3.2 继承 nn.Module 自定义网络 ───────────────────────────────────────────
# 更灵活，可以自定义 forward 逻辑（跳跃连接、多头输出等）

class PolicyNet(nn.Module):
    """
    策略网络：输入状态 obs，输出动作概率分布的 logits
    继承 nn.Module 的三个要点：
    1. __init__ 里调用 super().__init__()
    2. __init__ 里定义所有子层（它们的参数会被自动注册）
    3. 实现 forward(self, x) 定义数据流
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()              # 必须调用，初始化 nn.Module 内部状态
        # nn.Linear(in, out) → y = xW^T + b，W 和 b 是可学习参数
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward 定义了从输入到输出的计算逻辑
        # F.relu 是函数式接口（无参数），等价于 nn.ReLU()
        x = F.relu(self.fc1(x))        # (B, obs_dim) → (B, hidden)
        x = F.relu(self.fc2(x))        # (B, hidden)  → (B, hidden)
        return self.out(x)             # (B, hidden)  → (B, act_dim)

# 实例化网络
policy = PolicyNet(obs_dim=8, act_dim=4)
print("\nPolicyNet:\n", policy)

# ── 3.3 查看参数 ─────────────────────────────────────────────────────────────
# .parameters() → 迭代器，返回所有可学习参数（requires_grad=True 的张量）
total_params = sum(p.numel() for p in policy.parameters())
print(f"\n总参数量: {total_params}")      # numel() = 元素总数

# .named_parameters() → 同时返回参数名和参数值
for name, param in policy.named_parameters():
    print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

# ── 3.4 Buffers（缓冲区）vs Parameters ──────────────────────────────────────
# Parameters：需要梯度，被 optimizer 更新（如权重 W、偏置 b）
# Buffers：不需要梯度，但属于模型状态（如 BatchNorm 的 running_mean）
# 用 self.register_buffer('name', tensor) 注册

class NetWithBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        # register_buffer：这个张量会随模型一起保存/加载，但不会被 optimizer 更新
        self.register_buffer('running_mean', torch.zeros(4))

    def forward(self, x):
        self.running_mean = 0.9 * self.running_mean + 0.1 * x.mean(0).detach()
        return self.fc(x)

net_buf = NetWithBuffer()
print("\nBuffers:", dict(net_buf.named_buffers()))

# ── 3.5 state_dict：保存和加载模型 ──────────────────────────────────────────
# state_dict() → 返回有序字典，包含所有参数和缓冲区的当前值
sd = policy.state_dict()
print("\nstate_dict keys:", list(sd.keys()))

# 保存到磁盘（实际使用时）
# torch.save(policy.state_dict(), 'policy.pt')

# 加载（实际使用时）
# policy.load_state_dict(torch.load('policy.pt'))

# ── 3.6 Target Network 更新（Hard Update & Soft Update）─────────────────────
# DQN 中有两个网络：当前网络（online）和目标网络（target）
# 目标网络用来计算稳定的 TD 目标值，避免"追着自己的尾巴跑"导致训练震荡
# 两种更新方式：Hard Update（直接复制）和 Soft Update（缓慢混合）

# 准备两个结构相同的网络来演示
online_net = PolicyNet(obs_dim=8, act_dim=4)   # 当前网络（被 optimizer 更新）
target_net = PolicyNet(obs_dim=8, act_dim=4)   # 目标网络（不被 optimizer 直接更新）

# --- Hard Update ---
# 每隔固定步数，把 online_net 的参数完整复制给 target_net
# 相当于：θ_target ← θ_online
# 用到的函数：
#   .state_dict()       → 返回模型所有参数和 buffer 的字典（快照）
#   .load_state_dict()  → 把字典里的参数加载进模型（覆盖当前参数）

def hard_update(online: nn.Module, target: nn.Module):
    target.load_state_dict(online.state_dict())  # 直接完整覆盖

hard_update(online_net, target_net)
print("\n--- Hard Update ---")
print("online fc1.weight[0,:3]:", online_net.fc1.weight[0, :3].detach())
print("target fc1.weight[0,:3]:", target_net.fc1.weight[0, :3].detach())
# 两者现在完全相同

# --- Soft Update ---
# 每步用小比例 τ 把 online_net 参数混入 target_net
# 公式：θ_target ← τ·θ_online + (1-τ)·θ_target
# 用到的函数：
#   .named_parameters() → 同时返回参数名和参数张量的迭代器
#   .param.data         → 直接访问参数的底层数据（跳过 autograd，可就地修改）
#   注意：必须用 .data 操作，否则 in-place 修改会破坏计算图

def soft_update(online: nn.Module, target: nn.Module, tau: float = 0.005):
    for (_, online_param), (_, target_param) in zip(
        online.named_parameters(), target.named_parameters()
    ):
        # θ_target ← τ·θ_online + (1-τ)·θ_target
        target_param.data.copy_(
            tau * online_param.data + (1 - tau) * target_param.data
        )

# 先把 target_net 的参数改成全零，方便观察混合效果
for p in target_net.parameters():
    p.data.fill_(0.0)

print("\n--- Soft Update (tau=0.1, before) ---")
print("online fc1.weight[0,:3]:", online_net.fc1.weight[0, :3].detach())
print("target fc1.weight[0,:3]:", target_net.fc1.weight[0, :3].detach())  # 全零

soft_update(online_net, target_net, tau=0.1)

print("--- Soft Update (tau=0.1, after) ---")
print("target fc1.weight[0,:3]:", target_net.fc1.weight[0, :3].detach())
# target 参数 ≈ 0.1 * online + 0.9 * 0 = 0.1 * online

# ── 3.7（原 3.6）前向传播（调用网络） ────────────────────────────────────────
# 调用 net(x) 等价于调用 net.forward(x)，但前者会触发钩子（hooks）
batch_obs = torch.randn(16, 8)           # 16 个样本，每个观测维度为 8
logits = policy(batch_obs)               # shape=(16,4)
print("\n前向传播输出 shape:", logits.shape)

# ── 3.7 常用层速查 ───────────────────────────────────────────────────────────
# nn.Linear(in, out)                  → 全连接层
# nn.Conv2d(in_ch, out_ch, k)         → 2D 卷积
# nn.Embedding(vocab, dim)            → 词嵌入（离散 → 连续）
# nn.LSTM(in, hidden)                 → 长短期记忆
# nn.BatchNorm1d(features)            → 批归一化（训练/推理行为不同）
# nn.Dropout(p)                       → 随机丢弃（训练时激活，推理时关闭）
# nn.LayerNorm(shape)                 → 层归一化（Transformer 中常用）


# =============================================================================
# Part 4 — Training Loop（训练循环）
# 标准流程：zero_grad → forward → loss → backward → clip → step
# =============================================================================
print("\n" + "=" * 60)
print("Part 4 — Training Loop")
print("=" * 60)

# ── 4.1 损失函数（Loss functions）───────────────────────────────────────────
# nn.MSELoss()         → 均方误差，回归任务（如 DQN 的 Q 值回归）
# nn.CrossEntropyLoss()→ 交叉熵，分类任务（内含 softmax，输入是 logits）
# nn.BCEWithLogitsLoss()→ 二分类交叉熵

criterion = nn.MSELoss()               # 创建损失函数对象

# ── 4.2 优化器（Optimizer）──────────────────────────────────────────────────
# optim.SGD(params, lr)              → 随机梯度下降
# optim.Adam(params, lr)             → Adam（自适应学习率）
# optim.AdamW(params, lr, wd)        → Adam + 权重衰减（推荐用于现代模型）
#   weight_decay：L2 正则化系数，防止过拟合

optimizer = optim.AdamW(
    policy.parameters(),    # 传入模型的所有可学习参数
    lr=3e-4,                # 学习率（learning rate）：每步更新的步长
    weight_decay=1e-4,      # 权重衰减（L2 正则）：防止参数过大
)

# ── 4.3 学习率调度器（LR Scheduler）─────────────────────────────────────────
# 训练过程中动态调整学习率，帮助收敛

# CosineAnnealingLR：学习率按余弦曲线从 lr 降到 0，再升回来
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,              # 要调整的优化器
    T_max=100,              # 一个余弦周期的步数
)

# StepLR：每隔 step_size 个 epoch，学习率乘以 gamma
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# ── 4.4 完整训练循环 ─────────────────────────────────────────────────────────
print("\n--- Mini training demo ---")
policy.train()                          # 切换到训练模式（影响 Dropout/BatchNorm）

for step in range(5):                   # 演示用，只跑 5 步
    # Step 1: 生成假数据（实际中从 DataLoader 取数据）
    obs    = torch.randn(16, 8)         # (B, obs_dim)
    target = torch.randn(16, 4)         # (B, act_dim)，假设是回归目标

    # Step 2: 清零梯度（必须！否则梯度会累积）
    # 原因：PyTorch 默认累积梯度，每次 backward 之前必须手动清零
    optimizer.zero_grad()               # 等价于对每个参数 p.grad.zero_()

    # Step 3: 前向传播（forward pass）
    pred = policy(obs)                  # shape=(16,4)

    # Step 4: 计算损失
    loss = criterion(pred, target)      # 标量 tensor，MSE loss

    # Step 5: 反向传播（backward pass）——计算所有参数的梯度
    loss.backward()                     # 填充所有 param.grad

    # Step 6: 梯度裁剪（Gradient Clipping）———防止梯度爆炸
    # max_norm：梯度向量的最大 L2 范数，超过则等比缩小
    # RL 中非常重要，因为奖励信号可能引起梯度突变
    torch.nn.utils.clip_grad_norm_(
        policy.parameters(),            # 要裁剪的参数
        max_norm=0.5,                   # 梯度范数上限
    )

    # Step 7: 参数更新（optimizer step）
    optimizer.step()                    # 用梯度更新参数：θ ← θ - lr * grad

    # Step 8: 更新学习率（在每个 step/epoch 后调用）
    scheduler.step()

    # Step 9: 记录日志（detach 后才能转为 Python 数值）
    current_lr = scheduler.get_last_lr()[0]  # 获取当前学习率
    print(f"  step {step}: loss={loss.item():.4f}, lr={current_lr:.6f}")
    # .item() → 把标量 tensor 转成 Python float（脱离计算图）


# =============================================================================
# Part 5 — Data + Eval（数据加载与评估）
# Dataset / DataLoader / model.train() / model.eval() / inference_mode
# =============================================================================
print("\n" + "=" * 60)
print("Part 5 — Data + Eval")
print("=" * 60)

# ── 5.1 自定义 Dataset ───────────────────────────────────────────────────────
# 继承 Dataset，实现 __len__ 和 __getitem__ 两个方法

class RLReplayBuffer(Dataset):
    """
    简化版 Replay Buffer，用于 Off-policy RL（如 DQN）
    存储 (obs, action, reward, next_obs, done) 五元组
    """
    def __init__(self, size: int, obs_dim: int, act_dim: int):
        # 用张量存储，方便后续直接在 GPU 上操作
        self.obs      = torch.randn(size, obs_dim)     # 当前观测
        self.actions  = torch.randint(0, act_dim, (size,))  # 动作（离散）
        self.rewards  = torch.randn(size)              # 奖励
        self.next_obs = torch.randn(size, obs_dim)     # 下一个观测
        self.dones    = torch.zeros(size, dtype=torch.bool)  # 是否终止

    def __len__(self) -> int:
        # 返回数据集大小，DataLoader 用这个确定有多少个样本
        return len(self.obs)

    def __getitem__(self, idx: int):
        # 返回第 idx 个样本（DataLoader 会自动批量化）
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

# ── 5.2 DataLoader：自动批量、打乱、多进程加载 ──────────────────────────────
dataset = RLReplayBuffer(size=1000, obs_dim=8, act_dim=4)

loader = DataLoader(
    dataset,
    batch_size=32,          # 每个 batch 包含 32 个样本
    shuffle=True,           # 每个 epoch 开始时打乱数据顺序
    num_workers=0,          # 数据加载的子进程数（Windows 上建议设为 0）
    pin_memory=False,       # 如果用 GPU，设为 True 可加速 CPU→GPU 传输
    drop_last=False,        # 是否丢弃最后一个不完整的 batch
)

print(f"Dataset size: {len(dataset)}")
print(f"Batches per epoch: {len(loader)}")  # = ceil(1000/32) = 32

# ── 5.3 model.train() vs model.eval() ───────────────────────────────────────
# 某些层在训练和推理时行为不同：
#   Dropout：训练时随机丢弃神经元；推理时关闭（所有神经元激活）
#   BatchNorm：训练时用当前 batch 统计量；推理时用累积的 running_mean/var

policy.train()              # 开启训练模式（Dropout 等生效）
# ... 训练代码 ...

policy.eval()               # 切换到推理模式（Dropout 关闭，BN 用 running stats）
# ... 评估代码 ...

# ── 5.4 评估循环（eval loop）────────────────────────────────────────────────
print("\n--- Eval demo ---")
policy = policy.to(device)      # policy 移到目标设备，与输入数据保持一致
policy.eval()

all_losses = []
with torch.inference_mode():            # 推理时用 inference_mode 或 no_grad
    for batch in loader:
        obs_b, act_b, rew_b, next_obs_b, done_b = batch
        obs_b = obs_b.to(device)       # 移到目标设备
        pred_b = policy(obs_b)         # 前向传播（不建立计算图）
        target_b = torch.randn_like(pred_b).to(device)  # 假目标
        loss_b = F.mse_loss(pred_b, target_b)
        all_losses.append(loss_b.item())

avg_loss = sum(all_losses) / len(all_losses)
print(f"Eval avg loss: {avg_loss:.4f}")


# =============================================================================
# Part 6 — Practical Recipes（实用技巧）
# Checkpointing / Debugging / Common Gotchas
# =============================================================================
print("\n" + "=" * 60)
print("Part 6 — Practical Recipes")
print("=" * 60)

# ── 6.1 Checkpointing（检查点：保存和恢复训练状态）──────────────────────────
# 完整的 checkpoint 要保存：模型权重 + 优化器状态 + epoch/step 数

def save_checkpoint(model, optimizer, scheduler, step, path):
    """保存完整训练状态到磁盘"""
    torch.save({
        'step': step,                           # 当前训练步数
        'model_state_dict': model.state_dict(), # 模型权重
        'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态（含 momentum 等）
        'scheduler_state_dict': scheduler.state_dict(),  # 调度器状态
    }, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, scheduler, path):
    """从磁盘加载训练状态，恢复训练"""
    checkpoint = torch.load(path, map_location='cpu')  # map_location 防止设备不匹配
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['step']

# 演示保存（注释掉避免实际写文件）
# save_checkpoint(policy, optimizer, scheduler, step=100, path='ckpt.pt')

# ── 6.2 常见 Bug（Gotchas）──────────────────────────────────────────────────
print("\n--- Common Gotchas ---")

# Gotcha 1：忘记 optimizer.zero_grad()
# 症状：loss 异常大或不收敛（梯度不断累积）

# Gotcha 2：忘记 model.eval()
# 症状：评估时 Dropout 仍然生效，结果随机且偏低

# Gotcha 3：在 no_grad 块里用了 .backward()（会报错）

# Gotcha 4：CPU / GPU 设备不匹配
# 症状：RuntimeError: Expected all tensors to be on the same device
# 修复：在数据进入模型前统一 .to(device)
print("常见 bug 见注释，不实际执行")

# Gotcha 5：使用 loss.item() 之前忘记 detach
# 如果直接把 tensor 加入 Python list，会保留计算图，内存泄漏！
losses_wrong = []
for _ in range(3):
    fake_loss = policy(torch.randn(4, 8).to(device)).mean()  # 带计算图的 tensor
    losses_wrong.append(fake_loss.item())          # .item() 正确：返回 Python float
print("正确累积 loss:", losses_wrong)

# Gotcha 6：in-place 操作破坏计算图
# x += 1 等 in-place 操作会破坏 autograd，导致报错
# 解法：改用 x = x + 1（out-of-place）

# ── 6.3 调试工具（Debugging tools）─────────────────────────────────────────
print("\n--- Debugging tools ---")

# 检查 NaN / Inf（梯度爆炸时会出现）
t = torch.tensor([1.0, float('nan'), float('inf')])
print("has nan:", torch.isnan(t).any().item())   # True
print("has inf:", torch.isinf(t).any().item())   # True

# 检查梯度范数（训练时监控，判断是否梯度爆炸/消失）
policy.train()
fake_obs = torch.randn(4, 8).to(device)
fake_target = torch.randn(4, 4).to(device)
optimizer.zero_grad()
fake_pred = policy(fake_obs)
fake_loss = F.mse_loss(fake_pred, fake_target)
fake_loss.backward()

# 计算所有参数梯度的总 L2 范数
total_norm = 0.0
for p in policy.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)    # 单个参数的梯度 L2 范数
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5
print(f"Gradient norm before clip: {total_norm:.4f}")

# clip_grad_norm_ 返回裁剪前的范数（可用于日志）
norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
print(f"Gradient norm (returned by clip): {norm:.4f}")


# =============================================================================
# Part 7 — Modern PyTorch（torch.compile）
# torch.compile 是 PyTorch 2.0 引入的编译器加速，一行代码加速模型
# =============================================================================
print("\n" + "=" * 60)
print("Part 7 — Modern PyTorch: torch.compile")
print("=" * 60)

# torch.compile(model) → 用 TorchDynamo + TorchInductor 把模型编译成优化的内核
# 优化效果：一般能带来 1.5x~3x 的速度提升（取决于模型结构和硬件）
# 首次调用时有编译开销（warm-up），之后运行加速

# 用法：只需一行，API 与普通模型完全相同
# 注意：Windows 上 torch.compile + CUDA 需要 Triton（官方暂不支持 Windows）
# 生产环境建议用 try-except 优雅降级
try:
    compiled_policy = torch.compile(policy)
    with torch.inference_mode():
        test_obs = torch.randn(16, 8).to(device)
        out = compiled_policy(test_obs)     # 编译后的前向传播
        print("compiled output shape:", out.shape)
except Exception as e:
    print(f"torch.compile 不可用（{type(e).__name__}），回退到普通模式")
    with torch.inference_mode():
        test_obs = torch.randn(16, 8).to(device)
        out = policy(test_obs)
        print("fallback output shape:", out.shape)

# compile 的 mode 参数（权衡编译时间 vs 运行速度）：
# mode='default'          → 默认，平衡编译时间和速度
# mode='reduce-overhead'  → 减少框架开销，适合小 batch
# mode='max-autotune'     → 最大化自动调优，编译慢但运行最快
# compiled_policy = torch.compile(policy, mode='reduce-overhead')

print("\ntorch.compile 适用场景：")
print("  - 模型结构固定（无动态分支）时效果最好")
print("  - RL 中的 policy/value network 前向推理加速")
print("  - 如遇报错，可回退到普通模式继续调试")


# =============================================================================
# 综合示例：RL 场景下的完整训练框架
# =============================================================================
print("\n" + "=" * 60)
print("综合示例：RL-style Value Network 训练")
print("=" * 60)

class ValueNet(nn.Module):
    """价值网络：输入状态，输出标量价值 V(s)"""
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),                  # Tanh 在 RL 中比 ReLU 更稳定
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),       # 输出单个标量
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)  # (B,1) → (B,)，便于与 reward 对齐

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建网络并移到设备
value_net = ValueNet(obs_dim=8).to(device)

# AdamW + 余弦退火（现代训练标配）
val_optimizer = optim.AdamW(value_net.parameters(), lr=3e-4, weight_decay=1e-4)
val_scheduler = optim.lr_scheduler.CosineAnnealingLR(val_optimizer, T_max=200)

print(f"\n在 {device} 上训练价值网络...")

# 模拟 RL rollout 数据
n_steps = 10
obs_dim  = 8

value_net.train()                       # 切换训练模式

for step in range(n_steps):
    # 模拟 rollout 数据（实际 RL 中从环境采集）
    obs     = torch.randn(64, obs_dim).to(device)   # (B, obs_dim)
    returns = torch.randn(64).to(device)            # (B,)，折扣回报 G_t

    val_optimizer.zero_grad()           # 1. 清零梯度

    values = value_net(obs)             # 2. 前向传播，values shape=(64,)
    loss   = F.mse_loss(values, returns)  # 3. MSE(V(s), G_t)

    loss.backward()                     # 4. 反向传播

    torch.nn.utils.clip_grad_norm_(    # 5. 梯度裁剪
        value_net.parameters(), max_norm=0.5
    )

    val_optimizer.step()                # 6. 参数更新
    val_scheduler.step()                # 7. 学习率更新

    if step % 2 == 0:
        lr = val_scheduler.get_last_lr()[0]
        print(f"  step {step:3d} | loss={loss.item():.4f} | lr={lr:.6f}")

# 推理（评估价值）
value_net.eval()
with torch.inference_mode():
    test_obs = torch.randn(10, obs_dim).to(device)
    pred_values = value_net(test_obs)
    print(f"\n推理结果 shape: {pred_values.shape}")  # (10,)
    print(f"预测价值: {pred_values.detach().cpu().numpy().round(3)}")

print("\n--- 教程完成 ---")
print("覆盖内容：Tensors / Autograd / Modules / Training Loop / Data+Eval / Recipes / torch.compile")