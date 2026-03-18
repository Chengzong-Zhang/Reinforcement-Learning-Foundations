# ============================================================
#   PyTorch 速成教程 —— 一天掌握核心语法
#   涵盖：张量、自动微分、神经网络、训练循环、GPU
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ============================================================
# 第一部分：张量 (Tensor) —— PyTorch 的核心数据结构
# ============================================================

# ---------- 1.1 创建张量 ----------

x = torch.zeros(3, 4)          # 全0张量，形状 (3,4)，默认 float32
x = torch.ones(3, 4)           # 全1张量
x = torch.full((3, 4), 7.0)    # 全填充为7.0，形状 (3,4)
x = torch.eye(4)               # 4x4 单位矩阵（对角线为1）
x = torch.empty(3, 4)          # 未初始化张量（值随机，不要直接使用）

x = torch.randn(3, 4)          # 标准正态分布 N(0,1) 随机张量
x = torch.rand(3, 4)           # 均匀分布 [0,1) 随机张量
x = torch.randint(0, 10, (3, 4))  # 整数随机张量，值在 [0,10) 之间

x = torch.arange(0, 10, 2)     # 类似 range(0,10,2)，结果 [0,2,4,6,8]
x = torch.linspace(0, 1, 5)    # [0,1] 之间均匀取5个点

# 从 Python list 创建（会推断 dtype）
a = torch.tensor([1, 2, 3])            # int64
b = torch.tensor([1.0, 2.0, 3.0])     # float32
c = torch.tensor([[1, 2], [3, 4]])     # 2D，shape (2,2)

# 从 NumPy 数组创建
np_arr = np.array([1.0, 2.0, 3.0])
t_from_np = torch.from_numpy(np_arr)   # 共享内存，不拷贝
t_copy    = torch.tensor(np_arr)       # 拷贝数据

# ---------- 1.2 dtype 与 device ----------

w = torch.ones(5, dtype=torch.float32)   # 单精度浮点（最常用）
w = torch.ones(5, dtype=torch.float64)   # 双精度浮点
w = torch.ones(5, dtype=torch.int32)     # 32位整数
w = torch.ones(5, dtype=torch.bool)      # 布尔型

# 查看设备（CPU 或 GPU）
print(x.device)   # cpu

# 创建时指定设备（GPU 需要 CUDA 环境）
# y = torch.randn(10, device='cuda')      # 直接在 GPU 上创建
# y = torch.randn(10, device='cuda:0')    # 指定第0块 GPU

# 在 CPU / GPU 之间移动
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(3, 4)
x = x.to(device)          # 移动到目标设备（推荐写法）
# x = x.cuda()            # 等价，但不灵活

# ---------- 1.3 张量属性 ----------

x = torch.randn(2, 3, 4)
print(x.shape)      # torch.Size([2, 3, 4])
print(x.ndim)       # 维数：3
print(x.dtype)      # torch.float32
print(x.device)     # cpu
print(x.numel())    # 元素总数：24

# ============================================================
# 第二部分：张量操作
# ============================================================

# ---------- 2.1 形状变换 ----------

x = torch.randn(2, 3, 4)

# reshape / view：返回新形状的视图（尽量共享内存）
y = x.reshape(6, 4)         # 形状 (6,4)
y = x.view(6, 4)            # 同上，要求内存连续

# -1 表示让 PyTorch 自动推断该维度大小
y = x.reshape(2, -1)        # (2, 12)
y = x.reshape(-1)           # 展平为 1D，长度 24

# 增加/删除维度
y = x.unsqueeze(0)          # 在第0维插入一个新轴，(1,2,3,4)
y = x.unsqueeze(-1)         # 在最后插入新轴，(2,3,4,1)
y = x.unsqueeze(0).squeeze(0)  # squeeze 删除大小为1的维度

# 转置
x2d = torch.randn(3, 4)
y = x2d.T                   # 2D 转置，(4,3)
y = x.permute(2, 0, 1)      # 任意维度重排，(4,2,3)
y = x.transpose(0, 1)       # 交换第0和第1维，(3,2,4)

# 展平
y = x.flatten()             # 完全展平
y = x.flatten(start_dim=1)  # 从第1维开始展平，(2,12)

# ---------- 2.2 拼接与分割 ----------

a = torch.randn(2, 3)
b = torch.randn(2, 3)

# cat：沿已有维度拼接
y = torch.cat([a, b], dim=0)   # (4,3) —— 沿行拼接
y = torch.cat([a, b], dim=1)   # (2,6) —— 沿列拼接

# stack：新建一个维度然后拼接（输入形状必须完全相同）
y = torch.stack([a, b], dim=0) # (2,2,3) —— 新第0维

# chunk / split：分割张量
chunks = torch.chunk(y, 2, dim=0)   # 沿第0维分成2份
pieces = torch.split(y, 1, dim=0)   # 沿第0维每份大小为1

# ---------- 2.3 索引与切片 ----------

x = torch.arange(24).reshape(2, 3, 4).float()

print(x[0])            # 第0个"页"，shape (3,4)
print(x[0, 1])         # 第0页第1行，shape (4,)
print(x[0, 1, 2])      # 标量元素
print(x[:, :, 0])      # 所有页、所有行、第0列，shape (2,3)
print(x[..., 0])       # ... 代表省略前面所有维度，同上

# 花式索引（整数数组索引）
idx = torch.tensor([0, 2])
print(x[:, idx, :])    # 取第1维中第0和第2行

# 布尔掩码索引
mask = x > 10
print(x[mask])         # 返回满足条件的元素，1D

# ---------- 2.4 数学运算 ----------

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 元素级运算（逐元素）
print(a + b)            # 加
print(a - b)            # 减
print(a * b)            # 乘（不是矩阵乘法！）
print(a / b)            # 除
print(a ** 2)           # 幂
print(torch.sqrt(a))    # 开方

# 矩阵乘法
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B               # @ 运算符，等价于 torch.matmul(A, B)
C = torch.mm(A, B)      # 严格 2D 矩阵乘法

# 批量矩阵乘法（batch matmul）
A = torch.randn(8, 3, 4)   # batch=8，每个 (3,4)
B = torch.randn(8, 4, 5)
C = torch.bmm(A, B)         # (8,3,5)

# 向量点积
dot = torch.dot(a, b)       # 标量

# 归约操作
x = torch.randn(3, 4)
print(x.sum())              # 全部求和，标量
print(x.sum(dim=0))         # 沿第0维求和，shape (4,)
print(x.sum(dim=1))         # 沿第1维求和，shape (3,)
print(x.sum(dim=1, keepdim=True))  # 保持维度，shape (3,1)

print(x.mean())             # 均值
print(x.max())              # 最大值（标量）
print(x.max(dim=0))         # 返回 (values, indices) 两个张量
print(x.argmax(dim=1))      # 每行最大值的索引，shape (3,)
print(x.min())
print(x.std())
print(x.norm())             # L2 范数

# 广播（Broadcasting）：形状不同时自动扩展
# 规则：从尾部对齐，大小为1的维度可以被广播
A = torch.randn(3, 1)
B = torch.randn(1, 4)
C = A + B               # 结果 (3,4)，A 的列被复制4次，B 的行被复制3次

# ---------- 2.5 原地操作（inplace） ----------

x = torch.ones(3)
x.add_(1)       # 下划线结尾 = 原地操作，x 变为 [2,2,2]
x.mul_(2)       # 原地乘2
x.zero_()       # 清零

# ---------- 2.6 类型转换 ----------

x = torch.randn(3)
y = x.int()             # float32 -> int32
y = x.long()            # float32 -> int64
y = x.half()            # float32 -> float16
y = x.float()           # -> float32
y = x.double()          # -> float64
y = x.to(torch.int8)    # 通用写法

# ---------- 2.7 与 NumPy 互转 ----------

t = torch.randn(3, 4)
n = t.numpy()           # Tensor -> ndarray（CPU 上共享内存）
t2 = torch.from_numpy(n)  # ndarray -> Tensor（共享内存）

# ============================================================
# 第三部分：自动微分 (Autograd)
# ============================================================

# ---------- 3.1 requires_grad 与 grad ----------

# 只有 requires_grad=True 的张量才会被追踪梯度
x = torch.tensor(2.0, requires_grad=True)

# 前向计算：构建计算图
y = x ** 2 + 3 * x + 1   # y = x^2 + 3x + 1

# 反向传播：计算梯度
y.backward()              # dy/dx = 2x + 3，在 x=2 时 = 7

print(x.grad)             # tensor(7.)

# ---------- 3.2 多变量梯度 ----------

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()        # y = x1^2 + x2^2 + x3^2
y.backward()
print(x.grad)             # [2*1, 2*2, 2*3] = [2., 4., 6.]

# ---------- 3.3 不追踪梯度的场景 ----------

# 推理阶段不需要梯度，用 no_grad 节省内存和计算
x = torch.randn(3, requires_grad=True)
with torch.no_grad():
    y = x * 2              # 这里 y 不在计算图中

# 单次操作不追踪
y = x.detach()             # 返回不带梯度的张量（共享数据）
y = x.detach().numpy()     # 先 detach 再转 numpy（常用）

# ---------- 3.4 梯度累积与清零 ----------

x = torch.tensor(1.0, requires_grad=True)
for _ in range(3):
    y = x * 2
    y.backward()
    # 注意：PyTorch 默认累积梯度，不会自动清零！
    print(x.grad)          # 2, 4, 6（累积）
    x.grad.zero_()         # 手动清零，否则下次 backward 会累加

# ============================================================
# 第四部分：nn.Module —— 定义神经网络
# ============================================================

# ---------- 4.1 最简单的线性网络 ----------

# Sequential：按顺序堆叠层，适合简单网络
model_simple = nn.Sequential(
    nn.Linear(784, 256),   # 全连接层：输入784，输出256
    nn.ReLU(),             # 激活函数
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),    # 输出10类
)

# ---------- 4.2 继承 nn.Module（推荐，更灵活）----------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()  # 必须调用父类 __init__
        # 在 __init__ 里声明所有子层（会自动注册参数）
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  # 以50%概率随机丢弃神经元

    def forward(self, x):
        # forward 定义数据流向，自动微分会追踪这里的计算
        x = self.fc1(x)        # 线性变换
        x = self.relu(x)       # 激活
        x = self.dropout(x)    # Dropout（训练时随机丢弃）
        x = self.fc2(x)        # 输出层
        return x               # 返回 logits（未经 softmax）

model = MLP(784, 256, 10)

# ---------- 4.3 常用层速查 ----------

# 全连接层
nn.Linear(in_features=128, out_features=64, bias=True)

# 卷积层（图像）
nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
# padding='same' 可以保持特征图大小不变

# 池化层
nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化，特征图缩小一半
nn.AvgPool2d(kernel_size=2, stride=2)  # 平均池化

# 批归一化（加速训练、稳定梯度）
nn.BatchNorm1d(num_features=128)  # 用于全连接层输出
nn.BatchNorm2d(num_features=16)   # 用于卷积层输出（通道数）

# 层归一化（Transformer 常用）
nn.LayerNorm(normalized_shape=128)

# Dropout
nn.Dropout(p=0.5)    # 1D，全连接层后
nn.Dropout2d(p=0.5)  # 2D，卷积层后（整个通道置0）

# 嵌入层（NLP，将整数 token 转为向量）
nn.Embedding(num_embeddings=10000, embedding_dim=256)

# 循环神经网络
nn.RNN(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
nn.GRU(input_size=128, hidden_size=256, num_layers=2, batch_first=True)

# Transformer
nn.MultiheadAttention(embed_dim=512, num_heads=8)
nn.TransformerEncoderLayer(d_model=512, nhead=8)
nn.TransformerEncoder(nn.TransformerEncoderLayer(512, 8), num_layers=6)

# ---------- 4.4 激活函数 ----------

nn.ReLU()            # max(0, x)，最常用
nn.LeakyReLU(0.01)   # 负半轴有小斜率，解决"神经元死亡"
nn.GELU()            # 平滑版 ReLU，Transformer 常用
nn.Sigmoid()         # 输出 (0,1)，二分类输出层
nn.Tanh()            # 输出 (-1,1)
nn.Softmax(dim=1)    # 多分类概率，通常在损失函数内部处理

# 也可以用函数式 API（不需要创建对象）
F.relu(x)
F.gelu(x)
F.sigmoid(x)
F.softmax(x, dim=1)

# ---------- 4.5 查看模型 ----------

model = MLP(784, 256, 10)
print(model)                           # 打印网络结构
print(sum(p.numel() for p in model.parameters()))  # 总参数量
# 只计算可训练参数
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# 遍历参数
for name, param in model.named_parameters():
    print(name, param.shape)

# ============================================================
# 第五部分：损失函数
# ============================================================

# 多分类交叉熵（最常用）
# 输入：logits (N, C)，target (N,) 整数类别
criterion = nn.CrossEntropyLoss()
logits = torch.randn(8, 10)   # batch=8，10类
labels = torch.randint(0, 10, (8,))  # 每个样本的真实类别
loss = criterion(logits, labels)

# 二分类交叉熵（配合 Sigmoid）
# 输入：概率 (N,)，target (N,) 0或1
criterion = nn.BCELoss()
probs = torch.sigmoid(torch.randn(8))
labels_bin = torch.randint(0, 2, (8,)).float()
loss = criterion(probs, labels_bin)

# BCEWithLogitsLoss（更稳定，内部合并 Sigmoid）
criterion = nn.BCEWithLogitsLoss()
logits_bin = torch.randn(8)
loss = criterion(logits_bin, labels_bin)

# 均方误差（回归）
criterion = nn.MSELoss()
pred = torch.randn(8)
target = torch.randn(8)
loss = criterion(pred, target)

# 平均绝对误差（回归，对异常值鲁棒）
criterion = nn.L1Loss()

# Huber Loss（MSE 和 L1 的结合）
criterion = nn.SmoothL1Loss()

# ============================================================
# 第六部分：优化器
# ============================================================

model = MLP(784, 256, 10)

# SGD（随机梯度下降）
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
# lr: 学习率；momentum: 动量（加速收敛）；weight_decay: L2正则化

# Adam（最常用，自适应学习率）
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

# AdamW（Adam + 正确的权重衰减，Transformer 标配）
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.01)

# ---------- 学习率调度器 ----------

# 每隔 step_size 个 epoch，lr 乘以 gamma
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 余弦退火（推荐）
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 根据验证 loss 自动降低 lr
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# ============================================================
# 第七部分：数据加载
# ============================================================

# ---------- 7.1 自定义 Dataset ----------

class MyDataset(Dataset):
    def __init__(self, X, y):
        # X: numpy array 或 tensor，y: 标签
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        # 返回数据集大小，DataLoader 需要
        return len(self.X)

    def __getitem__(self, idx):
        # 返回单个样本（DataLoader 会自动批量化）
        return self.X[idx], self.y[idx]

# ---------- 7.2 DataLoader ----------

# 模拟数据
X_data = np.random.randn(1000, 20).astype(np.float32)
y_data = np.random.randint(0, 2, 1000)

dataset = MyDataset(X_data, y_data)

# 划分训练集/验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=32,       # 每批32个样本
    shuffle=True,        # 训练时打乱顺序
    num_workers=0,       # 并行加载的进程数（Windows 建议 0）
    pin_memory=True,     # GPU 训练时加速（将数据固定在内存）
)

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# ============================================================
# 第八部分：完整训练循环
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()              # 切换到训练模式（启用 Dropout、BatchNorm 更新）
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in loader:
        # 1. 将数据移到目标设备
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # 2. 清零上一步的梯度（必须！否则梯度累积）
        optimizer.zero_grad()

        # 3. 前向传播
        logits = model(batch_X)         # (batch, num_classes)

        # 4. 计算损失
        loss = criterion(logits, batch_y)

        # 5. 反向传播（计算梯度）
        loss.backward()

        # 6. 梯度裁剪（防止梯度爆炸，可选）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 7. 更新参数
        optimizer.step()

        # 统计
        total_loss += loss.item() * batch_X.size(0)  # .item() 将标量 tensor 转为 Python float
        pred = logits.argmax(dim=1)                  # 取最大 logit 对应的类别
        correct += (pred == batch_y).sum().item()
        total += batch_X.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()               # 切换到评估模式（关闭 Dropout，BatchNorm 使用统计值）
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():      # 不计算梯度，节省内存
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_X.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_X.size(0)

    return total_loss / total, correct / total


# ---------- 完整训练主程序 ----------

def main():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 模型、损失、优化器
    model = MLP(20, 64, 2).to(device)    # 输入20维，二分类
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val_acc = 0.0
    num_epochs = 50

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # 调整学习率
        scheduler.step()

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')  # 只保存参数

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"Train Loss {train_loss:.4f} Acc {train_acc:.3f} | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc:.3f} | "
                  f"LR {scheduler.get_last_lr()[0]:.2e}")

    print(f"最优验证准确率: {best_val_acc:.3f}")


# ============================================================
# 第九部分：模型保存与加载
# ============================================================

model = MLP(784, 256, 10)

# 保存（推荐只保存参数字典，不保存整个模型）
torch.save(model.state_dict(), 'model_weights.pth')

# 加载
model2 = MLP(784, 256, 10)          # 先创建同架构的模型
model2.load_state_dict(torch.load('model_weights.pth', map_location='cpu'))
model2.eval()                        # 加载后记得切换模式

# 保存完整 checkpoint（含优化器状态，可继续训练）
checkpoint = {
    'epoch': 10,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'best_val_acc': 0.95,
}
torch.save(checkpoint, 'checkpoint.pth')

# 加载 checkpoint
ckpt = torch.load('checkpoint.pth', map_location='cpu')
model.load_state_dict(ckpt['model_state'])
optimizer.load_state_dict(ckpt['optimizer_state'])
start_epoch = ckpt['epoch']

# ============================================================
# 第十部分：CNN 示例（图像分类）
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 特征提取部分
        self.features = nn.Sequential(
            # 输入 (B, 1, 28, 28) —— MNIST 灰度图
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # -> (B, 32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (B, 32, 14, 14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> (B, 64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (B, 64, 7, 7)
        )
        # 分类部分
        self.classifier = nn.Sequential(
            nn.Flatten(),              # (B, 64*7*7) = (B, 3136)
            nn.Linear(3136, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============================================================
# 第十一部分：实用技巧速查
# ============================================================

# ---------- 随机种子（保证实验可复现）----------
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # 如需完全确定性（会损失性能）：
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- 混合精度训练（加速 GPU 训练）----------
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()   # 梯度缩放器，防止 float16 梯度下溢

# 在训练循环中：
def train_step_amp(model, batch_X, batch_y, optimizer, criterion, scaler):
    optimizer.zero_grad()
    with autocast():                  # 自动选择 float16/float32
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
    scaler.scale(loss).backward()     # 缩放后反传
    scaler.step(optimizer)            # 更新参数
    scaler.update()                   # 调整缩放因子

# ---------- 常用调试技巧 ----------

# 检查张量统计
x = torch.randn(100)
print(f"mean={x.mean():.3f}, std={x.std():.3f}, min={x.min():.3f}, max={x.max():.3f}")

# 检查是否有 NaN/Inf
print(torch.isnan(x).any())    # 是否有 NaN
print(torch.isinf(x).any())    # 是否有 Inf

# 计算模型输出的形状（常用于调试）
model = SimpleCNN()
dummy = torch.randn(1, 1, 28, 28)  # batch=1 的假输入
with torch.no_grad():
    out = model(dummy)
print(out.shape)   # torch.Size([1, 10])

# ============================================================
# 主程序入口
# ============================================================

if __name__ == '__main__':
    set_seed(42)
    main()
