# PyTorch nn.Module 完整介绍

## 一、为什么需要继承它

PyTorch 不像 Keras 那样有 `model.fit()`，你需要自己写训练循环。`nn.Module` 是一切的基础，它帮你自动管理参数，让你只需专注于网络结构和训练逻辑。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()            # 必须第一行调用，初始化内部参数字典
        self.fc1 = nn.Linear(4, 128)  # 定义层，参数自动被追踪
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):             # 定义数据流向，调用 net(x) 时自动触发
        x = F.relu(self.fc1(x))
        return self.fc2(x)

net = MyNet()
output = net(some_input)   # 调用 net(x) 会自动触发 forward(x)，不要手动调用 forward
```

> **为什么 `super().__init__()` 必须调用？**
> 父类里有一个 `_parameters` 字典，专门收集所有 `nn.Linear` 等层的参数。
> 不调用就没有这个字典，`self.fc1 = ...` 赋值后参数无法被追踪，优化器就找不到参数。

---

## 二、最常用的方法

### 参数相关

```python
net.parameters()        # 返回所有可训练参数的迭代器，传给优化器
net.named_parameters()  # 同上，但同时返回参数名称，调试时常用

# 查看网络总参数量
total = sum(p.numel() for p in net.parameters())
print(f'参数总量: {total}')   # numel() = number of elements，即元素个数

# 查看每层参数量
for name, param in net.named_parameters():
    print(f'{name}: {param.shape}, 参数量={param.numel()}')
```

### 设备相关

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net.to(device)   # 把整个网络（所有参数）移到指定设备，原地操作，不需要重新赋值
net.cuda()       # 等价于 net.to('cuda')
net.cpu()        # 移回 CPU

# 注意：输入数据也要移到同一设备，否则报错
x = x.to(device)
output = net(x)
```

### 保存与加载参数

```python
# 保存（推荐：只保存参数字典，不保存网络结构）
torch.save(net.state_dict(), 'model.pth')

# 加载（需要先创建网络结构，再填入参数）
net2 = MyNet()
net2.load_state_dict(torch.load('model.pth'))

# DQN 里目标网络同步就是这样做的（硬更新：直接复制参数）：
target_net.load_state_dict(online_net.state_dict())

# 软更新（加权平均，更平滑）：
tau = 0.005
for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
    target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)
```

### 训练 / 推理模式切换

```python
net.train()   # 训练模式：开启 Dropout（随机丢弃神经元）、BatchNorm 用当前批次统计量
net.eval()    # 推理模式：关闭 Dropout、BatchNorm 改用训练期间积累的全局统计量
```

> **什么时候必须切换？**
> 网络里有 `nn.Dropout` 或 `nn.BatchNorm` 层时，训练和推理行为不同，必须手动切换。
> 只有全连接/卷积层时无影响，但养成习惯仍然推荐切换。

```python
# 标准训练循环写法
net.train()
for x, y in dataloader:
    pred = net(x)
    loss = F.mse_loss(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 标准推理写法（配合 no_grad 节省显存、加速）
net.eval()
with torch.no_grad():    # 关闭梯度计算，推理时不需要梯度，省显存且更快
    output = net(test_x)
```

### 子模块管理

```python
# 网络里嵌套子网络时，子网络的参数也会被自动追踪
class BigNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MyNet()        # 子模块，参数自动被追踪
        self.head = nn.Linear(2, 1)

big = BigNet()
big.to('cuda')          # encoder 和 head 的参数全部移到 GPU
big.parameters()        # 包含 encoder 和 head 的所有参数

# 查看所有子模块
for name, module in big.named_modules():
    print(name, module)
```

---

## 三、完整训练流程模板

```python
net = MyNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# ---- 训练 ----
net.train()
for x, y in dataloader:
    x, y = x.to(device), y.to(device)

    pred = net(x)                    # 前向传播（自动调用 forward）
    loss = F.mse_loss(pred, y)       # 计算损失

    optimizer.zero_grad()            # 清空上一步梯度（必须！PyTorch 梯度默认累加）
    loss.backward()                  # 反向传播，计算各参数的梯度
    optimizer.step()                 # 优化器根据梯度更新参数

# ---- 推理 ----
net.eval()
with torch.no_grad():
    result = net(test_x.to(device))
```

---

## 四、常见的内置层

| 层 | 用途 |
|---|---|
| `nn.Linear(in, out)` | 全连接层，适用于表格/向量输入 |
| `nn.Conv2d(in_ch, out_ch, kernel_size)` | 二维卷积层，适用于图像输入 |
| `nn.ReLU()` | ReLU 激活函数（也可用 `F.relu()`） |
| `nn.Dropout(p)` | 随机丢弃，防止过拟合 |
| `nn.BatchNorm1d/2d` | 批归一化，加速收敛 |
| `nn.LSTM / nn.GRU` | 循环神经网络，适用于序列数据 |
| `nn.Embedding` | 词嵌入层，适用于 NLP |
| `nn.Sequential` | 顺序容器，简化简单网络的写法 |

```python
# nn.Sequential 示例（适合简单的顺序网络）
net = nn.Sequential(
    nn.Linear(4, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
)
# 等价于上面的 MyNet，但无法实现复杂的分支结构
```

---

## 五、一张图总结

```
nn.Module
├── __init__()          → 定义层（赋值给 self.xxx 即自动追踪参数）
├── forward()           → 定义数据流向（调用 net(x) 时自动触发）
│
├── .parameters()       → 给优化器用（optimizer = Adam(net.parameters())）
├── .named_parameters() → 调试用，看每层参数名和形状
├── .to(device)         → 整个网络移到 GPU/CPU
│
├── .train()            → 开启训练模式（Dropout/BN 生效）
├── .eval()             → 开启推理模式（Dropout 关闭）
│
├── .state_dict()       → 导出参数字典（保存 / 复制给目标网络）
└── .load_state_dict()  → 导入参数字典（加载权重 / 同步目标网络）
```
