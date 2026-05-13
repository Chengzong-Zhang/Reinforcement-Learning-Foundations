"""
Actor-Critic on CartPole
========================

这个文件实现了一个最基础、最适合入门学习的 Actor-Critic 算法。

最简单的运行方式：

    python Actor-Critic.py

如果只是想快速检查代码能不能跑，可以少训练几局：

    python Actor-Critic.py --num-episodes 10

算法架构
--------
Actor-Critic 由两个神经网络组成：

1. Actor，也叫策略网络。
   输入：状态 state。
   输出：每个动作的概率。
   作用：决定“当前应该做什么动作”。

2. Critic，也叫价值网络。
   输入：状态 state。
   输出：状态价值 V(s)，也就是“这个状态大概有多好”。
   作用：评价 Actor 刚才做的动作好不好。

一次训练的大致流程
------------------
1. 环境给出当前状态 s。
2. Actor 根据 s 输出动作概率 pi(a|s)，并采样一个动作 a。
3. 环境执行动作 a，返回奖励 r、下一个状态 s'、是否结束 done。
4. 保存一整局游戏的数据。
5. 一局结束后更新网络：
   - Critic 学习 TD target。
   - Actor 根据 TD delta 调整动作概率。

核心公式
--------
TD target:

    r + gamma * V(s') * (1 - done)

TD delta:

    TD target - V(s)

Actor loss:

    -log pi(a|s) * TD delta

Critic loss:

    MSE(V(s), TD target)

直观理解
--------
如果 TD delta > 0，说明这个动作比 Critic 原来预期更好，Actor 会提高这个动作概率。
如果 TD delta < 0，说明这个动作比 Critic 原来预期更差，Actor 会降低这个动作概率。

阅读顺序建议
------------
编程基础还不熟时，可以按这个顺序看：
1. main()：看程序从哪里开始运行。
2. train()：看智能体如何和环境玩一局游戏。
3. ActorCritic.take_action()：看 Actor 如何选择动作。
4. ActorCritic.update()：看 Actor 和 Critic 如何学习。
"""

# argparse 用来读取命令行参数，例如 --num-episodes 100。
import argparse

# Path 用来处理文件路径，比手写字符串拼接更稳。
from pathlib import Path

# Any 表示“任意类型”，这里主要用来让 Pylance 理解 Gym 空间对象的动态属性。
from typing import Any

# matplotlib.pyplot 用来画训练曲线，并保存为图片。
import matplotlib.pyplot as plt

# numpy 用来处理数组、计算平均值、做滑动平均。
import numpy as np

# torch 是 PyTorch 主库，负责张量、神经网络、自动求导。
import torch

# torch.nn.functional 里放着常用神经网络函数，比如 relu、softmax、mse_loss。
import torch.nn.functional as F

# tqdm 用来显示训练进度条。
from tqdm import tqdm

# 使用 gymnasium。它是 gym 的新版维护版本，当前环境中已经安装。
# 这里不再写 import gym 的兜底分支，是为了避免 Pylance 报“无法解析导入 gym”。
import gymnasium as gym


# PolicyNet 是 Actor，也就是策略网络。
# 这个网络负责把状态变成动作概率。
class PolicyNet(torch.nn.Module):
    # __init__ 是初始化函数，创建 PolicyNet 对象时自动执行。
    # state_dim：状态维度，CartPole-v1 中是 4。
    # hidden_dim：隐藏层神经元数量，例如 128。
    # action_dim：动作数量，CartPole-v1 中是 2。
    def __init__(self, state_dim, hidden_dim, action_dim):
        # 初始化 PyTorch 神经网络父类。写自定义网络时基本都要写这一句。
        super().__init__()

        # 第一层全连接层：输入状态，输出隐藏特征。
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)

        # 第二层全连接层：输入隐藏特征，输出每个动作的分数。
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    # forward 定义网络如何从输入算到输出。
    # 当代码写 self.actor(x) 时，PyTorch 会自动调用 forward(x)。
    # x 的形状一般是 [batch_size, state_dim]。
    def forward(self, x):
        # 先通过第一层全连接层，再通过 ReLU 激活函数。
        # ReLU 会把负数变成 0，让网络能学习非线性关系。
        x = F.relu(self.fc1(x))

        # 再通过第二层得到动作分数，然后用 softmax 变成概率。
        # dim=1 表示每一行内部做 softmax，让每个样本的动作概率和为 1。
        return F.softmax(self.fc2(x), dim=1)


# ValueNet 是 Critic，也就是价值网络。
# 这个网络负责估计状态价值 V(s)。
class ValueNet(torch.nn.Module):
    # state_dim：状态维度。
    # hidden_dim：隐藏层神经元数量。
    def __init__(self, state_dim, hidden_dim):
        # 初始化 PyTorch 神经网络父类。
        super().__init__()

        # 第一层全连接层：输入状态，输出隐藏特征。
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)

        # 第二层全连接层：输出 1 个数字，也就是状态价值 V(s)。
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    # x 的形状一般是 [batch_size, state_dim]。
    def forward(self, x):
        # 状态先经过第一层和 ReLU。
        x = F.relu(self.fc1(x))

        # 输出状态价值，形状是 [batch_size, 1]。
        return self.fc2(x)


# ActorCritic 把 Actor、Critic 和更新算法包装在一个类里。
# 训练时只需要调用：
#   action = agent.take_action(state)
#   agent.update(transition_dict)
class ActorCritic:
    # 初始化智能体。
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        gamma,
        device,
    ):
        # 创建 Actor 策略网络，并移动到 CPU 或 GPU。
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)

        # 创建 Critic 价值网络，并移动到 CPU 或 GPU。
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        # Actor 的优化器，用来更新策略网络参数。
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic 的优化器，用来更新价值网络参数。
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # gamma 是折扣因子，决定未来奖励的重要程度。
        self.gamma = gamma

        # 保存设备信息。device 通常是 "cuda" 或 "cpu"。
        self.device = device

    # take_action 的作用：
    # 输入一个状态 state，输出一个动作 action。
    def take_action(self, state):
        # 环境返回的 state 可能是 list、tuple 或 numpy 数组。
        # 先用 np.asarray 转成 float32 数组，再用 torch.as_tensor 转成 PyTorch 张量。
        state = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device)

        # 神经网络通常要求输入是二维：[batch_size, state_dim]。
        # 这里只有一个状态，所以从 [state_dim] 变成 [1, state_dim]。
        state = state.unsqueeze(0)

        # Actor 输出每个动作的概率。
        # CartPole 中 probs 的形状是 [1, 2]。
        probs = self.actor(state)

        # Categorical 表示离散概率分布。
        # 如果 probs = [0.2, 0.8]，采样到动作 1 的概率更大。
        action_dist = torch.distributions.Categorical(probs=probs)

        # 从概率分布里随机采样一个动作，并转成普通 Python int。
        return action_dist.sample().item()

    # update 的作用：
    # 输入一局游戏收集到的数据，更新 Actor 和 Critic。
    def update(self, transition_dict):
        # states 保存每一步的当前状态。
        # 转换后形状是 [T, state_dim]，T 是这一局的步数。
        states = torch.as_tensor(
            np.asarray(transition_dict["states"], dtype=np.float32),
            device=self.device,
        )

        # actions 保存每一步选择的动作。
        # dtype=torch.long 是因为动作编号要作为索引使用。
        # view(-1, 1) 把形状从 [T] 变成 [T, 1]。
        actions = torch.as_tensor(
            transition_dict["actions"], dtype=torch.long, device=self.device
        ).view(-1, 1)

        # rewards 保存每一步得到的奖励。
        # view(-1, 1) 把形状变成 [T, 1]，方便和价值函数输出对齐。
        rewards = torch.as_tensor(
            transition_dict["rewards"], dtype=torch.float32, device=self.device
        ).view(-1, 1)

        # next_states 保存每一步转移后的下一个状态。
        # 形状是 [T, state_dim]。
        next_states = torch.as_tensor(
            np.asarray(transition_dict["next_states"], dtype=np.float32),
            device=self.device,
        )

        # dones 保存每一步之后 episode 是否结束。
        # True 会变成 1.0，False 会变成 0.0。
        dones = torch.as_tensor(
            transition_dict["dones"], dtype=torch.float32, device=self.device
        ).view(-1, 1)

        # Critic 预测当前状态价值 V(s)。
        values = self.critic(states)

        # TD target 是 Critic 的学习目标：
        # 如果没结束：r + gamma * V(s')
        # 如果结束了：r，因为终止状态没有未来价值。
        # (1 - dones) 在 done=True 时等于 0，可以去掉未来价值。
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)

        # TD delta 表示“目标价值”和“当前估计价值”的差。
        # 它也可以看作一个简单的一步优势函数 advantage。
        td_delta = td_target - values

        # Actor 对所有 states 计算动作概率。
        action_probs = self.actor(states)

        # gather(1, actions) 从每一行概率中取出“实际执行过的动作”的概率。
        # clamp_min(1e-8) 避免概率为 0 时 log(0) 产生无穷大。
        log_probs = torch.log(action_probs.gather(1, actions).clamp_min(1e-8))

        # Actor loss:
        #   -log pi(a|s) * td_delta
        # detach() 表示这里只把 td_delta 当作常数，不让 Actor 的反向传播改到 Critic。
        actor_loss = (-log_probs * td_delta.detach()).mean()

        # Critic loss:
        # 让 Critic 的 V(s) 尽量接近 TD target。
        # td_target.detach() 表示目标值不参与梯度计算。
        critic_loss = F.mse_loss(values, td_target.detach())

        # 清空 Actor 优化器中上一轮残留的梯度。
        self.actor_optimizer.zero_grad()

        # 清空 Critic 优化器中上一轮残留的梯度。
        self.critic_optimizer.zero_grad()

        # 根据 Actor loss 计算 Actor 参数的梯度。
        actor_loss.backward()

        # 根据 Critic loss 计算 Critic 参数的梯度。
        critic_loss.backward()

        # 根据梯度更新 Actor 网络参数。
        self.actor_optimizer.step()

        # 根据梯度更新 Critic 网络参数。
        self.critic_optimizer.step()


# moving_average 用来平滑曲线。
# 输入：一串数 values 和窗口大小 window_size。
# 输出：平滑后的数列。
def moving_average(values, window_size):
    # 转成 numpy 数组，方便做数学计算。
    values = np.asarray(values, dtype=np.float32)

    # 如果数据长度比窗口还短，就无法滑动平均，直接返回原数据。
    if len(values) < window_size:
        return values

    # 在最前面插入 0 后做累加和。
    # 用累加和可以快速计算每个窗口的总和。
    cumulative_sum = np.cumsum(np.insert(values, 0, 0.0))

    # 计算中间完整窗口的平均值。
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size

    # radius 用于处理开头和结尾不足一个完整窗口的部分。
    radius = np.arange(1, window_size - 1, 2)

    # 计算开头部分的平均。
    begin = np.cumsum(values[: window_size - 1])[::2] / radius

    # 计算结尾部分的平均。
    end = (np.cumsum(values[: -window_size : -1])[::2] / radius)[::-1]

    # 拼接开头、中间和结尾，得到完整平滑曲线。
    return np.concatenate((begin, middle, end))


# reset_env 封装环境重置。
# 这样写是为了兼容 Gymnasium 的 reset 返回格式。
def reset_env(env, seed=None):
    # 如果没有传 seed，就普通重置环境。
    if seed is None:
        result = env.reset()

    # 如果传了 seed，就用固定随机种子重置环境，便于复现实验。
    else:
        result = env.reset(seed=seed)

    # Gymnasium 返回 (state, info)，旧 Gym 返回 state。
    # 如果是 tuple，只取第一个元素 state。
    if isinstance(result, tuple):
        return result[0]

    # 如果不是 tuple，说明 result 本身就是 state。
    return result


# step_env 封装环境单步交互。
# 这样写是为了兼容 Gymnasium 的 step 返回格式。
def step_env(env, action):
    # 执行动作，得到环境返回结果。
    result = env.step(action)

    # Gymnasium 返回 5 个值：
    # next_state, reward, terminated, truncated, info。
    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result

        # terminated 是自然结束，truncated 是达到时间限制。
        # 训练时两者都可以看作这一局结束。
        return next_state, reward, terminated or truncated, info

    # 旧 Gym 返回 4 个值：
    # next_state, reward, done, info。
    next_state, reward, done, info = result
    return next_state, reward, done, info


# train 是训练循环。
# 它会让智能体玩 num_episodes 局游戏，并返回每局总奖励。
def train(env, agent, num_episodes):
    # 保存每局游戏总回报。
    return_list = []

    # 把训练过程分成大约 10 段显示进度条。
    episodes_per_iteration = max(1, num_episodes // 10)

    # 计算总共需要多少段。
    num_iterations = int(np.ceil(num_episodes / episodes_per_iteration))

    # 外层循环控制进度条段数。
    for iteration in range(num_iterations):
        # 当前这一段开始的 episode 编号。
        start_episode = iteration * episodes_per_iteration

        # 当前这一段结束的 episode 编号。
        end_episode = min(num_episodes, start_episode + episodes_per_iteration)

        # 当前这一段实际要训练多少局。
        total = end_episode - start_episode

        # 创建进度条。
        with tqdm(total=total, desc=f"Iteration {iteration}") as pbar:
            # 内层循环，每次跑一局完整游戏。
            for local_episode in range(total):
                # 记录这一局的总奖励。
                episode_return = 0.0

                # 保存这一局每一步的数据。
                transition_dict = {
                    "states": [],
                    "actions": [],
                    "next_states": [],
                    "rewards": [],
                    "dones": [],
                }

                # 重置环境，得到初始状态。
                state = reset_env(env)

                # done 表示这一局是否结束。
                done = False

                # 只要没结束，就持续交互。
                while not done:
                    # Actor 根据当前状态选择动作。
                    action = agent.take_action(state)

                    # 环境执行动作，返回下一个状态、奖励、是否结束。
                    next_state, reward, done, _ = step_env(env, action)

                    # 保存当前状态。
                    transition_dict["states"].append(state)

                    # 保存当前动作。
                    transition_dict["actions"].append(action)

                    # 保存下一个状态。
                    transition_dict["next_states"].append(next_state)

                    # 保存当前奖励。
                    transition_dict["rewards"].append(reward)

                    # 保存是否结束。
                    transition_dict["dones"].append(done)

                    # 状态向前推进。
                    state = next_state

                    # 累加奖励。
                    episode_return += reward

                # 一局结束后，记录这一局总奖励。
                return_list.append(episode_return)

                # 用这一局数据更新 Actor 和 Critic。
                agent.update(transition_dict)

                # 每 10 局或当前段最后一局更新进度条显示。
                if (local_episode + 1) % 10 == 0 or local_episode + 1 == total:
                    pbar.set_postfix(
                        {
                            "episode": start_episode + local_episode + 1,
                            "return": f"{np.mean(return_list[-10:]):.3f}",
                        }
                    )

                # 进度条前进一格。
                pbar.update(1)

    # 返回所有 episode 的回报。
    return return_list


# plot_returns 用来保存训练曲线。
# 它会保存两张图：原始曲线和平滑曲线。
def plot_returns(return_list, env_name, output_dir):
    # 如果输出目录不存在，就创建。
    output_dir.mkdir(parents=True, exist_ok=True)

    # 横坐标是 episode 编号。
    episodes = list(range(len(return_list)))

    # 创建第一张图。
    plt.figure()

    # 画原始回报曲线。
    plt.plot(episodes, return_list)

    # 横轴名称。
    plt.xlabel("Episodes")

    # 纵轴名称。
    plt.ylabel("Returns")

    # 图标题。
    plt.title(f"Actor-Critic on {env_name}")

    # 自动调整布局，避免文字被裁掉。
    plt.tight_layout()

    # 原始曲线保存路径。
    raw_path = output_dir / "actor_critic_returns.png"

    # 保存图片。
    plt.savefig(raw_path, dpi=150)

    # 关闭当前图片，释放内存。
    plt.close()

    # 创建第二张图。
    plt.figure()

    # 画平滑后的回报曲线。
    plt.plot(episodes, moving_average(return_list, 9))

    # 横轴名称。
    plt.xlabel("Episodes")

    # 纵轴名称。
    plt.ylabel("Returns")

    # 图标题。
    plt.title(f"Actor-Critic on {env_name}")

    # 自动调整布局。
    plt.tight_layout()

    # 平滑曲线保存路径。
    smooth_path = output_dir / "actor_critic_returns_smoothed.png"

    # 保存图片。
    plt.savefig(smooth_path, dpi=150)

    # 关闭当前图片。
    plt.close()

    # 返回图片路径，方便 main 打印。
    return raw_path, smooth_path


# parse_args 读取命令行参数。
# 比如 --num-episodes 100 会覆盖默认训练局数。
def parse_args():
    # 创建参数解析器。
    parser = argparse.ArgumentParser(description="Train Actor-Critic on CartPole.")

    # 环境名称，默认 CartPole-v1。
    parser.add_argument("--env-name", type=str, default="CartPole-v1")

    # 训练多少局。
    parser.add_argument("--num-episodes", type=int, default=1000)

    # 神经网络隐藏层大小。
    parser.add_argument("--hidden-dim", type=int, default=128)

    # Actor 学习率。
    parser.add_argument("--actor-lr", type=float, default=1e-3)

    # Critic 学习率。
    parser.add_argument("--critic-lr", type=float, default=1e-2)

    # 折扣因子。
    parser.add_argument("--gamma", type=float, default=0.98)

    # 随机种子。
    parser.add_argument("--seed", type=int, default=0)

    # 输出目录。
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )

    # 返回解析后的参数对象。
    return parser.parse_args()


# main 是程序入口。
def main():
    # 读取命令行参数。
    args = parse_args()

    # episode 数必须是正数。
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive.")

    # 有 CUDA GPU 就用 GPU，否则用 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置 numpy 随机种子。
    np.random.seed(args.seed)

    # 设置 PyTorch 随机种子。
    torch.manual_seed(args.seed)

    # 如果使用 GPU，也设置 GPU 随机种子。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 创建环境。
    env = gym.make(args.env_name)

    # 用 seed 重置一次环境，初始化随机性。
    reset_env(env, seed=args.seed)

    # Pylance 对 Gym 的空间类型推断比较保守。
    # 这里标成 Any，表示“运行时有这个属性，我知道自己在做什么”。
    action_space: Any = env.action_space

    # 如果动作空间支持 seed，就设置动作空间随机种子。
    if hasattr(action_space, "seed"):
        action_space.seed(args.seed)

    # 同样把 observation_space 标成 Any，避免 Pylance 把 shape 推成可能为 None。
    observation_space: Any = env.observation_space

    # 如果环境没有 shape，说明这个脚本不支持该环境。
    if not hasattr(observation_space, "shape") or observation_space.shape is None:
        raise ValueError("This script requires an observation space with a shape.")

    # 读取状态维度。
    state_dim = int(observation_space.shape[0])

    # 如果动作空间没有 n，说明不是离散动作空间。
    if not hasattr(action_space, "n"):
        raise ValueError("This script requires a discrete action space with attribute n.")

    # 读取动作数量。
    action_dim = int(action_space.n)

    # 创建 Actor-Critic 智能体。
    agent = ActorCritic(
        state_dim,
        args.hidden_dim,
        action_dim,
        args.actor_lr,
        args.critic_lr,
        args.gamma,
        device,
    )

    # 开始训练。
    return_list = train(env, agent, args.num_episodes)

    # 训练结束后关闭环境。
    env.close()

    # 保存训练曲线。
    raw_path, smooth_path = plot_returns(return_list, args.env_name, args.output_dir)

    # 打印训练设备。
    print(f"Training finished on {device}.")

    # 打印最近 10 局平均回报。
    print(f"Last 10 episode mean return: {np.mean(return_list[-10:]):.3f}")

    # 打印图片保存路径。
    print(f"Saved plots: {raw_path}, {smooth_path}")


# 只有直接运行本文件时才执行 main。
# 如果别的文件 import 这个文件，不会自动开始训练。
if __name__ == "__main__":
    main()
