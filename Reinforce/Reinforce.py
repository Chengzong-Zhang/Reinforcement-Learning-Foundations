"""
REINFORCE 算法在 CartPole 环境上的完整可运行示例。

这个文件可以当成一份“带代码的学习笔记”来看。它的目标不是把代码写得最短，
而是尽量把每一行为什么存在、每个函数怎么用、整个算法怎么组织讲清楚。

一、这个脚本怎么运行
------------------------------------------------------------
在仓库根目录运行：

    python .\\Reinforce\\Reinforce.py

如果只是想快速检查程序能不能跑，可以少跑几个 episode：

    python .\\Reinforce\\Reinforce.py --num-episodes 5

训练结束后，程序会在 Reinforce/results 目录保存两张图：
1. reinforce_returns.png：每个 episode 的原始回报。
2. reinforce_returns_smoothed.png：平滑后的回报曲线。

二、REINFORCE 算法的整体架构
------------------------------------------------------------
强化学习里，智能体 agent 和环境 env 反复互动：

    state -> policy_net -> action -> env.step(action) -> reward, next_state

在这个程序中，架构可以拆成 5 块：

1. PolicyNet
   策略网络。输入一个状态 state，输出每个动作的概率。
   例如 CartPole 有两个动作：向左推、向右推。
   网络输出可能是 [0.35, 0.65]，表示更倾向选择动作 1。

2. REINFORCE
   算法类。它持有一个 PolicyNet，并负责两件事：
   - take_action(state)：根据策略网络输出的概率随机采样动作。
   - update(transition_dict)：用一整条轨迹的数据更新策略网络。

3. train
   训练循环。它让 agent 跟 env 玩很多局游戏，每玩完一局就更新一次策略。
   REINFORCE 是 on-policy 算法，所以一条轨迹用完就丢，不放进经验回放池。

4. plot_returns
   画图函数。把训练过程中的回报保存成图片，方便观察学习效果。

5. main
   程序入口。负责读取命令行参数、创建环境、创建 agent、开始训练、保存结果。

三、REINFORCE 的核心思想
------------------------------------------------------------
一局游戏会产生一条轨迹：

    s0, a0, r0, s1, a1, r1, ..., sT

从某个时刻 t 开始往后的折扣回报记为 Gt：

    Gt = rt + gamma * r(t+1) + gamma^2 * r(t+2) + ...

如果某个动作后面的 Gt 很大，说明这个动作在当时状态下比较好，
于是我们提高策略以后再次选到它的概率。

代码里使用的损失是：

    loss = -log_prob(action) * G

因为 PyTorch 优化器默认做“梯度下降”，而强化学习目标是“最大化回报”，
所以前面加负号，把最大化问题改写成最小化 loss。
"""

import argparse  # argparse 用来读取命令行参数，比如 --num-episodes 5。
from pathlib import Path  # Path 用来跨平台地处理文件夹和文件路径。

import gymnasium as gym  # Gymnasium 是 Gym 的维护版，用来创建强化学习环境。
import matplotlib.pyplot as plt  # matplotlib 用来画训练回报曲线。
import numpy as np  # numpy 用来做数组转换、随机种子和移动平均。
import torch  # torch 是 PyTorch 主库，用来搭建神经网络和自动求导。
import torch.nn.functional as F  # F 里有 relu、softmax 等常用函数。
from gymnasium.spaces import Box, Discrete  # Box 表示连续状态空间，Discrete 表示离散动作空间。
from tqdm import tqdm  # tqdm 用来显示训练进度条。


class PolicyNet(torch.nn.Module):
    """
    策略网络：输入状态，输出动作概率分布。

    怎么用：
        net = PolicyNet(state_dim=4, hidden_dim=128, action_dim=2)
        state = torch.randn(1, 4)
        probs = net(state)

    对 CartPole 来说：
    - state_dim = 4，因为状态包含小车位置、速度、杆角度、杆角速度。
    - action_dim = 2，因为动作只有向左推和向右推。
    - hidden_dim = 128，是隐藏层神经元数量，可以自己调。
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        # super().__init__() 会初始化 torch.nn.Module 的内部机制。
        # 只有调用它，PyTorch 才能正确追踪网络参数。
        super().__init__()

        # 第一层全连接层：把 state_dim 维状态映射到 hidden_dim 维隐藏特征。
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)

        # 第二层全连接层：把 hidden_dim 维隐藏特征映射到 action_dim 个动作分数。
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # forward 定义“数据如何通过网络”。调用 net(x) 时会自动执行 forward。

        # 先经过第一层线性变换，再用 ReLU 增加非线性表达能力。
        x = F.relu(self.fc1(x))

        # 第二层输出每个动作的原始分数，然后用 softmax 转成概率。
        # dim=1 表示对每一行的动作维度做 softmax。
        return F.softmax(self.fc2(x), dim=1)


class REINFORCE:
    """
    REINFORCE 智能体。

    这个类把“策略网络”和“策略梯度更新规则”封装在一起。

    怎么用：
        agent = REINFORCE(state_dim, hidden_dim, action_dim, lr, gamma, device)
        action = agent.take_action(state)
        agent.update(transition_dict)

    transition_dict 是一整局游戏的数据，格式大概是：
        {
            "states": [s0, s1, ...],
            "actions": [a0, a1, ...],
            "next_states": [s1, s2, ...],
            "rewards": [r0, r1, ...],
            "dones": [False, False, ..., True],
        }
    """

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        # 创建策略网络，并把它放到 device 上。
        # device 可能是 cuda，也可能是 cpu。
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)

        # Adam 是常用优化器。它会根据 loss.backward() 产生的梯度更新网络参数。
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # gamma 是折扣因子。越接近 1，越重视长期奖励；越小，越重视眼前奖励。
        self.gamma = gamma

        # 保存 device，后面创建 tensor 时要保证 tensor 和网络在同一个设备上。
        self.device = device

    def take_action(self, state):
        """
        根据当前策略随机选择一个动作。

        输入：
            state：环境返回的状态，通常是 numpy 数组。

        输出：
            action.item()：Python 整数，传给 env.step(action) 使用。

        为什么是“随机”：
            策略网络输出的是动作概率，不是固定动作。
            例如输出 [0.3, 0.7] 时，不是永远选动作 1，
            而是 30% 概率选动作 0，70% 概率选动作 1。
            这能保证训练早期有探索。
        """

        # np.asarray 把 state 统一转成 numpy 数组；dtype=np.float32 是神经网络常用精度。
        state = np.asarray(state, dtype=np.float32)

        # torch.as_tensor 把 numpy 数组转成 PyTorch tensor，并放到指定设备上。
        state = torch.as_tensor(state, device=self.device)

        # 神经网络通常接收 batch 数据，形状应为 [batch_size, state_dim]。
        # 单个状态原本是 [state_dim]，unsqueeze(0) 后变成 [1, state_dim]。
        state = state.unsqueeze(0)

        # 把状态输入策略网络，得到每个动作的概率。
        probs = self.policy_net(state)

        # Categorical 表示离散概率分布，适合 CartPole 这种离散动作空间。
        action_dist = torch.distributions.Categorical(probs)

        # 按概率采样动作。比如 probs=[0.3, 0.7]，动作 1 更容易被采到。
        action = action_dist.sample()

        # action 是 tensor，env.step 需要普通 Python 整数，所以用 item() 取出数值。
        return action.item()

    def update(self, transition_dict):
        """
        用一整条轨迹更新策略网络。

        输入：
            transition_dict：train 函数收集的一局游戏数据。

        核心步骤：
        1. 从最后一步往前算每一步的折扣回报 G。
        2. 计算当时动作的 log 概率 log_prob。
        3. 构造 loss = -log_prob * G。
        4. 反向传播并更新网络。

        注意：
            REINFORCE 使用蒙特卡洛回报，所以必须等一整局结束后才能更新。
        """

        # 取出这一局里的所有奖励，格式如 [1.0, 1.0, 1.0, ...]。
        reward_list = transition_dict["rewards"]

        # 取出这一局里每一步的状态。
        state_list = transition_dict["states"]

        # 取出这一局里每一步选择的动作。
        action_list = transition_dict["actions"]

        # G 表示“从当前时刻开始往后的折扣回报”，先初始化为 0。
        G = 0.0

        # losses 用来保存每一个时间步的损失，最后加起来一次性反向传播。
        losses = []

        # reversed(range(...)) 表示从最后一步倒着遍历。
        # 倒着算 G 很方便：G_t = r_t + gamma * G_(t+1)。
        for i in reversed(range(len(reward_list))):
            # 用递推公式计算当前时间步的折扣回报。
            G = self.gamma * G + reward_list[i]

            # 把当前时间步的状态转成 float32 numpy 数组。
            state = np.asarray(state_list[i], dtype=np.float32)

            # 把状态转成 tensor，并增加 batch 维度，形状变为 [1, state_dim]。
            state = torch.as_tensor(state, device=self.device).unsqueeze(0)

            # 把当前时间步实际执行过的动作转成 tensor。
            action = torch.tensor(action_list[i], device=self.device)

            # 再次用策略网络计算这个状态下的动作概率。
            probs = self.policy_net(state)

            # 构造离散概率分布，方便计算某个动作的 log_prob。
            action_dist = torch.distributions.Categorical(probs)

            # log_prob 表示“策略选择当时那个动作的对数概率”。
            log_prob = action_dist.log_prob(action)

            # REINFORCE 损失。
            # G 越大，说明这个动作越好，梯度下降会提高这个动作的概率。
            losses.append(-log_prob * G)

        # 清空上一轮更新留下来的梯度。PyTorch 的梯度默认会累加。
        self.optimizer.zero_grad()

        # 把每一步 loss 堆成 tensor 后求和，得到这一整条轨迹的总损失。
        loss = torch.stack(losses).sum()

        # 自动求导：计算 loss 对策略网络参数的梯度。
        loss.backward()

        # 优化器根据梯度更新策略网络参数。
        self.optimizer.step()


def moving_average(values, window_size):
    """
    计算移动平均，用来让回报曲线更平滑。

    怎么用：
        smooth_values = moving_average(return_list, 9)

    输入：
        values：原始数值列表，例如每局游戏的回报。
        window_size：窗口大小。窗口越大，曲线越平滑。

    输出：
        平滑后的 numpy 数组，长度和输入尽量保持一致。
    """

    # 把输入统一转成 float32 numpy 数组，方便后续做数学计算。
    values = np.asarray(values, dtype=np.float32)

    # 如果数据比窗口还短，就没必要平滑，直接返回原始数据。
    if len(values) < window_size:
        return values

    # 在数组开头插入 0，然后计算累积和。
    # 累积和可以高效计算任意连续窗口的和。
    cumulative_sum = np.cumsum(np.insert(values, 0, 0.0))

    # 计算中间主体部分的窗口平均值。
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size

    # radius 用来处理曲线开头和结尾，因为边界处凑不齐完整窗口。
    radius = np.arange(1, window_size - 1, 2)

    # 计算曲线开头部分的渐进平均。
    begin = np.cumsum(values[: window_size - 1])[::2] / radius

    # 计算曲线结尾部分的渐进平均，并反转回正确顺序。
    end = (np.cumsum(values[: -window_size : -1])[::2] / radius)[::-1]

    # 把开头、中间、结尾拼起来，得到完整平滑曲线。
    return np.concatenate((begin, middle, end))


def reset_env(env, seed=None):
    """
    重置环境，并兼容 Gymnasium 和旧 Gym 的不同返回格式。

    怎么用：
        state = reset_env(env, seed=0)

    Gymnasium:
        env.reset(seed=0) 返回 (state, info)

    旧 Gym:
        env.reset() 可能只返回 state
    """

    # 调用环境的 reset，让游戏回到初始状态。
    result = env.reset(seed=seed)

    # 新版 Gymnasium 返回 tuple: (state, info)，我们只需要 state。
    if isinstance(result, tuple):
        return result[0]

    # 旧版 Gym 直接返回 state。
    return result


def step_env(env, action):
    """
    执行动作，并兼容 Gymnasium 和旧 Gym 的不同 step 返回格式。

    怎么用：
        next_state, reward, done, info = step_env(env, action)

    Gymnasium:
        env.step(action) 返回
        (next_state, reward, terminated, truncated, info)

    旧 Gym:
        env.step(action) 返回
        (next_state, reward, done, info)
    """

    # 把动作交给环境，环境会推进一步。
    result = env.step(action)

    # 新版 Gymnasium 返回 5 个值。
    if len(result) == 5:
        # terminated 表示任务自然结束，比如杆倒了。
        # truncated 表示因为时间限制被截断，比如达到最大步数。
        next_state, reward, terminated, truncated, info = result

        # 训练循环只需要一个 done，所以二者有一个为 True 就算结束。
        return next_state, reward, terminated or truncated, info

    # 旧版 Gym 返回 4 个值，第三个值已经是 done。
    next_state, reward, done, info = result

    # 返回统一格式，方便 train 函数不用关心 Gym 版本。
    return next_state, reward, done, info


def train(env, agent, num_episodes):
    """
    训练主循环：让智能体玩很多局游戏，并不断更新策略。

    怎么用：
        return_list = train(env, agent, num_episodes=1000)

    输入：
        env：环境，例如 gym.make("CartPole-v1")。
        agent：REINFORCE 智能体。
        num_episodes：训练多少局游戏。

    输出：
        return_list：每一局游戏的总回报列表。

    训练流程：
        for 每一局 episode:
            1. reset 环境，得到初始 state。
            2. 不断用 agent.take_action(state) 选择动作。
            3. 用 env.step(action) 得到 reward 和 next_state。
            4. 把这一局的所有数据存进 transition_dict。
            5. 一局结束后调用 agent.update(transition_dict)。
    """

    # return_list 用来记录每一局的总奖励，后面画图需要它。
    return_list = []

    # 为了显示 10 段进度条，把总 episode 数大致分成 10 份。
    # max(1, ...) 可以避免 num_episodes 很小时变成 0。
    episodes_per_iteration = max(1, num_episodes // 10)

    # np.ceil 表示向上取整，确保所有 episode 都能被训练到。
    num_iterations = int(np.ceil(num_episodes / episodes_per_iteration))

    # 外层循环负责分段显示进度条。
    for iteration in range(num_iterations):
        # 当前分段的起始 episode 编号。
        start_episode = iteration * episodes_per_iteration

        # 当前分段的结束 episode 编号，不能超过 num_episodes。
        end_episode = min(num_episodes, start_episode + episodes_per_iteration)

        # 当前分段实际包含多少局。
        total = end_episode - start_episode

        # tqdm 会创建一个进度条，total 表示这一段要跑多少局。
        with tqdm(total=total, desc=f"Iteration {iteration}") as pbar:
            # 内层循环真正开始一局一局训练。
            for local_episode in range(total):
                # episode_return 记录当前这一局累计拿到多少奖励。
                episode_return = 0.0

                # transition_dict 保存当前这一整局轨迹。
                # REINFORCE 要等一整局结束后，用整条轨迹更新。
                transition_dict = {
                    "states": [],  # 每一步观察到的状态。
                    "actions": [],  # 每一步实际选择的动作。
                    "next_states": [],  # 每一步执行动作后的下一个状态。
                    "rewards": [],  # 每一步拿到的即时奖励。
                    "dones": [],  # 每一步后游戏是否结束。
                }

                # 重置环境，拿到初始状态。
                state = reset_env(env)

                # done 表示当前这一局是否结束。
                done = False

                # 只要这一局没结束，就继续和环境交互。
                while not done:
                    # 智能体根据当前状态选择动作。
                    action = agent.take_action(state)

                    # 环境执行动作，返回下一个状态、奖励、是否结束等信息。
                    next_state, reward, done, _ = step_env(env, action)

                    # 记录当前状态。
                    transition_dict["states"].append(state)

                    # 记录当前动作。
                    transition_dict["actions"].append(action)

                    # 记录下一个状态。
                    transition_dict["next_states"].append(next_state)

                    # 记录即时奖励。
                    transition_dict["rewards"].append(reward)

                    # 记录是否结束。
                    transition_dict["dones"].append(done)

                    # 状态向前推进，为下一步交互做准备。
                    state = next_state

                    # 累加这一局的总回报。
                    episode_return += reward

                # 一局结束，把这一局总回报放进列表。
                return_list.append(episode_return)

                # 用这一整局轨迹更新策略网络。
                agent.update(transition_dict)

                # 每 10 局，或者当前分段结束时，更新进度条右侧信息。
                if (local_episode + 1) % 10 == 0 or local_episode + 1 == total:
                    # 最近 10 局平均回报比单局回报更稳定，便于观察训练趋势。
                    recent_mean_return = np.mean(return_list[-10:])

                    # pbar.set_postfix 会在进度条右侧显示字典内容。
                    pbar.set_postfix(
                        {
                            "episode": start_episode + local_episode + 1,
                            "return": f"{recent_mean_return:.3f}",
                        }
                    )

                # 告诉 tqdm 当前已经完成 1 局。
                pbar.update(1)

    # 训练结束后返回每一局的回报。
    return return_list


def plot_returns(return_list, env_name, output_dir):
    """
    保存训练曲线图。

    怎么用：
        raw_path, smooth_path = plot_returns(return_list, "CartPole-v1", Path("results"))

    输入：
        return_list：每局游戏回报。
        env_name：环境名称，用在图标题里。
        output_dir：图片保存目录。

    输出：
        raw_path：原始回报曲线图片路径。
        smooth_path：平滑回报曲线图片路径。
    """

    # 如果输出目录不存在，就自动创建。parents=True 表示父目录也可以一起创建。
    output_dir.mkdir(parents=True, exist_ok=True)

    # episodes 是横轴，表示第几局游戏。
    episodes = list(range(len(return_list)))

    # 创建一张新图，避免不同图之间互相污染。
    plt.figure()

    # 画原始回报曲线。
    plt.plot(episodes, return_list)

    # 设置横轴标签。
    plt.xlabel("Episodes")

    # 设置纵轴标签。
    plt.ylabel("Returns")

    # 设置图标题。
    plt.title(f"REINFORCE on {env_name}")

    # 自动调整布局，防止文字被裁掉。
    plt.tight_layout()

    # 拼出原始曲线图的保存路径。
    raw_path = output_dir / "reinforce_returns.png"

    # 保存图片，dpi=150 表示图片清晰度。
    plt.savefig(raw_path, dpi=150)

    # 关闭当前图，释放内存。
    plt.close()

    # 创建第二张图，用来画平滑曲线。
    plt.figure()

    # 先计算移动平均，再画图。
    plt.plot(episodes, moving_average(return_list, 9))

    # 设置横轴标签。
    plt.xlabel("Episodes")

    # 设置纵轴标签。
    plt.ylabel("Returns")

    # 设置图标题。
    plt.title(f"REINFORCE on {env_name}")

    # 自动调整布局。
    plt.tight_layout()

    # 拼出平滑曲线图的保存路径。
    smooth_path = output_dir / "reinforce_returns_smoothed.png"

    # 保存平滑曲线图。
    plt.savefig(smooth_path, dpi=150)

    # 关闭当前图。
    plt.close()

    # 把两个图片路径返回，方便 main 函数打印出来。
    return raw_path, smooth_path


def parse_args():
    """
    读取命令行参数。

    怎么用：
        args = parse_args()
        print(args.num_episodes)

    用户可以在命令行里改超参数，例如：
        python .\\Reinforce\\Reinforce.py --num-episodes 200 --learning-rate 0.0005

    如果用户没有传参数，就使用这里写好的默认值。
    """

    # 创建参数解析器，description 会显示在 --help 帮助信息里。
    parser = argparse.ArgumentParser(description="Train REINFORCE on CartPole.")

    # 环境名称。CartPole-v1 满分通常是 500，CartPole-v0 满分通常是 200。
    parser.add_argument("--env-name", type=str, default="CartPole-v1")

    # 训练多少局游戏。教程里常用 1000。
    parser.add_argument("--num-episodes", type=int, default=1000)

    # 策略网络隐藏层大小。
    parser.add_argument("--hidden-dim", type=int, default=128)

    # 学习率，控制每次参数更新的步子大小。
    parser.add_argument("--learning-rate", type=float, default=1e-3)

    # 折扣因子，控制未来奖励的重要程度。
    parser.add_argument("--gamma", type=float, default=0.98)

    # 随机种子，用来让实验尽量可复现。
    parser.add_argument("--seed", type=int, default=0)

    # 结果图片保存目录。默认放在当前 py 文件旁边的 results 文件夹里。
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )

    # 解析命令行参数，并返回一个 args 对象。
    return parser.parse_args()


def main():
    """
    程序入口。

    Python 执行这个文件时，会从最下面的 if __name__ == "__main__" 进入 main。

    main 的职责是把所有组件组装起来：
    1. 读取参数。
    2. 设置随机种子和运行设备。
    3. 创建环境。
    4. 创建 REINFORCE 智能体。
    5. 开始训练。
    6. 保存训练曲线。
    """

    # 读取命令行参数。
    args = parse_args()

    # episode 数必须是正数，否则训练循环没有意义。
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive.")

    # 如果电脑有可用 GPU，就用 cuda；否则用 cpu。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置 numpy 随机种子，影响 numpy 相关随机行为。
    np.random.seed(args.seed)

    # 设置 PyTorch CPU 随机种子。
    torch.manual_seed(args.seed)

    # 如果使用 GPU，也设置 PyTorch GPU 随机种子。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 创建环境。例如 CartPole-v1。
    env = gym.make(args.env_name)

    # CartPole 的状态空间是 Box，动作空间是 Discrete。
    # 这里做类型检查有两个好处：
    # 1. 如果换成不适合当前代码的环境，可以尽早给出清晰错误。
    # 2. 告诉 Pylance 后面可以安全访问 shape[0] 和 n。
    if not isinstance(env.observation_space, Box):
        raise TypeError("This example expects a Box observation space.")
    if not isinstance(env.action_space, Discrete):
        raise TypeError("This example expects a Discrete action space.")

    # 经过上面的 isinstance 检查后，这两个变量的类型已经被 Pylance 收窄。
    observation_space = env.observation_space
    action_space = env.action_space

    # Box 空间正常都会有 shape，例如 CartPole 是 (4,)。
    # 这里显式检查 None，是为了让静态检查器知道后面可以安全下标。
    observation_shape = observation_space.shape
    if observation_shape is None:
        raise TypeError("Observation space shape must not be None.")
    if len(observation_shape) == 0:
        raise TypeError("Observation space shape must have at least one dimension.")

    # 用固定种子重置一次环境，让初始随机性尽量可复现。
    reset_env(env, seed=args.seed)

    # 给动作空间设置随机种子，影响 action_space 内部采样。
    action_space.seed(args.seed)

    # observation_space.shape[0] 是状态向量维度。
    state_dim = int(observation_shape[0])

    # action_space.n 是离散动作数量。
    action_dim = int(action_space.n)

    # 创建 REINFORCE 智能体。
    agent = REINFORCE(
        state_dim,  # 状态维度，作为策略网络输入大小。
        args.hidden_dim,  # 隐藏层大小。
        action_dim,  # 动作数量，作为策略网络输出大小。
        args.learning_rate,  # 学习率。
        args.gamma,  # 折扣因子。
        device,  # cpu 或 cuda。
    )

    # 开始训练，返回每一局的总回报。
    return_list = train(env, agent, args.num_episodes)

    # 训练结束后关闭环境，释放资源。
    env.close()

    # 保存原始曲线和平滑曲线。
    raw_path, smooth_path = plot_returns(return_list, args.env_name, args.output_dir)

    # 打印当前使用的设备。
    print(f"Training finished on {device}.")

    # 打印最近 10 局的平均回报，作为一个简单的训练效果指标。
    print(f"Last 10 episode mean return: {np.mean(return_list[-10:]):.3f}")

    # 打印图片保存位置。
    print(f"Saved plots: {raw_path}, {smooth_path}")


# 这个判断表示：只有直接运行本文件时才执行 main。
# 如果别的 Python 文件 import 了这个文件，则不会自动开始训练。
if __name__ == "__main__":
    main()
