"""
TRPO 车杆实验（离散动作版）
================================

这个文件是一个可以直接运行的 TRPO（Trust Region Policy Optimization，信任区域策略优化）
示例，环境是 CartPole-v1，也就是经典的“车杆”任务。

一、这个算法整体在做什么？
------------------------
CartPole 的状态是 4 个数字：
1. 小车位置
2. 小车速度
3. 杆子角度
4. 杆子角速度

CartPole 的动作是 2 个离散动作：
0. 小车向左推
1. 小车向右推

TRPO 训练两个神经网络：
1. actor，也叫策略网络：
   输入状态，输出每个动作的概率，例如 [0.45, 0.55]。
   智能体根据这个概率分布采样动作。

2. critic，也叫价值网络：
   输入状态，输出这个状态大概“值多少钱”，也就是从这个状态开始未来奖励的估计。
   critic 用来帮助 actor 判断“刚才这个动作比平均水平好还是差”。

二、TRPO 的一次训练更新长什么样？
------------------------------
1. 用当前 actor 和环境交互一整局，收集 state/action/reward/next_state/done。
2. 用 critic 计算 TD 误差。
3. 用 GAE（广义优势估计）把 TD 误差整理成 advantage。
4. 构造策略目标函数：新策略比旧策略更倾向于选择高 advantage 的动作。
5. 但不能让新策略离旧策略太远，所以用 KL 散度限制更新范围。
6. 用共轭梯度近似求解自然梯度方向。
7. 用线性搜索找一个既提升目标、又满足 KL 限制的步长。
8. 更新 actor，再用均方误差更新 critic。

三、怎么运行？
------------
短测试：
python "TRPO\\demo\\TRPO 车杆.py" --num-episodes 5

完整训练：
python "TRPO\\demo\\TRPO 车杆.py"

常用参数：
--num-episodes 训练多少局
--env-name 环境名，默认 CartPole-v1
--output-dir 曲线图片保存目录
"""

import argparse  # 用来读取命令行参数，例如 --num-episodes 100
import copy  # 用来深拷贝 actor，线性搜索时临时测试新参数
from pathlib import Path  # 用面向对象的方式处理文件夹和文件路径

import gymnasium as gym  # 强化学习环境库，这里用 CartPole-v1
import matplotlib.pyplot as plt  # 画训练曲线并保存成图片
import numpy as np  # 处理数组、随机种子、移动平均等
import torch  # PyTorch 主库，用来搭神经网络和自动求梯度
import torch.nn.functional as F  # 常用神经网络函数，例如 relu、mse_loss
from tqdm import tqdm  # 显示训练进度条


class PolicyNet(torch.nn.Module):
    """离散动作策略网络。

    用法：
        net = PolicyNet(state_dim=4, hidden_dim=128, action_dim=2)
        probs = net(states)

    输入：
        states: 形状一般是 [batch_size, state_dim]。

    输出：
        probs: 形状是 [batch_size, action_dim]，每一行都是动作概率。

    在 CartPole 中，action_dim=2，所以输出类似：
        [[0.48, 0.52],
         [0.80, 0.20]]
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()  # 初始化 torch.nn.Module 的内部机制
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 第一层：状态 -> 隐藏层
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)  # 第二层：隐藏层 -> 动作分数

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 先做线性变换，再用 ReLU 增加非线性表达能力
        return F.softmax(self.fc2(x), dim=1)  # 把动作分数变成概率，dim=1 表示按每一行归一化


class ValueNet(torch.nn.Module):
    """状态价值网络。

    用法：
        net = ValueNet(state_dim=4, hidden_dim=128)
        values = net(states)

    输入：
        states: 形状一般是 [batch_size, state_dim]。

    输出：
        values: 形状是 [batch_size, 1]，表示每个状态的价值估计 V(s)。

    critic 的作用：
        actor 只知道“这次得到了多少奖励”，但不知道这个动作是否真的比平均水平好。
        critic 估计 V(s)，然后 TD 误差可以告诉 actor：这一步比预期好还是差。
    """

    def __init__(self, state_dim, hidden_dim):
        super().__init__()  # 初始化 torch.nn.Module
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 第一层：状态 -> 隐藏层
        self.fc2 = torch.nn.Linear(hidden_dim, 1)  # 第二层：隐藏层 -> 一个价值数字

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 用 ReLU 处理隐藏层输出
        return self.fc2(x)  # 输出状态价值 V(s)，不需要 softmax


def compute_advantage(gamma, lmbda, td_delta):
    """用 GAE 计算优势函数 advantage。

    用法：
        advantage = compute_advantage(gamma=0.98, lmbda=0.95, td_delta=td_delta)

    参数：
        gamma: 折扣因子，越接近 1 越重视长期奖励。
        lmbda: GAE 参数，控制“短期 TD”和“长期累计 TD”的折中。
        td_delta: 每一步的 TD 误差，形状一般是 [T, 1]。

    返回：
        advantage: 每一步的优势估计，形状是 [T, 1]。

    直觉：
        advantage > 0 说明这个动作比 critic 原本预期更好，actor 应该更愿意选它。
        advantage < 0 说明这个动作比预期更差，actor 应该降低它的概率。
    """

    td_delta = td_delta.detach().cpu().numpy().flatten()  # 从计算图中取出 TD 误差，转成一维 numpy 数组
    advantage_list = []  # 用来保存从后往前算出的 advantage
    advantage = 0.0  # 递推变量，表示当前累计到的优势
    for delta in td_delta[::-1]:  # GAE 要从轨迹末尾倒着往前计算
        advantage = gamma * lmbda * advantage + delta  # 递推公式：A_t = delta_t + gamma*lambda*A_{t+1}
        advantage_list.append(advantage)  # 保存当前时间步的优势
    advantage_list.reverse()  # 因为刚才是倒着算的，所以反转回正常时间顺序
    return torch.tensor(advantage_list, dtype=torch.float32).view(-1, 1)  # 转回 PyTorch 张量，形状整理成 [T, 1]


class TRPO:
    """TRPO 智能体，负责动作采样和网络更新。

    用法：
        agent = TRPO(...)
        action = agent.take_action(state)
        agent.update(transition_dict)

    这个类把 TRPO 的核心步骤都封装起来：
    1. take_action：用 actor 根据状态采样动作。
    2. update：用一条完整轨迹更新 critic 和 actor。
    3. policy_learn：TRPO 的策略更新主流程。
    4. conjugate_gradient：共轭梯度，近似求自然梯度方向。
    5. line_search：线性搜索，确保新策略没有走出信任区域。
    """

    def __init__(
        self,
        hidden_dim,
        state_space,
        action_space,
        lmbda,
        kl_constraint,
        alpha,
        critic_lr,
        gamma,
        device,
    ):
        state_dim = state_space.shape[0]  # CartPole 状态维度是 4
        action_dim = action_space.n  # CartPole 离散动作数量是 2
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)  # 创建策略网络并放到 CPU/GPU
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 创建价值网络并放到 CPU/GPU
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)  # critic 用 Adam 更新
        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # GAE 的 lambda
        self.kl_constraint = kl_constraint  # TRPO 信任区域大小，也就是 KL 最大允许值
        self.alpha = alpha  # 线性搜索缩放系数，例如 0.5 表示每次步长减半
        self.device = device  # 保存训练设备，可能是 cuda 或 cpu

    def take_action(self, state):
        """根据一个状态采样一个动作。

        用法：
            action = agent.take_action(state)

        在训练时，我们不是永远选概率最大的动作，而是按概率随机采样。
        这样智能体可以探索不同动作。
        """

        state = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device)  # 把环境状态转成 float32 张量
        probs = self.actor(state.unsqueeze(0))  # 神经网络需要 batch 维度，所以用 unsqueeze(0) 变成 [1, state_dim]
        action_dist = torch.distributions.Categorical(probs=probs)  # 用动作概率创建离散分布
        return action_dist.sample().item()  # 从分布中采样动作，并转成 Python int

    def hessian_matrix_vector_product(
        self, states, old_action_dists, vector, damping=0.1
    ):
        """计算 H*v，其中 H 是平均 KL 散度的 Hessian 矩阵。

        用法：
            hvp = self.hessian_matrix_vector_product(states, old_action_dists, vector)

        为什么需要它？
            TRPO 理论上要用 H 的逆矩阵，但神经网络参数很多，显式构造 H 很贵。
            共轭梯度只需要能算 H*v，不需要真的把 H 矩阵写出来。

        damping：
            给结果加 damping * vector，让数值更稳定，防止除零或方向异常。
        """

        new_action_dists = torch.distributions.Categorical(probs=self.actor(states))  # 当前 actor 产生的新动作分布
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists)
        )  # 计算旧策略和新策略之间的平均 KL 散度
        actor_params = tuple(self.actor.parameters())  # Pylance 更喜欢明确的 tuple，而不是 parameters() 迭代器
        kl_grad = torch.autograd.grad(
            kl, actor_params, create_graph=True, allow_unused=False
        )  # 对 KL 求一阶梯度；create_graph=True 允许后面继续求二阶梯度
        kl_grad_vector = torch.cat([grad.contiguous().view(-1) for grad in kl_grad])  # 把所有参数梯度拼成一个长向量
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)  # 先算 grad(KL) 与向量 v 的点积
        grad2 = torch.autograd.grad(kl_grad_vector_product, actor_params)  # 再对点积求梯度，得到 Hessian-vector product
        grad2_vector = torch.cat([grad.contiguous().view(-1) for grad in grad2])  # 把二阶梯度也拼成一个长向量
        return grad2_vector + damping * vector  # 返回 H*v，并加阻尼项提高稳定性

    def conjugate_gradient(self, grad, states, old_action_dists, max_iterations=10):
        """用共轭梯度法近似求解 H*x = grad。

        用法：
            direction = self.conjugate_gradient(obj_grad, states, old_action_dists)

        输入：
            grad: 策略目标函数对 actor 参数的梯度，也就是 g。

        输出：
            x: 近似的 H^{-1}g，也就是 TRPO 的自然梯度方向。

        你可以把它理解成：
            普通梯度只看“目标函数往哪里变大”。
            自然梯度还考虑“策略分布改变得有多大”，所以更新更稳。
        """

        x = torch.zeros_like(grad)  # 初始解 x=0
        r = grad.clone()  # 初始残差 r = b - A*x，因为 x=0，所以 r=grad
        p = grad.clone()  # 初始搜索方向 p=r
        rdotr = torch.dot(r, r)  # 残差平方，用来判断是否收敛
        for _ in range(max_iterations):  # 通常 10 次就够用，不需要精确求解
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)  # 计算 H*p
            alpha = rdotr / torch.dot(p, Hp).clamp_min(1e-8)  # 计算这次沿 p 方向走多远
            x += alpha * p  # 更新当前解
            r -= alpha * Hp  # 更新残差
            new_rdotr = torch.dot(r, r)  # 新残差平方
            if new_rdotr < 1e-10:  # 残差足够小就提前停止
                break
            beta = new_rdotr / rdotr  # 计算下一个搜索方向的混合系数
            p = r + beta * p  # 更新搜索方向
            rdotr = new_rdotr  # 保存新残差平方，供下一轮使用
        return x  # 返回近似解，也就是更新方向

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):
        """计算 TRPO 的替代目标函数。

        用法：
            obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)

        目标函数大致是：
            E[ 新策略概率 / 旧策略概率 * advantage ]

        如果 advantage > 0：
            希望新策略增加这个动作的概率。

        如果 advantage < 0：
            希望新策略降低这个动作的概率。
        """

        log_probs = torch.log(actor(states).gather(1, actions).clamp_min(1e-8))  # 取出实际动作对应的概率并求 log
        ratio = torch.exp(log_probs - old_log_probs)  # exp(log新概率 - log旧概率) = 新概率 / 旧概率
        return torch.mean(ratio * advantage)  # 对所有时间步求平均，得到策略目标

    def line_search(
        self, states, actions, advantage, old_log_probs, old_action_dists, max_vec
    ):
        """线性搜索，找到一个安全的新 actor 参数。

        用法：
            new_para = self.line_search(...)

        为什么需要线性搜索？
            共轭梯度和二阶近似给出的更新方向不一定百分百安全。
            所以我们从最大步长开始尝试，如果不满足条件，就不断缩小步长。

        接受新参数的两个条件：
            1. 新目标函数比旧目标函数更大。
            2. 新旧策略的 KL 散度小于 kl_constraint。
        """

        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters()
        )  # 把 actor 所有参数拉平成一个长向量，便于整体加更新量
        old_obj = self.compute_surrogate_obj(
            states, actions, advantage, old_log_probs, self.actor
        )  # 计算旧策略目标值，后面要比较是否提升
        for i in range(15):  # 最多尝试 15 个不同步长
            coef = self.alpha**i  # 第 i 次尝试的缩放系数，例如 1、0.5、0.25 ...
            new_para = old_para + coef * max_vec  # 沿 TRPO 方向走一小步
            new_actor = copy.deepcopy(self.actor)  # 拷贝一个临时 actor，用来测试新参数
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters()
            )  # 把长向量参数写回临时 actor
            new_action_dists = torch.distributions.Categorical(
                probs=new_actor(states)
            )  # 用临时 actor 计算新动作分布
            kl_div = torch.mean(
                torch.distributions.kl.kl_divergence(
                    old_action_dists, new_action_dists
                )
            )  # 计算旧策略和临时新策略的平均 KL 散度
            new_obj = self.compute_surrogate_obj(
                states, actions, advantage, old_log_probs, new_actor
            )  # 计算临时新策略的目标函数
            if new_obj.item() > old_obj.item() and kl_div.item() < self.kl_constraint:  # 同时满足提升和 KL 限制
                return new_para  # 接受这组新参数
        return old_para  # 如果所有尝试都不安全，就保持旧参数不变

    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage):
        """更新 actor，也就是 TRPO 最核心的策略学习步骤。

        用法：
            self.policy_learn(states, actions, old_action_dists, old_log_probs, advantage)

        步骤：
            1. 计算策略替代目标。
            2. 求目标对 actor 参数的梯度 g。
            3. 用共轭梯度求 H^{-1}g。
            4. 根据 KL 约束计算最大步长。
            5. 用线性搜索确认最终可接受的参数。
        """

        surrogate_obj = self.compute_surrogate_obj(
            states, actions, advantage, old_log_probs, self.actor
        )  # 计算当前 actor 的策略目标
        actor_params = tuple(self.actor.parameters())  # 显式保存参数 tuple，方便 autograd 和类型检查器识别
        grads = torch.autograd.grad(surrogate_obj, actor_params)  # 对策略目标求梯度
        obj_grad = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()  # 拼成一个长向量，并断开计算图

        direction = self.conjugate_gradient(obj_grad, states, old_action_dists)  # 近似求自然梯度方向 H^{-1}g
        Hd = self.hessian_matrix_vector_product(states, old_action_dists, direction)  # 计算 H*d，用来估计 KL 步长
        max_coef = torch.sqrt(
            2 * self.kl_constraint / torch.dot(direction, Hd).clamp_min(1e-8)
        )  # 根据二阶 KL 近似计算最大可走系数
        new_para = self.line_search(
            states,
            actions,
            advantage,
            old_log_probs,
            old_action_dists,
            direction * max_coef,
        )  # 在线性搜索中找最终参数
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor.parameters()
        )  # 把最终参数真正写回 actor

    def update(self, transition_dict):
        """用一条完整轨迹更新 critic 和 actor。

        用法：
            transition_dict = {
                "states": [...],
                "actions": [...],
                "next_states": [...],
                "rewards": [...],
                "dones": [...]
            }
            agent.update(transition_dict)

        注意：
            TRPO 是 on-policy 算法。
            也就是说，当前这批数据必须来自当前 actor，不能像 DQN 那样反复用旧数据。
        """

        states = torch.as_tensor(
            np.asarray(transition_dict["states"], dtype=np.float32),
            device=self.device,
        )  # 把状态列表转成 [T, state_dim] 张量
        actions = torch.as_tensor(
            transition_dict["actions"], dtype=torch.long, device=self.device
        ).view(-1, 1)  # 把动作列表转成 [T, 1]，离散动作要用 long 类型
        rewards = torch.as_tensor(
            transition_dict["rewards"], dtype=torch.float32, device=self.device
        ).view(-1, 1)  # 把奖励列表转成 [T, 1]
        next_states = torch.as_tensor(
            np.asarray(transition_dict["next_states"], dtype=np.float32),
            device=self.device,
        )  # 把下一状态列表转成 [T, state_dim]
        dones = torch.as_tensor(
            transition_dict["dones"], dtype=torch.float32, device=self.device
        ).view(-1, 1)  # done=True 表示一局结束，这里转成 1.0；False 转成 0.0

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)  # TD 目标：r + gamma*V(s')
        td_delta = td_target - self.critic(states)  # TD 误差：目标价值 - 当前价值
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta).to(self.device)  # 用 GAE 算 advantage
        advantage = (advantage - advantage.mean()) / (
            advantage.std(unbiased=False) + 1e-8
        )  # 标准化 advantage，让训练更稳定

        with torch.no_grad():  # 旧策略只用于比较，不需要保存梯度
            old_probs = self.actor(states)  # 用更新前的 actor 计算旧动作概率
            old_log_probs = torch.log(old_probs.gather(1, actions).clamp_min(1e-8))  # 记录旧策略下实际动作的 log 概率
        old_action_dists = torch.distributions.Categorical(probs=old_probs)  # 构造旧策略分布，用于 KL 约束

        critic_loss = F.mse_loss(self.critic(states), td_target.detach())  # critic 用均方误差拟合 TD 目标
        self.critic_optimizer.zero_grad()  # 清空 critic 上一次的梯度
        critic_loss.backward()  # 反向传播，计算 critic 梯度
        self.critic_optimizer.step()  # Adam 更新 critic 参数

        self.policy_learn(states, actions, old_action_dists, old_log_probs, advantage)  # 用 TRPO 更新 actor


def reset_env(env, seed=None):
    """兼容不同 Gym API 的 reset 函数。

    用法：
        state = reset_env(env, seed=0)
        state = reset_env(env)

    gymnasium 的 reset 返回 (state, info)。
    老版本 gym 的 reset 只返回 state。
    这个函数把两种情况统一成只返回 state。
    """

    result = env.reset(seed=seed) if seed is not None else env.reset()  # 如果给了 seed，就用固定随机种子重置环境
    return result[0] if isinstance(result, tuple) else result  # 新 API 返回 tuple，所以取第一个元素 state


def step_env(env, action):
    """兼容不同 Gym API 的 step 函数。

    用法：
        next_state, reward, done, info = step_env(env, action)

    gymnasium 返回：
        next_state, reward, terminated, truncated, info

    老版本 gym 返回：
        next_state, reward, done, info

    这个函数统一成：
        next_state, reward, done, info
    """

    result = env.step(action)  # 把动作交给环境，环境前进一步
    if len(result) == 5:  # gymnasium 新 API 有 5 个返回值
        next_state, reward, terminated, truncated, info = result  # terminated 是自然结束，truncated 是达到时间上限
        return next_state, reward, terminated or truncated, info  # 两种结束都算 done
    next_state, reward, done, info = result  # 老 API 已经直接给 done
    return next_state, reward, done, info  # 返回统一格式


def train_on_policy_agent(env, agent, num_episodes):
    """训练 on-policy 智能体。

    用法：
        returns = train_on_policy_agent(env, agent, num_episodes=500)

    这个函数负责：
        1. 跑一局游戏。
        2. 收集这一局的所有转移数据。
        3. 一局结束后调用 agent.update。
        4. 记录每一局总回报。

    为什么一局结束才 update？
        GAE 需要从轨迹末尾往前算 advantage，所以这里按完整 episode 更新。
    """

    return_list = []  # 保存每一局的总奖励
    episodes_per_iteration = max(1, num_episodes // 10)  # 把训练过程分成约 10 段显示进度
    num_iterations = int(np.ceil(num_episodes / episodes_per_iteration))  # 计算实际需要多少段

    for iteration in range(num_iterations):  # 外层循环负责显示第几个进度段
        start_episode = iteration * episodes_per_iteration  # 当前进度段起始 episode 编号
        end_episode = min(num_episodes, start_episode + episodes_per_iteration)  # 当前进度段结束 episode 编号
        with tqdm(total=end_episode - start_episode, desc=f"Iteration {iteration}") as pbar:  # 创建进度条
            for local_episode in range(end_episode - start_episode):  # 当前进度段内逐局训练
                episode_return = 0.0  # 当前这一局累计奖励
                transition_dict = {
                    "states": [],
                    "actions": [],
                    "next_states": [],
                    "rewards": [],
                    "dones": [],
                }  # 用字典保存一整局轨迹

                state = reset_env(env)  # 重置环境，拿到初始状态
                done = False  # done=False 表示这一局还没结束
                while not done:  # 一直交互到这一局结束
                    action = agent.take_action(state)  # 智能体根据当前状态采样动作
                    next_state, reward, done, _ = step_env(env, action)  # 环境执行动作，返回下一状态和奖励
                    transition_dict["states"].append(state)  # 保存当前状态
                    transition_dict["actions"].append(action)  # 保存采取的动作
                    transition_dict["next_states"].append(next_state)  # 保存下一状态
                    transition_dict["rewards"].append(reward)  # 保存奖励
                    transition_dict["dones"].append(done)  # 保存是否结束
                    state = next_state  # 状态前进到下一步
                    episode_return += reward  # 累加这一局奖励

                return_list.append(episode_return)  # 一局结束后记录总回报
                agent.update(transition_dict)  # 用这一局数据更新 TRPO 智能体
                if (local_episode + 1) % 10 == 0 or local_episode + 1 == end_episode - start_episode:  # 每 10 局或最后一局刷新显示
                    pbar.set_postfix(
                        {
                            "episode": start_episode + local_episode + 1,
                            "return": f"{np.mean(return_list[-10:]):.3f}",
                        }
                    )  # 在进度条后显示最近 10 局平均回报
                pbar.update(1)  # 进度条前进一步
    return return_list  # 返回所有 episode 的回报，后面用于画图


def moving_average(values, window_size):
    """计算移动平均，让训练曲线更平滑。

    用法：
        smoothed = moving_average(return_list, 9)

    输入：
        values: 原始回报列表。
        window_size: 平滑窗口大小。

    输出：
        平滑后的 numpy 数组。
    """

    values = np.asarray(values, dtype=np.float32)  # 转成 numpy 数组，方便做累加和
    if len(values) < window_size:  # 如果数据还不够一个窗口
        return values  # 直接返回原始数据
    cumulative_sum = np.cumsum(np.insert(values, 0, 0.0))  # 前缀和，用来快速算窗口平均
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size  # 中间部分的标准移动平均
    radius = np.arange(1, window_size - 1, 2)  # 两端用较小窗口，避免曲线长度明显变短
    begin = np.cumsum(values[: window_size - 1])[::2] / radius  # 开头部分的平滑值
    end = (np.cumsum(values[: -window_size : -1])[::2] / radius)[::-1]  # 结尾部分的平滑值
    return np.concatenate((begin, middle, end))  # 拼回完整平滑曲线


def plot_returns(return_list, env_name, output_dir):
    """保存原始回报曲线和平滑回报曲线。

    用法：
        raw_path, smooth_path = plot_returns(return_list, "CartPole-v1", Path("results"))

    输出：
        raw_path: 原始曲线图片路径。
        smooth_path: 平滑曲线图片路径。
    """

    output_dir.mkdir(parents=True, exist_ok=True)  # 如果输出目录不存在，就自动创建
    episodes = list(range(len(return_list)))  # x 轴是 episode 编号

    raw_path = output_dir / "trpo_cartpole_returns.png"  # 原始曲线保存路径
    smooth_path = output_dir / "trpo_cartpole_returns_smoothed.png"  # 平滑曲线保存路径

    plt.figure()  # 新建一张图
    plt.plot(episodes, return_list)  # 画原始回报曲线
    plt.xlabel("Episodes")  # x 轴标签
    plt.ylabel("Returns")  # y 轴标签
    plt.title(f"TRPO on {env_name}")  # 图标题
    plt.tight_layout()  # 自动调整边距，防止文字被裁掉
    plt.savefig(raw_path, dpi=150)  # 保存图片
    plt.close()  # 关闭当前图，释放内存

    plt.figure()  # 新建第二张图
    plt.plot(episodes, moving_average(return_list, 9))  # 画平滑回报曲线
    plt.xlabel("Episodes")  # x 轴标签
    plt.ylabel("Returns")  # y 轴标签
    plt.title(f"TRPO on {env_name}")  # 图标题
    plt.tight_layout()  # 自动调整布局
    plt.savefig(smooth_path, dpi=150)  # 保存平滑曲线图片
    plt.close()  # 关闭当前图

    return raw_path, smooth_path  # 返回两个图片路径，方便主函数打印


def parse_args():
    """读取命令行参数。

    用法：
        args = parse_args()

    你可以在终端里这样改参数：
        python "TRPO\\demo\\TRPO 车杆.py" --num-episodes 100 --hidden-dim 64

    如果你什么都不传，就使用这里写好的默认值。
    """

    parser = argparse.ArgumentParser(description="Train TRPO on CartPole.")  # 创建参数解析器
    parser.add_argument("--env-name", type=str, default="CartPole-v1")  # 环境名，默认新版 CartPole-v1
    parser.add_argument("--num-episodes", type=int, default=500)  # 训练局数
    parser.add_argument("--hidden-dim", type=int, default=128)  # 神经网络隐藏层大小
    parser.add_argument("--gamma", type=float, default=0.98)  # 折扣因子
    parser.add_argument("--lmbda", type=float, default=0.95)  # GAE lambda
    parser.add_argument("--critic-lr", type=float, default=1e-2)  # critic 学习率
    parser.add_argument("--kl-constraint", type=float, default=5e-4)  # TRPO 的 KL 约束大小
    parser.add_argument("--alpha", type=float, default=0.5)  # 线性搜索步长缩放系数
    parser.add_argument("--seed", type=int, default=0)  # 随机种子，方便复现实验
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )  # 曲线图片输出目录
    return parser.parse_args()  # 解析终端参数并返回


def main():
    """程序入口。

    用法：
        直接运行这个文件时，Python 会执行 main()。

    主流程：
        1. 读取参数。
        2. 设置随机种子和训练设备。
        3. 创建环境。
        4. 创建 TRPO 智能体。
        5. 训练智能体。
        6. 保存训练曲线。
    """

    args = parse_args()  # 读取命令行参数
    if args.num_episodes <= 0:  # episode 数量必须是正数
        raise ValueError("--num-episodes must be positive.")  # 如果参数非法，直接报错提醒

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先使用 GPU，没有 GPU 就用 CPU
    np.random.seed(args.seed)  # 设置 numpy 随机种子
    torch.manual_seed(args.seed)  # 设置 PyTorch CPU 随机种子
    if torch.cuda.is_available():  # 如果有 GPU
        torch.cuda.manual_seed_all(args.seed)  # 设置 PyTorch GPU 随机种子

    env = gym.make(args.env_name)  # 创建 CartPole 环境
    reset_env(env, seed=args.seed)  # 用随机种子初始化环境
    if hasattr(env.action_space, "seed"):  # 有些环境的动作空间也可以设置随机种子
        env.action_space.seed(args.seed)  # 设置动作空间随机种子，让采样更可复现

    agent = TRPO(
        args.hidden_dim,
        env.observation_space,
        env.action_space,
        args.lmbda,
        args.kl_constraint,
        args.alpha,
        args.critic_lr,
        args.gamma,
        device,
    )  # 创建 TRPO 智能体
    return_list = train_on_policy_agent(env, agent, args.num_episodes)  # 开始训练，并拿到每局回报
    env.close()  # 关闭环境，释放资源

    raw_path, smooth_path = plot_returns(return_list, args.env_name, args.output_dir)  # 保存训练曲线
    print(f"Training finished on {device}.")  # 打印训练设备
    print(f"Last 10 episode mean return: {np.mean(return_list[-10:]):.3f}")  # 打印最近 10 局平均回报
    print(f"Saved plots: {raw_path}, {smooth_path}")  # 打印图片保存位置


if __name__ == "__main__":  # 只有直接运行这个文件时才执行 main；被 import 时不会自动训练
    main()  # 调用主函数
