"""
TRPO 倒立杆实验（连续动作版）
================================

这个文件是一个可以直接运行的 TRPO（Trust Region Policy Optimization，信任区域策略优化）
示例，环境是 Pendulum-v1，也就是经典的连续控制任务“倒立摆/倒立杆”。

一、这个环境和 CartPole 有什么不同？
--------------------------------
CartPole 的动作是离散的：
    0 表示向左推，1 表示向右推。

Pendulum 的动作是连续的：
    动作是一个实数，表示施加的力矩，通常范围是 [-2, 2]。

因此，连续动作版 actor 不能输出“每个动作的概率”，因为连续动作有无限多个。
常见做法是让 actor 输出一个高斯分布：
    mu  表示均值，也就是当前最想选的动作附近。
    std 表示标准差，也就是探索范围有多大。

训练时从 Normal(mu, std) 里采样动作，再交给环境。

二、这个算法整体架构是什么？
------------------------
本文件依旧训练两个网络：

1. actor（策略网络）：
   输入状态，输出连续动作高斯分布的 mu 和 std。
   采样动作时：
       action ~ Normal(mu, std)

2. critic（价值网络）：
   输入状态，输出状态价值 V(s)。
   用来计算 TD 误差和 advantage。

三、TRPO 更新流程
----------------
1. 用当前 actor 跑一整局，收集轨迹。
2. critic 计算 TD 目标和 TD 误差。
3. GAE 根据 TD 误差计算 advantage。
4. actor 构造替代目标：
       E[ exp(log_prob_new - log_prob_old) * advantage ]
5. 用 KL 散度限制新旧高斯策略之间的距离。
6. 用 Hessian-vector product 和共轭梯度求自然梯度方向。
7. 用线性搜索找到安全更新步长。
8. 更新 actor，并用 MSE 更新 critic。

四、怎么运行？
------------
短测试：
python "TRPO\\demo\\TRPO 倒立杆.py" --num-episodes 5

完整训练：
python "TRPO\\demo\\TRPO 倒立杆.py"

连续控制训练通常比 CartPole 慢，完整 2000 局需要一些时间。
"""

import argparse  # 读取命令行参数，例如 --num-episodes 100
import copy  # 深拷贝 actor，线性搜索时用临时网络试参数
from pathlib import Path  # 处理结果保存路径

import gymnasium as gym  # 强化学习环境库，这里用 Pendulum-v1
import matplotlib.pyplot as plt  # 画训练曲线
import numpy as np  # 数组处理、随机种子、移动平均
import torch  # PyTorch 主库
import torch.nn.functional as F  # relu、softplus、mse_loss 等函数
from tqdm import tqdm  # 训练进度条


class PolicyNetContinuous(torch.nn.Module):
    """连续动作策略网络。

    用法：
        net = PolicyNetContinuous(state_dim=3, hidden_dim=128, action_dim=1, action_bound=[2.0])
        mu, std = net(states)

    输入：
        states: 形状一般是 [batch_size, state_dim]。

    输出：
        mu:  动作高斯分布均值，形状是 [batch_size, action_dim]。
        std: 动作高斯分布标准差，形状是 [batch_size, action_dim]。

    为什么输出 mu 和 std？
        连续动作不能像离散动作那样列出每个动作概率。
        所以我们让策略表示一个概率分布，再从分布中采样动作。
    """

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()  # 初始化 torch.nn.Module
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 第一层：状态 -> 隐藏层
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)  # 输出高斯均值 mu
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)  # 输出高斯标准差 std 的原始值
        self.action_bound = torch.as_tensor(action_bound, dtype=torch.float32)  # 保存动作最大幅度，例如 Pendulum 是 2

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 先经过隐藏层和 ReLU
        action_bound = self.action_bound.to(x.device)  # 把动作范围移动到和输入同一个设备
        mu = action_bound * torch.tanh(self.fc_mu(x))  # tanh 限制到 [-1, 1]，再乘动作范围得到合法均值
        std = F.softplus(self.fc_std(x)) + 1e-5  # softplus 保证标准差为正，1e-5 防止太接近 0
        return mu, std  # 返回高斯分布的两个参数


class ValueNet(torch.nn.Module):
    """状态价值网络。

    用法：
        net = ValueNet(state_dim=3, hidden_dim=128)
        values = net(states)

    输入：
        states: 形状一般是 [batch_size, state_dim]。

    输出：
        values: 每个状态的价值 V(s)，形状是 [batch_size, 1]。

    critic 的作用：
        帮 actor 判断某一步动作是“比预期好”还是“比预期差”。
    """

    def __init__(self, state_dim, hidden_dim):
        super().__init__()  # 初始化父类
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 状态 -> 隐藏层
        self.fc2 = torch.nn.Linear(hidden_dim, 1)  # 隐藏层 -> 一个价值数字

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层激活
        return self.fc2(x)  # 输出 V(s)


def compute_advantage(gamma, lmbda, td_delta):
    """用 GAE 计算优势函数 advantage。

    用法：
        advantage = compute_advantage(gamma=0.9, lmbda=0.9, td_delta=td_delta)

    参数：
        gamma: 折扣因子。
        lmbda: GAE 平滑参数。
        td_delta: 每一步 TD 误差。

    返回：
        advantage: 形状是 [T, 1]。

    直觉：
        advantage 越大，说明这一步动作越值得增加概率。
        advantage 越小，说明这一步动作越应该降低概率。
    """

    td_delta = td_delta.detach().cpu().numpy().flatten()  # 从计算图取出 TD 误差，转成一维 numpy 数组
    advantage_list = []  # 保存每个时间步的优势
    advantage = 0.0  # 从轨迹最后一步开始累计
    for delta in td_delta[::-1]:  # 从后往前遍历 TD 误差
        advantage = gamma * lmbda * advantage + delta  # GAE 递推公式
        advantage_list.append(advantage)  # 保存当前优势
    advantage_list.reverse()  # 恢复成从前往后的顺序
    return torch.tensor(advantage_list, dtype=torch.float32).view(-1, 1)  # 转成 [T, 1] 张量


class TRPOContinuous:
    """连续动作 TRPO 智能体。

    用法：
        agent = TRPOContinuous(...)
        action = agent.take_action(state)
        agent.update(transition_dict)

    主要方法：
        take_action:
            根据状态，从 actor 输出的高斯分布中采样连续动作。

        update:
            用一整局轨迹更新 critic 和 actor。

        policy_learn:
            执行 TRPO 策略更新。

        conjugate_gradient:
            近似求自然梯度方向。

        line_search:
            缩小步长，直到满足目标提升和 KL 约束。
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
        state_dim = state_space.shape[0]  # Pendulum 状态维度通常是 3
        action_dim = action_space.shape[0]  # Pendulum 动作维度通常是 1
        self.action_low = np.asarray(action_space.low, dtype=np.float32)  # 动作下界，例如 [-2]
        self.action_high = np.asarray(action_space.high, dtype=np.float32)  # 动作上界，例如 [2]
        action_bound = np.maximum(np.abs(self.action_low), np.abs(self.action_high))  # 动作最大绝对值，用来缩放 tanh 输出
        self.actor = PolicyNetContinuous(
            state_dim, hidden_dim, action_dim, action_bound
        ).to(device)  # 创建连续动作策略网络
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 创建价值网络
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)  # critic 使用 Adam 优化器
        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # GAE lambda
        self.kl_constraint = kl_constraint  # KL 信任区域大小
        self.alpha = alpha  # 线性搜索缩放系数
        self.device = device  # 训练设备

    def take_action(self, state):
        """根据状态采样连续动作。

        用法：
            action = agent.take_action(state)

        返回：
            一个 numpy 数组，例如 array([0.37], dtype=float32)。

        为什么要 clip？
            高斯分布可能采样出超过环境动作范围的值。
            环境要求动作在 [low, high] 之间，所以要裁剪。
        """

        state = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device)  # 把状态转成张量
        mu, std = self.actor(state.unsqueeze(0))  # 增加 batch 维度，得到高斯分布参数
        action_dist = torch.distributions.Normal(mu, std)  # 创建正态分布 Normal(mu, std)
        action = action_dist.sample().detach().cpu().numpy()[0]  # 采样动作，并转到 numpy
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)  # 裁剪到合法范围后返回

    def hessian_matrix_vector_product(
        self, states, old_action_dists, vector, damping=0.1
    ):
        """计算 H*v，H 是平均 KL 散度对 actor 参数的 Hessian。

        用法：
            hvp = self.hessian_matrix_vector_product(states, old_action_dists, vector)

        连续动作里的 KL：
            old_action_dists 和 new_action_dists 都是高斯分布。
            PyTorch 可以直接计算两个 Normal 分布的 KL。

        为什么要 sum(dim=1)？
            如果动作有多个维度，每个维度都有一个 KL。
            一个动作向量的总 KL 是所有动作维度 KL 的和。
        """

        mu, std = self.actor(states)  # 当前 actor 输出新高斯分布参数
        new_action_dists = torch.distributions.Normal(mu, std)  # 构造新高斯策略分布
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists).sum(
                dim=1, keepdim=True
            )
        )  # 计算平均 KL；先对动作维度求和，再对时间步求平均
        actor_params = tuple(self.actor.parameters())  # 显式转成 tuple，避免类型检查器误解
        kl_grad = torch.autograd.grad(
            kl, actor_params, create_graph=True, allow_unused=False
        )  # 对 KL 求一阶梯度，并保留计算图以便继续求二阶
        kl_grad_vector = torch.cat([grad.contiguous().view(-1) for grad in kl_grad])  # 把所有一阶梯度拼成长向量
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)  # 和给定向量做点积
        grad2 = torch.autograd.grad(kl_grad_vector_product, actor_params)  # 再求一次梯度，得到 H*v
        grad2_vector = torch.cat([grad.contiguous().view(-1) for grad in grad2])  # 把 H*v 拼成长向量
        return grad2_vector + damping * vector  # 加阻尼项，提高数值稳定性

    def conjugate_gradient(self, grad, states, old_action_dists, max_iterations=10):
        """用共轭梯度求解 H*x = grad。

        用法：
            direction = self.conjugate_gradient(obj_grad, states, old_action_dists)

        返回：
            direction 近似等于 H^{-1}grad。

        在 TRPO 中：
            grad 表示“如何提高收益”。
            H 表示“策略变化有多大”。
            H^{-1}grad 就是在 KL 几何下更合适的更新方向。
        """

        x = torch.zeros_like(grad)  # 初始解设为 0
        r = grad.clone()  # 初始残差，因为 x=0，所以 r=grad
        p = grad.clone()  # 初始搜索方向
        rdotr = torch.dot(r, r)  # 残差平方
        for _ in range(max_iterations):  # 迭代求近似解
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)  # 计算 H*p
            alpha = rdotr / torch.dot(p, Hp).clamp_min(1e-8)  # 当前搜索方向上的步长
            x += alpha * p  # 更新解
            r -= alpha * Hp  # 更新残差
            new_rdotr = torch.dot(r, r)  # 计算新残差平方
            if new_rdotr < 1e-10:  # 如果已经足够接近解，就停止
                break
            beta = new_rdotr / rdotr  # 计算新旧搜索方向的混合比例
            p = r + beta * p  # 更新搜索方向
            rdotr = new_rdotr  # 保存新残差平方
        return x  # 返回近似的 H^{-1}grad

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):
        """计算连续动作 TRPO 的替代目标函数。

        用法：
            obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)

        公式直觉：
            ratio = 新策略下动作概率 / 旧策略下动作概率
            objective = mean(ratio * advantage)

        为什么 log_prob 要 sum(dim=1)？
            多维连续动作中，每个维度都有一个 log_prob。
            一个完整动作向量的 log_prob 是各维 log_prob 之和。
        """

        mu, std = actor(states)  # 用传入的 actor 计算高斯分布参数
        action_dists = torch.distributions.Normal(mu, std)  # 构造高斯动作分布
        log_probs = action_dists.log_prob(actions).sum(dim=1, keepdim=True)  # 计算实际动作在新策略下的 log 概率
        ratio = torch.exp(log_probs - old_log_probs)  # 新旧概率比
        return torch.mean(ratio * advantage)  # 策略替代目标

    def line_search(
        self, states, actions, advantage, old_log_probs, old_action_dists, max_vec
    ):
        """线性搜索，找到满足 TRPO 约束的新参数。

        用法：
            new_para = self.line_search(...)

        每次尝试：
            new_para = old_para + coef * max_vec

        如果新参数让目标变大，并且 KL 没超限制，就接受。
        如果不行，就把 coef 变小继续试。
        """

        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters()
        )  # 把 actor 参数拉平成一个长向量
        old_obj = self.compute_surrogate_obj(
            states, actions, advantage, old_log_probs, self.actor
        )  # 旧 actor 的策略目标
        for i in range(15):  # 最多尝试 15 次步长缩小
            coef = self.alpha**i  # 例如 alpha=0.5，则 coef 是 1、0.5、0.25 ...
            new_para = old_para + coef * max_vec  # 构造候选新参数
            new_actor = copy.deepcopy(self.actor)  # 拷贝临时 actor，不直接改原 actor
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters()
            )  # 把候选参数写入临时 actor
            mu, std = new_actor(states)  # 临时 actor 输出新高斯参数
            new_action_dists = torch.distributions.Normal(mu, std)  # 构造新高斯策略
            kl_div = torch.mean(
                torch.distributions.kl.kl_divergence(
                    old_action_dists, new_action_dists
                ).sum(dim=1, keepdim=True)
            )  # 计算旧策略和候选新策略的 KL
            new_obj = self.compute_surrogate_obj(
                states, actions, advantage, old_log_probs, new_actor
            )  # 计算候选新策略的目标
            if new_obj.item() > old_obj.item() and kl_div.item() < self.kl_constraint:  # 同时满足收益提升和 KL 约束
                return new_para  # 接受候选参数
        return old_para  # 如果所有候选都失败，就保持原参数

    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage):
        """TRPO 的 actor 更新主函数。

        用法：
            self.policy_learn(states, actions, old_action_dists, old_log_probs, advantage)

        步骤：
            1. 算替代目标。
            2. 对 actor 参数求梯度。
            3. 用共轭梯度求自然梯度方向。
            4. 计算满足 KL 约束的最大步长。
            5. 线性搜索确认真正可用的新参数。
        """

        surrogate_obj = self.compute_surrogate_obj(
            states, actions, advantage, old_log_probs, self.actor
        )  # 当前策略目标
        actor_params = tuple(self.actor.parameters())  # actor 参数 tuple
        grads = torch.autograd.grad(surrogate_obj, actor_params)  # 目标函数对 actor 参数求梯度
        obj_grad = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()  # 拼成长向量并断开图

        direction = self.conjugate_gradient(obj_grad, states, old_action_dists)  # 求自然梯度方向
        Hd = self.hessian_matrix_vector_product(states, old_action_dists, direction)  # 计算 H*d
        max_coef = torch.sqrt(
            2 * self.kl_constraint / torch.dot(direction, Hd).clamp_min(1e-8)
        )  # 根据 KL 二阶近似得到最大步长系数
        new_para = self.line_search(
            states,
            actions,
            advantage,
            old_log_probs,
            old_action_dists,
            direction * max_coef,
        )  # 用线性搜索选择最终参数
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor.parameters()
        )  # 把最终参数写回真实 actor

    def update(self, transition_dict):
        """用一整局轨迹更新 critic 和 actor。

        用法：
            agent.update(transition_dict)

        transition_dict 需要包含：
            states
            actions
            next_states
            rewards
            dones

        连续动作这里特别注意：
            actions 必须是 float32，形状一般是 [T, action_dim]。
        """

        states = torch.as_tensor(
            np.asarray(transition_dict["states"], dtype=np.float32),
            device=self.device,
        )  # 状态张量，形状 [T, state_dim]
        actions = torch.as_tensor(
            np.asarray(transition_dict["actions"], dtype=np.float32),
            device=self.device,
        )  # 连续动作张量，形状 [T, action_dim]
        rewards = torch.as_tensor(
            transition_dict["rewards"], dtype=torch.float32, device=self.device
        ).view(-1, 1)  # 奖励张量，形状 [T, 1]
        next_states = torch.as_tensor(
            np.asarray(transition_dict["next_states"], dtype=np.float32),
            device=self.device,
        )  # 下一状态张量
        dones = torch.as_tensor(
            transition_dict["dones"], dtype=torch.float32, device=self.device
        ).view(-1, 1)  # 结束标记，True/False 转成 1.0/0.0

        rewards = (rewards + 8.0) / 8.0  # 简单奖励缩放，让 Pendulum 的训练数值更舒服
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)  # TD 目标
        td_delta = td_target - self.critic(states)  # TD 误差
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta).to(self.device)  # GAE 优势
        advantage = (advantage - advantage.mean()) / (
            advantage.std(unbiased=False) + 1e-8
        )  # 标准化 advantage，减少训练震荡

        with torch.no_grad():  # 旧策略信息不需要梯度
            mu, std = self.actor(states)  # 用更新前 actor 计算旧高斯分布参数
            old_log_probs = (
                torch.distributions.Normal(mu, std)
                .log_prob(actions)
                .sum(dim=1, keepdim=True)
            )  # 记录旧策略下实际动作的 log 概率
        old_action_dists = torch.distributions.Normal(mu.detach(), std.detach())  # 构造旧策略分布，用于 KL

        critic_loss = F.mse_loss(self.critic(states), td_target.detach())  # critic 拟合 TD 目标
        self.critic_optimizer.zero_grad()  # 清空 critic 梯度
        critic_loss.backward()  # 反向传播 critic loss
        self.critic_optimizer.step()  # 更新 critic 参数

        self.policy_learn(states, actions, old_action_dists, old_log_probs, advantage)  # 更新 actor


def reset_env(env, seed=None):
    """兼容 gymnasium 和老版本 gym 的 reset。

    用法：
        state = reset_env(env, seed=0)
        state = reset_env(env)

    返回：
        只返回 state，不返回 info。
    """

    result = env.reset(seed=seed) if seed is not None else env.reset()  # 根据是否给 seed 调用 reset
    return result[0] if isinstance(result, tuple) else result  # 新 API 返回 (state, info)，这里取 state


def step_env(env, action):
    """兼容不同环境 API 的 step。

    用法：
        next_state, reward, done, info = step_env(env, action)

    返回：
        next_state: 下一状态
        reward: 当前步奖励
        done: 这一局是否结束
        info: 环境额外信息
    """

    result = env.step(action)  # 环境执行动作
    if len(result) == 5:  # gymnasium 新 API
        next_state, reward, terminated, truncated, info = result  # 分别取出 5 个返回值
        return next_state, reward, terminated or truncated, info  # 自然结束或时间截断都算 done
    next_state, reward, done, info = result  # 老 API
    return next_state, reward, done, info  # 返回统一格式


def train_on_policy_agent(env, agent, num_episodes):
    """训练 on-policy 智能体。

    用法：
        returns = train_on_policy_agent(env, agent, 2000)

    每一局做的事情：
        1. reset 环境。
        2. actor 采样动作。
        3. step 环境。
        4. 保存转移数据。
        5. 一局结束后 agent.update。

    为什么不使用经验回放？
        TRPO 是 on-policy 方法，需要使用当前策略刚采样的数据。
    """

    return_list = []  # 保存每一局总奖励
    episodes_per_iteration = max(1, num_episodes // 10)  # 把训练分成约 10 段显示
    num_iterations = int(np.ceil(num_episodes / episodes_per_iteration))  # 实际段数

    for iteration in range(num_iterations):  # 外层进度段循环
        start_episode = iteration * episodes_per_iteration  # 当前段起始 episode
        end_episode = min(num_episodes, start_episode + episodes_per_iteration)  # 当前段结束 episode
        with tqdm(total=end_episode - start_episode, desc=f"Iteration {iteration}") as pbar:  # 创建进度条
            for local_episode in range(end_episode - start_episode):  # 当前段内逐局训练
                episode_return = 0.0  # 当前局累计奖励
                transition_dict = {
                    "states": [],
                    "actions": [],
                    "next_states": [],
                    "rewards": [],
                    "dones": [],
                }  # 一局轨迹数据

                state = reset_env(env)  # 重置环境，得到初始状态
                done = False  # 这一局开始时还没结束
                while not done:  # 持续交互直到 episode 结束
                    action = agent.take_action(state)  # actor 从高斯策略中采样连续动作
                    next_state, reward, done, _ = step_env(env, action)  # 环境执行动作
                    transition_dict["states"].append(state)  # 保存状态
                    transition_dict["actions"].append(action)  # 保存连续动作
                    transition_dict["next_states"].append(next_state)  # 保存下一状态
                    transition_dict["rewards"].append(reward)  # 保存奖励
                    transition_dict["dones"].append(done)  # 保存是否结束
                    state = next_state  # 进入下一状态
                    episode_return += reward  # 累加奖励

                return_list.append(episode_return)  # 记录这一局总回报
                agent.update(transition_dict)  # 用这一局数据更新智能体
                if (local_episode + 1) % 10 == 0 or local_episode + 1 == end_episode - start_episode:  # 每 10 局或段末更新显示
                    pbar.set_postfix(
                        {
                            "episode": start_episode + local_episode + 1,
                            "return": f"{np.mean(return_list[-10:]):.3f}",
                        }
                    )  # 显示最近 10 局平均回报
                pbar.update(1)  # 进度条前进一步
    return return_list  # 返回所有回报，用于画图


def moving_average(values, window_size):
    """计算移动平均曲线。

    用法：
        smooth = moving_average(return_list, 9)

    作用：
        原始强化学习曲线通常抖动很大，移动平均能看出整体趋势。
    """

    values = np.asarray(values, dtype=np.float32)  # 转成 numpy 数组
    if len(values) < window_size:  # 数据长度不足一个窗口
        return values  # 直接返回原始值
    cumulative_sum = np.cumsum(np.insert(values, 0, 0.0))  # 前缀和
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size  # 中间标准移动平均
    radius = np.arange(1, window_size - 1, 2)  # 边缘窗口大小
    begin = np.cumsum(values[: window_size - 1])[::2] / radius  # 开头部分平滑
    end = (np.cumsum(values[: -window_size : -1])[::2] / radius)[::-1]  # 结尾部分平滑
    return np.concatenate((begin, middle, end))  # 拼接完整平滑曲线


def plot_returns(return_list, env_name, output_dir):
    """保存训练曲线图片。

    用法：
        raw_path, smooth_path = plot_returns(return_list, "Pendulum-v1", Path("results"))

    输出：
        trpo_pendulum_returns.png
        trpo_pendulum_returns_smoothed.png
    """

    output_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录
    episodes = list(range(len(return_list)))  # x 轴 episode 编号

    raw_path = output_dir / "trpo_pendulum_returns.png"  # 原始曲线路径
    smooth_path = output_dir / "trpo_pendulum_returns_smoothed.png"  # 平滑曲线路径

    plt.figure()  # 新建图像
    plt.plot(episodes, return_list)  # 画原始回报
    plt.xlabel("Episodes")  # x 轴
    plt.ylabel("Returns")  # y 轴
    plt.title(f"TRPO on {env_name}")  # 标题
    plt.tight_layout()  # 自动排版
    plt.savefig(raw_path, dpi=150)  # 保存图像
    plt.close()  # 关闭图像，释放内存

    plt.figure()  # 新建第二张图
    plt.plot(episodes, moving_average(return_list, 9))  # 画平滑曲线
    plt.xlabel("Episodes")  # x 轴
    plt.ylabel("Returns")  # y 轴
    plt.title(f"TRPO on {env_name}")  # 标题
    plt.tight_layout()  # 自动排版
    plt.savefig(smooth_path, dpi=150)  # 保存平滑曲线
    plt.close()  # 关闭图像

    return raw_path, smooth_path  # 返回图片路径


def parse_args():
    """读取命令行参数。

    用法：
        args = parse_args()

    示例：
        python "TRPO\\demo\\TRPO 倒立杆.py" --num-episodes 100 --gamma 0.9

    不传参数时，会使用默认的 2000 局训练。
    """

    parser = argparse.ArgumentParser(description="Train TRPO on Pendulum.")  # 创建参数解析器
    parser.add_argument("--env-name", type=str, default="Pendulum-v1")  # 环境名称
    parser.add_argument("--num-episodes", type=int, default=2000)  # 训练局数
    parser.add_argument("--hidden-dim", type=int, default=128)  # 隐藏层维度
    parser.add_argument("--gamma", type=float, default=0.9)  # 折扣因子
    parser.add_argument("--lmbda", type=float, default=0.9)  # GAE lambda
    parser.add_argument("--critic-lr", type=float, default=1e-2)  # critic 学习率
    parser.add_argument("--kl-constraint", type=float, default=5e-5)  # KL 约束，连续控制这里设得更小
    parser.add_argument("--alpha", type=float, default=0.5)  # 线性搜索缩放系数
    parser.add_argument("--seed", type=int, default=0)  # 随机种子
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )  # 图片输出目录
    return parser.parse_args()  # 返回解析后的参数对象


def main():
    """程序入口。

    主流程：
        1. 读取参数。
        2. 设置随机种子。
        3. 创建 Pendulum 环境。
        4. 创建连续动作 TRPO 智能体。
        5. 训练并保存曲线。
    """

    args = parse_args()  # 读取命令行参数
    if args.num_episodes <= 0:  # 检查训练局数是否合法
        raise ValueError("--num-episodes must be positive.")  # 非法就报错

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先使用 GPU
    np.random.seed(args.seed)  # 设置 numpy 随机种子
    torch.manual_seed(args.seed)  # 设置 PyTorch CPU 随机种子
    if torch.cuda.is_available():  # 如果有 CUDA
        torch.cuda.manual_seed_all(args.seed)  # 设置 PyTorch GPU 随机种子

    env = gym.make(args.env_name)  # 创建 Pendulum-v1 环境
    reset_env(env, seed=args.seed)  # 用指定 seed 重置环境
    if hasattr(env.action_space, "seed"):  # 如果动作空间支持设置 seed
        env.action_space.seed(args.seed)  # 设置动作采样随机种子

    agent = TRPOContinuous(
        args.hidden_dim,
        env.observation_space,
        env.action_space,
        args.lmbda,
        args.kl_constraint,
        args.alpha,
        args.critic_lr,
        args.gamma,
        device,
    )  # 创建连续动作 TRPO 智能体
    return_list = train_on_policy_agent(env, agent, args.num_episodes)  # 训练智能体
    env.close()  # 关闭环境

    raw_path, smooth_path = plot_returns(return_list, args.env_name, args.output_dir)  # 保存曲线图
    print(f"Training finished on {device}.")  # 打印训练设备
    print(f"Last 10 episode mean return: {np.mean(return_list[-10:]):.3f}")  # 打印最近 10 局平均回报
    print(f"Saved plots: {raw_path}, {smooth_path}")  # 打印保存路径


if __name__ == "__main__":  # 只有直接运行本文件时才执行 main
    main()  # 启动训练流程
