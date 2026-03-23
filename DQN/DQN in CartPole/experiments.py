"""
experiments.py —— DQN Ablation Study & Sensitivity Analysis
环境: Pendulum-v1（离散化动作空间）

运行方式:
    python DQN/experiments.py

输出:
    DQN/ablation_curves.png   —— 各变体学习曲线对比
    DQN/ablation_bar.png      —— 各变体最终性能柱状图
    DQN/sensitivity_<param>.png —— 每个超参数的敏感性曲线
    DQN/sensitivity_summary.png —— 超参数敏感性总览

Ablation Study 验证的核心设计：
    1. Target Network   ——  去掉后用在线网络直接算TD目标，暴露训练不稳定性
    2. Replay Buffer    ——  去掉后每步立即更新，暴露样本相关性问题
    3. Double DQN       ——  解耦动作选择与评估，缓解过估计
    4. Dueling DQN      ——  V/A流分离，在稀疏奖励下更高效

Sensitivity Analysis 分析的关键超参数：
    lr, gamma, epsilon, target_update, batch_size
"""

import random
import os
import json
import csv
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')           # 非交互式后端，用于保存图片（不弹出窗口）
import matplotlib.pyplot as plt
import collections


# ══════════════════════════════════════════════════════════════
# 1. 网络结构
# ══════════════════════════════════════════════════════════════

class Qnet(torch.nn.Module):
    """标准两层全连接 Q 网络（Vanilla DQN / Double DQN）"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class VAnet(torch.nn.Module):
    """
    Dueling Network：共享特征层 + 独立的 V 流和 A 流
    Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]
    减去 A 的均值保证可识别性（V 和 A 的分解唯一）
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1  = torch.nn.Linear(state_dim, hidden_dim)   # 共享特征层
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)  # 优势流 A(s,a)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)           # 价值流 V(s)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        A = self.fc_A(h)
        V = self.fc_V(h)
        return V + A - A.mean(dim=1, keepdim=True)


# ══════════════════════════════════════════════════════════════
# 2. 经验回放池
# ══════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*transitions)
        return np.array(s), a, r, np.array(ns), d

    def size(self):
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════
# 3. 统一 DQN Agent（支持所有 ablation 组合）
# ══════════════════════════════════════════════════════════════

class DQN:
    """
    统一 DQN Agent，通过参数控制各组件的开/关：

    dqn_type:
        'VanillaDQN'       —— 标准 DQN
        'DoubleDQN'        —— Double DQN（在线网络选动作，目标网络评估）
        'DuelingDQN'       —— Dueling DQN（VAnet 双流结构）
        'DoubleDuelingDQN' —— Double + Dueling 的组合

    use_target:
        True  —— 使用独立目标网络（standard）
        False —— 目标值也用在线网络计算（ablation：去掉目标网络）
                 预期现象：训练不稳定，曲线震荡，收敛更慢
    """
    def __init__(self, state_dim, hidden_dim, action_dim,
                 learning_rate, gamma, epsilon, target_update,
                 device, dqn_type='VanillaDQN', use_target=True):
        self.action_dim    = action_dim
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.target_update = target_update
        self.count         = 0
        self.dqn_type      = dqn_type
        self.use_target    = use_target
        self.device        = device

        # 根据算法类型选择网络结构
        NetCls = VAnet if 'Dueling' in dqn_type else Qnet
        self.q_net        = NetCls(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = NetCls(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())  # 初始参数同步

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def take_action(self, state):
        """ε-贪婪策略选动作"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        s = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        return self.q_net(s).argmax().item()

    def max_q_value(self, state):
        """返回当前状态下在线网络的最大 Q 值（用于监控过估计）"""
        s = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        return self.q_net(s).max().item()

    def update(self, transition_dict):
        s  = torch.tensor(transition_dict['states'],      dtype=torch.float).to(self.device)
        a  = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        r  = torch.tensor(transition_dict['rewards'],     dtype=torch.float).view(-1, 1).to(self.device)
        ns = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        d  = torch.tensor(transition_dict['dones'],       dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(s).gather(1, a)

        # Ablation: use_target=False 时用在线网络代替目标网络计算 TD 目标
        target_net = self.target_q_net if self.use_target else self.q_net

        if 'Double' in self.dqn_type:
            # Double DQN：在线网络选动作，目标网络（或在线网络）评估
            # 解耦"选择"与"评估"，降低过估计偏差
            best_a     = self.q_net(ns).max(1)[1].view(-1, 1)
            max_next_q = target_net(ns).gather(1, best_a)
        else:
            # Vanilla DQN：目标网络直接取最大 Q 值（选择与评估都用目标网络）
            max_next_q = target_net(ns).max(1)[0].view(-1, 1)

        q_targets = r + self.gamma * max_next_q * (1 - d)  # Bellman 目标
        loss = F.mse_loss(q_values, q_targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 硬更新目标网络
        if self.use_target and self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


# ══════════════════════════════════════════════════════════════
# 4. Pendulum 离散动作辅助函数
# ══════════════════════════════════════════════════════════════

ACTION_DIM = 11  # 将 Pendulum 连续力矩空间等分为 11 个离散动作


def dis_to_con(discrete_action, env):
    """将离散动作编号 [0, ACTION_DIM-1] 线性映射回连续力矩值"""
    low  = env.action_space.low[0]   # Pendulum: -2.0
    high = env.action_space.high[0]  # Pendulum: +2.0
    return low + (discrete_action / (ACTION_DIM - 1)) * (high - low)


# ══════════════════════════════════════════════════════════════
# 5. 训练主循环
# ══════════════════════════════════════════════════════════════

def train_DQN(agent, env, num_episodes,
              buffer_size=5000, minimal_size=1000, batch_size=64,
              use_replay=True, seed=0, verbose=False):
    """
    训练一个 DQN agent，返回 (return_list, max_q_list)。

    use_replay=False：Ablation "No Replay Buffer"
        不使用经验回放，每一步立即用当前单条经验更新网络。
        预期现象：样本相关性高，梯度估计方差大，收敛不稳定。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    replay_buffer = ReplayBuffer(buffer_size)
    return_list   = []
    max_q_list    = []
    max_q_ema     = 0.0  # 对最大Q值做指数移动平均，平滑监控曲线

    iter_bar = range(num_episodes)
    if verbose:
        iter_bar = tqdm(iter_bar, desc=f'seed={seed}')

    for i_episode in iter_bar:
        episode_return = 0.0
        state, _ = env.reset(seed=seed + i_episode)
        done = False

        while not done:
            action = agent.take_action(state)
            # 指数移动平均平滑 Q 值曲线（系数 0.005 使曲线更平滑）
            max_q_ema = agent.max_q_value(state) * 0.005 + max_q_ema * 0.995
            max_q_list.append(max_q_ema)

            action_cont = dis_to_con(action, env)
            next_state, reward, terminated, truncated, _ = env.step([action_cont])
            done = terminated or truncated

            if use_replay:
                # 标准经验回放：存入缓冲区，达到 minimal_size 后随机采样训练
                replay_buffer.add(state, action, reward, next_state, done)
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    agent.update({'states': b_s, 'actions': b_a, 'rewards': b_r,
                                  'next_states': b_ns, 'dones': b_d})
            else:
                # Ablation: 无经验回放 —— 用当前单步经验立即更新（在线学习）
                agent.update({
                    'states':      np.array([state]),
                    'actions':     [action],
                    'rewards':     [reward],
                    'next_states': np.array([next_state]),
                    'dones':       [float(done)],
                })

            state = next_state
            episode_return += reward

        return_list.append(episode_return)

    return return_list, max_q_list


# ══════════════════════════════════════════════════════════════
# 6. 实验运行器（多种子）
# ══════════════════════════════════════════════════════════════

def run_experiment(config: dict, seeds=(0, 1, 2)) -> dict:
    """
    用同一配置跑 len(seeds) 次，返回均值和标准差数组。

    返回: {'mean': ndarray(num_episodes,), 'std': ndarray(num_episodes,)}
    """
    num_episodes  = config.get('num_episodes', 200)
    hidden_dim    = config.get('hidden_dim', 128)
    lr            = config.get('lr', 1e-2)
    gamma         = config.get('gamma', 0.98)
    epsilon       = config.get('epsilon', 0.01)
    target_update = config.get('target_update', 50)
    buffer_size   = config.get('buffer_size', 5000)
    minimal_size  = config.get('minimal_size', 1000)
    batch_size    = config.get('batch_size', 64)
    dqn_type      = config.get('dqn_type', 'VanillaDQN')
    use_target    = config.get('use_target', True)
    use_replay    = config.get('use_replay', True)
    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_returns = []
    for seed in seeds:
        env       = gym.make('Pendulum-v1')
        state_dim = env.observation_space.shape[0]  # type: ignore[index]  # 3: cos θ, sin θ, 角速度
        agent = DQN(state_dim, hidden_dim, ACTION_DIM,
                    lr, gamma, epsilon, target_update,
                    device, dqn_type=dqn_type, use_target=use_target)
        returns, _ = train_DQN(agent, env, num_episodes,
                                buffer_size, minimal_size, batch_size,
                                use_replay, seed)
        env.close()
        all_returns.append(returns)

    arr = np.array(all_returns)  # shape: (num_seeds, num_episodes)
    return {'mean': arr.mean(axis=0), 'std': arr.std(axis=0)}


# ══════════════════════════════════════════════════════════════
# 7. 绘图工具
# ══════════════════════════════════════════════════════════════

def smooth(a, window=9):
    """简单滑动平均平滑曲线"""
    return np.convolve(a, np.ones(window) / window, mode='same')


def plot_curves(results: dict, title: str, filepath: str, ylabel='Returns'):
    """
    绘制多条学习曲线（均值曲线 + 标准差阴影），自动保存为 PNG。
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, data in results.items():
        mean = smooth(data['mean'])
        std  = smooth(data['std'])
        eps  = np.arange(len(mean))
        line, = ax.plot(eps, mean, label=label)
        ax.fill_between(eps, mean - std, mean + std, alpha=0.18, color=line.get_color())
    ax.set_xlabel('Episodes')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f'  已保存 → {filepath}')


def plot_bar(names, means, stds, title: str, filepath: str):
    """绘制最终性能柱状图（带误差棒），保存为 PNG。"""
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.4), 4))
    colors = matplotlib.colormaps['Set2'](np.linspace(0, 1, len(names)))
    ax.bar(range(len(names)), means, yerr=stds, capsize=5, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Final Return (last 20 eps, mean±std)')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f'  已保存 → {filepath}')


def save_results(results: dict, json_path: str, csv_path: str):
    """
    将实验结果持久化到磁盘：
      json_path：完整学习曲线（mean/std 数组），可用于后续重新绘图
      csv_path ：汇总表（每个变体的最终回报均值和标准差），方便查阅
    """
    # ── JSON：保存完整曲线数据（numpy array → list 才能序列化）
    serializable = {
        name: {'mean': data['mean'].tolist(), 'std': data['std'].tolist()}
        for name, data in results.items()
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f'  已保存 → {json_path}')

    # ── CSV：汇总最终性能（后20回合的均值±标准差）
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['variant', 'final_mean', 'final_std'])
        for name, data in results.items():
            final_mean = float(data['mean'][-20:].mean())
            final_std  = float(data['std'][-20:].mean())
            writer.writerow([name, f'{final_mean:.2f}', f'{final_std:.2f}'])
    print(f'  已保存 → {csv_path}')


def print_summary_table(results: dict, title: str):
    """在终端打印对齐的汇总表格（最终回报排名）。"""
    rows = []
    for name, data in results.items():
        rows.append((data['mean'][-20:].mean(), data['std'][-20:].mean(), name))
    rows.sort(reverse=True)  # 按最终回报从高到低排序

    col = max(len(r[2]) for r in rows) + 2
    print(f'\n{"─" * (col + 30)}')
    print(f'  {title}')
    print(f'{"─" * (col + 30)}')
    print(f'  {"Variant":<{col}} {"Final Mean":>12}  {"Std":>8}')
    print(f'  {"─"*col} {"─"*12}  {"─"*8}')
    for mean, std, name in rows:
        print(f'  {name:<{col}} {mean:>12.1f}  {std:>8.1f}')
    print(f'{"─" * (col + 30)}\n')


# ══════════════════════════════════════════════════════════════
# 8. Q 值过估计分析（Vanilla vs Double）
# ══════════════════════════════════════════════════════════════

def _plot_q_overestimation(base_cfg: dict, seeds=(0, 1, 2)):
    """
    对比 VanillaDQN 与 DoubleDQN 在训练过程中的最大 Q 值曲线。

    Vanilla DQN 的 TD 目标：Q_target = max_a Q_target(s', a)
      → max 操作天然高估（选的是采样最大值，存在正偏差）
      → 目标网络与在线网络结构相同，过估计会被 Bellman 迭代逐步放大

    Double DQN 的修正：
      → 用在线网络"选"动作（argmax），用目标网络"评估"该动作的 Q 值
      → 选择与评估解耦，两个网络的误差不再同向叠加，有效压制过估计

    预期现象：VanillaDQN 的 Q 值曲线持续高于 DoubleDQN，
    差距越大说明过估计越严重。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    variants = {'VanillaDQN': 'VanillaDQN', 'DoubleDQN': 'DoubleDQN'}
    q_results = {}

    for label, dqn_type in variants.items():
        all_q = []
        for seed in seeds:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            env = gym.make('Pendulum-v1')
            state_dim = env.observation_space.shape[0]  # type: ignore[index]
            agent = DQN(state_dim, base_cfg.get('hidden_dim', 128), ACTION_DIM,
                        base_cfg.get('lr', 1e-2), base_cfg.get('gamma', 0.98),
                        base_cfg.get('epsilon', 0.01), base_cfg.get('target_update', 50),
                        device, dqn_type=dqn_type, use_target=True)
            _, max_q_list = train_DQN(
                agent, env, base_cfg.get('num_episodes', 200),
                base_cfg.get('buffer_size', 5000), base_cfg.get('minimal_size', 1000),
                base_cfg.get('batch_size', 64), use_replay=True, seed=seed)
            env.close()
            # 降采样到与 episode 数相同长度，方便对齐
            n = base_cfg.get('num_episodes', 200)
            idx = np.linspace(0, len(max_q_list) - 1, n, dtype=int)
            all_q.append(np.array(max_q_list)[idx])
        q_results[label] = {'mean': np.array(all_q).mean(axis=0),
                            'std':  np.array(all_q).std(axis=0)}

    plot_curves(q_results,
                'Q-value Overestimation: VanillaDQN vs DoubleDQN (Pendulum-v1)',
                'DQN/q_overestimation.png',
                ylabel='Max Q Value (EMA)')
    print('  已保存 → DQN/q_overestimation.png')


# ══════════════════════════════════════════════════════════════
# 9. Ablation Study
# ══════════════════════════════════════════════════════════════

def ablation_study(seeds=(0, 1, 2), num_episodes=200):
    """
    消融实验：逐一移除 DQN 的关键设计，对比对性能的影响。

    消融维度及预期现象：
    ─────────────────────────────────────────────────────
    VanillaDQN (baseline)    完整系统，作为对照基线
    ─────────────────────────────────────────────────────
    No Target Network        TD 目标用在线网络计算 → "自举自"
                             目标随网络同步变化，产生追逐不稳定的目标
                             预期：训练曲线震荡严重，难以收敛

    No Replay Buffer         每步用单条经验立即更新
                             相邻样本高度相关（违反 i.i.d. 假设）
                             预期：梯度估计方差大，收敛慢且不稳

    Double DQN               在线网络选动作，目标网络评估 Q 值
                             解耦选择与评估，降低 max 操作引起的过估计
                             预期：收敛稳定性提升，Q 值更准确（不过高）

    Dueling DQN              VAnet 将 Q 分解为 V(s) + A(s,a)
                             V 流可在不需要区分动作时快速更新
                             预期：在 Pendulum 这类动作价值差异小的环境效果更好

    Double + Dueling         两种改进合并，叠加效益
    ─────────────────────────────────────────────────────
    """
    print('\n' + '=' * 60)
    print('ABLATION STUDY')
    print('=' * 60)

    base = dict(num_episodes=num_episodes, hidden_dim=128,
                lr=1e-2, gamma=0.98, epsilon=0.01,
                target_update=50, buffer_size=5000,
                minimal_size=1000, batch_size=64)

    configs = {
        'VanillaDQN (baseline)': {**base, 'dqn_type': 'VanillaDQN',       'use_target': True,  'use_replay': True},
        'No Target Network':     {**base, 'dqn_type': 'VanillaDQN',       'use_target': False, 'use_replay': True},
        'No Replay Buffer':      {**base, 'dqn_type': 'VanillaDQN',       'use_target': True,  'use_replay': False},
        'Double DQN':            {**base, 'dqn_type': 'DoubleDQN',        'use_target': True,  'use_replay': True},
        'Dueling DQN':           {**base, 'dqn_type': 'DuelingDQN',       'use_target': True,  'use_replay': True},
        'Double + Dueling':      {**base, 'dqn_type': 'DoubleDuelingDQN', 'use_target': True,  'use_replay': True},
    }

    results = {}
    for name, cfg in configs.items():
        print(f'  Running: {name} ...')
        results[name] = run_experiment(cfg, seeds)
        final = results[name]['mean'][-20:].mean()
        std   = results[name]['std'][-20:].mean()
        print(f'    最终回报（后20回合均值）: {final:.1f} ± {std:.1f}')

    # 学习曲线对比图
    plot_curves(results,
                'Ablation Study: DQN Key Components (Pendulum-v1)',
                'DQN/ablation_curves.png')

    # 最终性能柱状图
    names  = list(results.keys())
    finals = [results[n]['mean'][-20:].mean() for n in names]
    stds   = [results[n]['std'][-20:].mean()  for n in names]
    plot_bar(names, finals, stds,
             'Ablation Study: Final Performance',
             'DQN/ablation_bar.png')

    # Q 值过估计对比（VanillaDQN vs DoubleDQN）
    # Double DQN 的核心贡献是降低过估计，需要单独对比 Q 值曲线才能看出来
    print('\n  Running Q-value overestimation analysis (VanillaDQN vs DoubleDQN) ...')
    _plot_q_overestimation(
        base_cfg={**dict(num_episodes=num_episodes, hidden_dim=128,
                         lr=1e-2, gamma=0.98, epsilon=0.01,
                         target_update=50, buffer_size=5000,
                         minimal_size=1000, batch_size=64),
                  'use_target': True, 'use_replay': True},
        seeds=seeds,
    )

    # 保存数值结果
    save_results(results, 'DQN/ablation_results.json', 'DQN/ablation_summary.csv')
    print_summary_table(results, 'Ablation Study — Final Return Ranking')

    return results


# ══════════════════════════════════════════════════════════════
# 9. Sensitivity Analysis
# ══════════════════════════════════════════════════════════════

def sensitivity_analysis(seeds=(0, 1, 2), num_episodes=200):
    """
    超参数敏感性分析：每次只变动一个超参数，其余保持默认值。

    分析的超参数及关注点：
    ──────────────────────────────────────────────────────
    lr (学习率)
        太小 → 收敛极慢；太大 → 梯度爆炸/震荡
        关键问题：有没有一个"宽容"的范围？

    gamma (折扣因子)
        太小 → 智能体目光短浅，忽略长期奖励
        太大（→1）→ 训练不稳定，有效时间步数极长
        关键问题：Pendulum 需要多远的规划时域？

    epsilon (探索率)
        太小 → 过早收敛到次优策略（exploitation-only）
        太大 → 过度随机探索，浪费样本效率
        关键问题：探索和利用的最佳平衡点？

    target_update (目标网络更新频率)
        太频繁（值小）→ 目标追逐本身，效果接近无目标网络
        太稀疏（值大）→ 目标过于陈旧，TD 误差不准
        关键问题：目标稳定性 vs. 目标新鲜度的权衡

    batch_size (批大小)
        太小 → 梯度估计噪声大
        太大 → 计算开销大，且不一定收益明显
        关键问题：样本效率与梯度质量的权衡
    ──────────────────────────────────────────────────────
    """
    print('\n' + '=' * 60)
    print('SENSITIVITY ANALYSIS')
    print('=' * 60)

    base = dict(num_episodes=num_episodes, dqn_type='VanillaDQN',
                use_target=True, use_replay=True, hidden_dim=128,
                lr=1e-2, gamma=0.98, epsilon=0.01,
                target_update=50, buffer_size=5000,
                minimal_size=1000, batch_size=64)

    sweeps = {
        'Learning Rate (lr)': {
            'param':  'lr',
            'values': [1e-3, 1e-2, 1e-1],
            'labels': ['1e-3', '1e-2 (default)', '1e-1'],
        },
        'Discount Factor (γ)': {
            'param':  'gamma',
            'values': [0.90, 0.98, 0.99],
            'labels': ['0.90', '0.98 (default)', '0.99'],
        },
        'Exploration Rate (ε)': {
            'param':  'epsilon',
            'values': [0.001, 0.01, 0.10],
            'labels': ['0.001', '0.01 (default)', '0.10'],
        },
        'Target Update Freq': {
            'param':  'target_update',
            'values': [1, 50, 200],
            'labels': ['1', '50 (default)', '200'],
        },
        'Batch Size': {
            'param':  'batch_size',
            'values': [16, 64, 256],
            'labels': ['16', '64 (default)', '256'],
        },
    }

    summary = {}  # {sweep_name: {'labels':[], 'finals':[], 'errs':[]}}

    for sweep_name, sweep_cfg in sweeps.items():
        print(f'\n  Sweeping: {sweep_name}')
        param, values, labels = sweep_cfg['param'], sweep_cfg['values'], sweep_cfg['labels']
        results = {}

        for val, lbl in zip(values, labels):
            cfg = {**base, param: val}
            print(f'    {param}={val} ...', end='', flush=True)
            results[lbl] = run_experiment(cfg, seeds)
            final = results[lbl]['mean'][-20:].mean()
            print(f'  最终回报: {final:.1f}')

        # 每个超参数单独的学习曲线图
        plot_curves(results,
                    f'Sensitivity: {sweep_name} (Pendulum-v1)',
                    f'DQN/sensitivity_{param}.png')

        summary[sweep_name] = {
            'labels': labels,
            'finals': [results[l]['mean'][-20:].mean() for l in labels],
            'errs':   [results[l]['std'][-20:].mean()  for l in labels],
        }

    # 总览图：一行 N 个子图，每个超参数一个柱状图
    n = len(sweeps)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    palette = matplotlib.colormaps['Pastel1'](np.linspace(0, 1, 5))
    for ax, (sweep_name, data) in zip(axes, summary.items()):
        ax.bar(range(len(data['labels'])), data['finals'],
               yerr=data['errs'], capsize=4, color=palette)
        ax.set_xticks(range(len(data['labels'])))
        ax.set_xticklabels(data['labels'], rotation=30, ha='right', fontsize=7)
        ax.set_title(sweep_name, fontsize=9)
        ax.set_ylabel('Final Return')
    plt.suptitle('Sensitivity Analysis Summary (Pendulum-v1, VanillaDQN)', fontsize=11)
    plt.tight_layout()
    plt.savefig('DQN/sensitivity_summary.png', dpi=150)
    plt.close()
    print('\n  已保存 → DQN/sensitivity_summary.png')

    # 将 summary 转为与 save_results 兼容的格式后保存
    # summary 结构: {sweep_name: {labels, finals, errs}}
    # 展平为 {"{sweep}|{label}": {mean, std}} 方便统一处理
    flat = {}
    for sweep_name, data in summary.items():
        for lbl, mean_val, std_val in zip(data['labels'], data['finals'], data['errs']):
            key = f'{sweep_name} | {lbl}'
            # 用标量填充为长度1数组，复用 save_results 的接口
            flat[key] = {'mean': np.array([mean_val]), 'std': np.array([std_val])}
    save_results(flat, 'DQN/sensitivity_results.json', 'DQN/sensitivity_summary.csv')

    return summary


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # 速度与可靠性的权衡：
    #   快速验证：SEEDS=(0,1),  NUM_EPISODES=100  → ~25 分钟
    #   标准实验：SEEDS=(0,1,2), NUM_EPISODES=200  → ~2 小时
    #   严格实验：SEEDS=(0,1,2,3,4), NUM_EPISODES=300 → ~5 小时
    SEEDS        = (0, 1)
    NUM_EPISODES = 100

    os.makedirs('DQN', exist_ok=True)  # 确保输出目录存在

    ablation_results    = ablation_study(seeds=SEEDS, num_episodes=NUM_EPISODES)
    sensitivity_results = sensitivity_analysis(seeds=SEEDS, num_episodes=NUM_EPISODES)

    print('\n' + '=' * 60)
    print('全部实验完成。输出文件清单：')
    print('  图片：DQN/ablation_curves.png, ablation_bar.png')
    print('        DQN/q_overestimation.png')
    print('        DQN/sensitivity_<param>.png, sensitivity_summary.png')
    print('  数据：DQN/ablation_results.json, ablation_summary.csv')
    print('        DQN/sensitivity_results.json, sensitivity_summary.csv')
    print('=' * 60)
