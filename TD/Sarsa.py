import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # tqdm是显示循环进度条的库


class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0  # 记录当前智能体位置的横坐标
        self.y = self.nrow - 1  # 记录当前智能体位置的纵坐标

    def step(self, action):  # 外部调用这个函数来改变当前位置
        # 4种动作: change[0]:上, change[1]:下, change[2]:左, change[3]:右
        # 坐标系原点(0,0)定义在左上角，x向右，y向下
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        # 移动后用 min/max 确保不越界
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        reward = -1  # 默认每步惩罚 -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:  # 到达最底行的悬崖区域或目标
            if self.x == self.ncol - 1:
                # 到达右下角终点，回合结束
                done = True
            else:
                # 掉入悬崖：给予 -100 惩罚，将智能体传送回起点，回合继续
                # 注意：next_state 必须在传送之后计算，否则会返回悬崖位置
                reward = -100
                self.x = 0
                self.y = self.nrow - 1
        # 必须在悬崖传送之后再计算 next_state，保证返回的是传送后的起点位置
        next_state = self.y * self.ncol + self.x
        return next_state, reward, done

    def reset(self):  # 回归初始状态：起点在左下角 (x=0, y=nrow-1)
        self.x = 0
        self.y = self.nrow - 1
        # 返回线性化的状态索引：state = y * ncol + x
        return self.y * self.ncol + self.x

class Sarsa:
    """
    Sarsa（on-policy TD控制）算法。
    与 Q-learning 的区别：更新时使用实际执行的下一步动作 a1（on-policy），
    而非贪婪最优动作，因此策略更保守，在 Cliff Walking 中会走远离悬崖的安全路线。
    """
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        # Q 表：行对应状态（共 nrow*ncol 个），列对应动作（共 n_action 个），全部初始化为 0
        self.Q_table = np.zeros([nrow * ncol, n_action])
        self.n_action = n_action  # 动作个数
        self.alpha = alpha        # 学习率：控制每次更新的步长
        self.gamma = gamma        # 折扣因子：对未来奖励的衰减程度
        self.epsilon = epsilon    # ε-贪婪策略的探索概率

    def take_action(self, state):
        """ε-贪婪策略选动作：以 ε 概率随机探索，以 1-ε 概率利用当前最优估计"""
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)   # 随机探索
        else:
            action = np.argmax(self.Q_table[state])     # 贪婪利用
        return action

    def best_action(self, state):
        """返回当前状态下所有最优动作的指示列表（用于打印策略）"""
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):  # 若多个动作价值相同，全部标记为最优
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1):
        """
        Sarsa 更新公式（on-policy TD(0)）：
          Q(s0, a0) ← Q(s0, a0) + α * [r + γ * Q(s1, a1) - Q(s0, a0)]
        其中 a1 是在 s1 上用当前 ε-贪婪策略实际选取的动作（而非最优动作）。
        终止状态的 Q 值始终为 0（初始化后从不作为非终止状态被更新），
        所以达到目标时目标值自动退化为 r，无需额外判断 done。
        """
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    """
    打印每个格子的最优策略。
    - disaster：悬崖格子的状态索引列表，显示为 ****
    - end：终点格子的状态索引列表，显示为 EEEE
    - 其他格子：显示最优动作符号（若多个动作并列最优则同时显示）
    """
    for i in range(env.nrow):
        for j in range(env.ncol):
            state = i * env.ncol + j
            if state in disaster:
                print('****', end=' ')
            elif state in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(state)
                # 对于每个动作：如果是最优动作就打印对应符号，否则打印 'o'
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


if __name__ == '__main__':
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv(ncol, nrow)
    np.random.seed(0)
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
    num_episodes = 500  # 智能体在环境中运行的序列的数量

    return_list = []  # 记录每条回合的累计回报（不做折扣衰减，便于直观观察）
    for i in range(10):  # 将训练分为10段，每段显示一个进度条
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()         # 重置环境，获取初始状态
                # Sarsa 特点：在进入循环前就先选好第一个动作（与 Q-learning 不同）
                action = agent.take_action(state)
                done = False
                while not done:
                    # 执行动作 a，得到 r 和 s'
                    next_state, reward, done = env.step(action)
                    # on-policy：用当前 ε-贪婪策略在 s' 上选 a'（不是最优动作）
                    next_action = agent.take_action(next_state)
                    episode_return += reward  # 累计本回合原始回报（不乘折扣）
                    # 用 (s, a, r, s', a') 五元组更新 Q 表
                    agent.update(state, action, reward, next_state, next_action)
                    # 推进状态和动作：下一步的 (s, a) 就是本步选好的 (s', a')
                    state = next_state
                    action = next_action
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每10条回合打印一次平均回报
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':  '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Sarsa on {}'.format('Cliff Walking'))
    plt.show()

    # action_meaning 顺序与 change 列表一致：上/下/左/右
    action_meaning = ['^', 'v', '<', '>']
    print('Sarsa算法最终收敛得到的策略为：')
    # 悬崖：状态37~46（底行第2到第11列），终点：状态47（右下角）
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47])