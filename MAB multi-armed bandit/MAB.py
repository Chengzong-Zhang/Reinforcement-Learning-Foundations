import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    """ 伯努利多臂老虎机,输入K表示拉杆个数 """
    def __init__(self, K):
        self.probs: np.ndarray = np.array(np.random.uniform(size=K))  # 随机生成（均匀分布）K个0～1的数,作为拉动每根拉杆的获奖
        # 概率
        self.best_idx = np.argmax(self.probs)  # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx]  # 最大的获奖概率
        self.K = K

    def step(self, k):
        # 当玩家选择了k号拉杆后,根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未
        # 获奖）
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


class Solver:
    """ 多臂老虎机算法基本框架 """
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # 每根拉杆的尝试次数，K是多少个，这个生成K元素的全0列表
        self.regret = 0.  # 当前步的累积懊悔
        self.actions = []  # 维护一个列表,记录每一步的动作
        self.regrets = []  # 维护一个列表,记录每一步的累积懊悔

    def update_regret(self, k):
        # 计算累积懊悔并保存,k为本次动作选择的拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)   #pop扔，append加

    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆,由每个具体的策略实现
        #模板方法模式，意思是这个方法必须由子类实现，如果你直接调用我，就报错提醒你
        raise NotImplementedError

    def run(self, num_steps):
        # 运行一定次数,num_steps为总运行次数
        for _ in range(num_steps):
            k = self.run_one_step()    # 策略决定：这一步拉第 k 根，k就是个索引
            self.counts[k] += 1        # 第 k 根的尝试次数 +1
            self.actions.append(k)     # 记录这一步选了 k
            self.update_regret(k)      # 用 k 计算懊悔值


#------------------------------------------------
#算法选择
#--------------------------------------------------------

class EpsilonGreedy(Solver):
    # ε-贪婪算法在完全贪婪算法的基础上添加了噪声，每次以概率1-ε选择以往经验中
    # 期望奖励估值最大的那根拉杆（利用），以概率ε随机选择一根拉杆（探索）
    # 随着探索次数的不断增加，我们对各个动作的奖励估计得越来越准，
    # 我们可以令ε随时间衰减，即探索的概率将会不断降低
    #有点像遗传算法等启发式算法，贪心加扰动
    """ epsilon贪婪算法,继承Solver类 """
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        #初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)  # 随机选择一根拉杆
        else:
            k = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆
        r = self.bandit.step(k)  # 得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        #增量式计算均值的公式
        return k


class DecayingEpsilonGreedy(Solver):
    """ epsilon值随时间衰减的epsilon-贪婪算法（改进版）,继承Solver类 """
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:  # epsilon值随时间衰减
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])

        return k


#-----------------------------------------------------------------------------
# 上置信界算法，理论依据 Hoeffding's Inequality（大偏差不等式）
#
# 霍夫丁不等式（Hoeffding's Inequality）：
#   对独立有界随机变量 X_1,...,X_n（取值在 [0,1]），其样本均值 X̄ 满足：
#
#   P( E[X̄] - X̄ ≥ t ) ≤ exp( -2nt² )
#
#   即：真实均值超过样本均值 t 以上的概率，随 t 和 n 指数级衰减。
#
# 应用到 UCB：令上式右边等于 δ，解出 t，得到置信上界：
#
#   UCB_k(t) = μ̂_k(t) + √( log(t) / (2 * N_k(t)) )
#
#   其中 μ̂_k(t) 是第 k 根拉杆的当前估计均值，N_k(t) 是拉动次数，t 是总步数。
#   coef 参数用于调节不确定性项的权重。

class UCB(Solver):
    """ UCB算法,继承Solver类 """
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1)))  # 计算上置信界
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    np.random.seed(1)  # 设定随机种子,使实验具有可重复性
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    print("随机生成了一个%d臂伯努利老虎机" % K)
    print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %
        (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

    np.random.seed(1)
    epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
    epsilon_greedy_solver.run(5000)
    print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
    plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])

    # ── 敏感性分析：不同 ε 值对比 ──────────────────────────────────
    np.random.seed(0)
    epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
    epsilon_greedy_solver_list = [
        EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons
    ]
    epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
    for solver in epsilon_greedy_solver_list:
        solver.run(5000)
    plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)
    print('\n── 敏感性分析结论 ──')
    for e, solver in zip(epsilons, epsilon_greedy_solver_list):
        print('ε=%-6s  累积懊悔：%.4f' % (e, solver.regret))
    print('结论：ε 越大，探索越多，后期因无效尝试累积懊悔持续攀升；ε 越小，收敛快但前期可能错过更优拉杆。')

    # ── 消融实验：固定 ε vs 衰减 ε ──────────────────────────────────
    np.random.seed(1)
    ablation_fixed = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
    ablation_decaying = DecayingEpsilonGreedy(bandit_10_arm)
    ablation_fixed.run(5000)
    ablation_decaying.run(5000)
    plot_results([ablation_fixed, ablation_decaying],
                 ["EpsilonGreedy(ε=0.01，固定)", "DecayingEpsilonGreedy(衰减)"])
    print('\n── 消融实验结论 ──')
    print('EpsilonGreedy(固定)   累积懊悔：%.4f（线性增长：始终保留固定探索率，懊悔持续累积）' % ablation_fixed.regret)
    print('DecayingEpsilonGreedy 累积懊悔：%.4f（次线性增长：探索率随时间衰减，增长逐渐趋缓）' % ablation_decaying.regret)
    print('结论：衰减策略在后期几乎完全利用，累积懊悔曲线斜率趋近于 0，证明了随时间减少探索的必要性。')

    np.random.seed(1)
    coef = 1  # 控制不确定性比重的系数
    UCB_solver = UCB(bandit_10_arm, coef)
    UCB_solver.run(5000)
    print('上置信界算法的累积懊悔为：', UCB_solver.regret)
    plot_results([UCB_solver], ["UCB"])
