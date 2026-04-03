"""
4.2 Parameter Tuning Analysis - DQN on PongNoFrameskip-v4 (20M env steps)
==========================================================================
Direction 1: Effect of n_step on reward performance
  - All three reward curves on the same axes for direct comparison of
    convergence speed and final score
  - Milestone bar chart: env-steps required to reach key score thresholds

Direction 2: Effect of n_step on training dynamics
  - All three TD-loss curves on the same axes (compare magnitude & trend)
  - Reward std over time (policy stability)
  - Training wall-clock speed per 100k env-steps

Run from the outputs/ directory:
    conda run -n rl310 python plot_4_2_nstep_comparison.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

# ── 路径 ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

def load(rel_path):
    return pd.read_csv(os.path.join(BASE, rel_path))

# n_step = 1
n1_mean = load("results4.1/2026-03-31_22-49-49_episode_rewards_mean.csv")
n1_std  = load("results4.1/2026-03-31_22-49-49_episode_reward_std.csv")
n1_net  = load("results4.1/2026-03-31_22-49-49_network.csv")

# n_step = 2
n2_mean = load("result4.2.2/result_mean_n=2.csv")
n2_std  = load("result4.2.2/result_std_n=2.csv")
n2_net  = load("result4.2.2/network_n=2.csv")

# n_step = 3
n3_mean = load("results4.2.1/reward_mean_n=3.csv")
n3_std  = load("results4.2.1/reward_std_n=3.csv")
n3_net  = load("results4.2.1/network_n=3.csv")

# ── smooth helper ─────────────────────────────────────────────────────────────
def smooth(y, w=7):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="same")

# ── 样式 ──────────────────────────────────────────────────────────────────────
CFG = {
    1: dict(color="royalblue", dark="navy",      label="n_step = 1  (1-step TD)"),
    2: dict(color="tomato",    dark="darkred",   label="n_step = 2"),
    3: dict(color="seagreen",  dark="darkgreen", label="n_step = 3"),
}

datasets = {
    1: (n1_mean, n1_std, n1_net),
    2: (n2_mean, n2_std, n2_net),
    3: (n3_mean, n3_std, n3_net),
}

# ── 收敛里程碑 ─────────────────────────────────────────────────────────────────
THRESHOLDS = [0, 10, 15, 19, 21]

def first_reach(mean_vals, steps, threshold):
    """首次达到 threshold 时的 env steps；若未达到返回 None"""
    idx = next((i for i, v in enumerate(mean_vals) if v >= threshold), None)
    return steps[idx] / 1e6 if idx is not None else None

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    "DQN Parameter Tuning: n_step in {1, 2, 3}  |  PongNoFrameskip-v4  |  20M env steps",
    fontsize=14, fontweight="bold", y=0.99
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

ax_reward = fig.add_subplot(gs[0, :2])   # Dir 1: reward curves (2 cols wide)
ax_miles  = fig.add_subplot(gs[0, 2])    # Dir 1: convergence milestones bar
ax_loss   = fig.add_subplot(gs[1, 0])    # Dir 2: TD-Loss curves
ax_std    = fig.add_subplot(gs[1, 1])    # Dir 2: reward std
ax_speed  = fig.add_subplot(gs[1, 2])    # Dir 2: training speed


# ═══════════════════════════════════════════════════════════════════════════════
# 方向一 Panel A — 奖励曲线（同轴对比）
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_reward
ax.set_title("[Direction 1]  Reward Curves Comparison (same axes)", fontsize=12, fontweight="bold")

milestones = {}   # n → dict{threshold: steps_M}
for n, (mean_df, std_df, _) in datasets.items():
    steps   = mean_df["Step"].values
    mean    = mean_df["Value"].values
    std     = std_df["Value"].values
    steps_M = steps / 1e6
    c = CFG[n]

    ax.fill_between(steps_M, mean - std, mean + std,
                    alpha=0.12, color=c["color"])
    ax.plot(steps_M, mean,              color=c["color"], lw=0.8, alpha=0.5)
    ax.plot(steps_M, smooth(mean, 7),   color=c["dark"],  lw=2.0, label=c["label"])

    milestones[n] = {t: first_reach(mean, steps, t) for t in THRESHOLDS}

ax.axhline(21,   color="black",  lw=1.0, ls="--", alpha=0.5, label="Perfect score (21)")
ax.axhline(0,    color="gray",   lw=0.6, ls=":")
ax.axhline(-21,  color="gray",   lw=0.6, ls=":")

ax.set_xlabel("Environment Steps (×10⁶)", fontsize=11)
ax.set_ylabel("Mean Episode Reward (10 epi.)", fontsize=11)
ax.legend(fontsize=9, loc="lower right")
ax.set_ylim(-23, 24)
ax.set_xlim(0, 20)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

# 分析注释
final_means = {n: datasets[n][0]["Value"].values[-10:].mean() for n in [1,2,3]}
best_n = max(final_means, key=final_means.get)
note = (
    f"Final score (last-10 eval mean):\n"
    f"  n=1: {final_means[1]:.1f}   n=2: {final_means[2]:.1f}   n=3: {final_means[3]:.1f}\n"
    f"  Best: n={best_n}"
)
ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=8,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85))


# ═══════════════════════════════════════════════════════════════════════════════
# 方向一 Panel B — 收敛里程碑（柱状图）
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_miles
ax.set_title("Convergence Milestones\n(env-steps to first reach each threshold)", fontsize=10, fontweight="bold")

thresholds_plot = [0, 15, 19]
x = np.arange(len(thresholds_plot))
width = 0.25

for i, n in enumerate([1, 2, 3]):
    heights = []
    for t in thresholds_plot:
        v = milestones[n][t]
        heights.append(v if v is not None else 20.0)   # 若未达到，显示 20M (满训练量)
    bars = ax.bar(x + (i - 1) * width, heights, width,
                  color=CFG[n]["color"], alpha=0.85, label=f"n={n}",
                  edgecolor="white", linewidth=0.5)
    for bar, h, t in zip(bars, heights, thresholds_plot):
        reach = milestones[n][t]
        txt = f"{h:.1f}M" if reach is not None else "N/A"
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                txt, ha="center", va="bottom", fontsize=6.5)

ax.set_xticks(x)
ax.set_xticklabels([f"score≥{t}" for t in thresholds_plot], fontsize=9)
ax.set_ylabel("Steps to reach (M)", fontsize=9)
ax.set_ylim(0, 22)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")


# ═══════════════════════════════════════════════════════════════════════════════
# 方向二 Panel C — TD Loss 曲线（同轴对比）
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_loss
ax.set_title("[Direction 2]  TD-Loss Curves Comparison (same axes)", fontsize=10, fontweight="bold")

DS = 100   # down-sample for readability
for n, (_, _, net_df) in datasets.items():
    ls = net_df["Step"].values[::DS] / 1e6
    lv = net_df["Value"].values[::DS]
    c  = CFG[n]
    ax.plot(ls, lv,              color=c["color"], lw=0.5, alpha=0.35)
    ax.plot(ls, smooth(lv, 20),  color=c["dark"],  lw=1.8, label=c["label"])

ax.set_xlabel("Gradient Steps (×10⁶)", fontsize=10)
ax.set_ylabel("TD Loss (MSE)", fontsize=10)
ax.set_yscale("log")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which="both")


# ═══════════════════════════════════════════════════════════════════════════════
# 方向二 Panel D — 奖励标准差（策略稳定性，同轴对比）
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_std
ax.set_title("Reward Std over Time (Policy Stability)", fontsize=10, fontweight="bold")

for n, (mean_df, std_df, _) in datasets.items():
    steps_M = mean_df["Step"].values / 1e6
    std     = std_df["Value"].values
    c       = CFG[n]
    ax.plot(steps_M, std,             color=c["color"], lw=0.6, alpha=0.4)
    ax.plot(steps_M, smooth(std, 7),  color=c["dark"],  lw=1.8, label=c["label"])

ax.set_xlabel("Environment Steps (×10⁶)", fontsize=10)
ax.set_ylabel("Reward Std", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# 方向二 Panel E — 训练速度（wall-clock time per 100k steps）
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_speed
ax.set_title("Training Speed (Wall-clock time / 100k env-steps)", fontsize=10, fontweight="bold")

for n, (mean_df, _, _) in datasets.items():
    wt = mean_df["Wall time"].values
    steps_M = mean_df["Step"].values[1:] / 1e6
    dt_min  = np.diff(wt) / 60          # seconds → minutes
    c = CFG[n]
    ax.plot(steps_M, dt_min,             color=c["color"], lw=0.6, alpha=0.4)
    ax.plot(steps_M, smooth(dt_min, 5),  color=c["dark"],  lw=1.8, label=c["label"])

ax.set_xlabel("Environment Steps (×10⁶)", fontsize=10)
ax.set_ylabel("Minutes / 100k env-steps", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)


# ── 保存 ──────────────────────────────────────────────────────────────────────
out = os.path.join(BASE, "4_2_nstep_comparison.png")
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
