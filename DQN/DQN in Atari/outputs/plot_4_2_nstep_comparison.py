"""
4.2 Parameter Tuning - DQN on PongNoFrameskip-v4 (20M env steps)
    Effect of n_step ∈ {1, 2, 3, 5} on reward, TD-loss, stability, and speed.

4.3 Environment Generalization
    Same DQN applied to BreakoutNoFrameskip-v4 and SpaceInvadersNoFrameskip-v4.

Run from the outputs/ directory:
    conda run -n rl310 python plot_4_2_nstep_comparison.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

BASE = os.path.dirname(os.path.abspath(__file__))


# ── helpers ───────────────────────────────────────────────────────────────────
def load(rel_path):
    return pd.read_csv(os.path.join(BASE, rel_path))


def load_tfevents(run_dir, tag):
    """Load a scalar tag from the TFEvents file in run_dir as a DataFrame."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    fpath = next(
        os.path.join(run_dir, f)
        for f in sorted(os.listdir(run_dir))
        if f.startswith("events.out.tfevents")
    )
    ea = EventAccumulator(fpath)
    ea.Reload()
    items = ea.scalars.Items(tag)
    return pd.DataFrame({
        "Wall time": [e.wall_time for e in items],
        "Step":      [e.step      for e in items],
        "Value":     [e.value     for e in items],
    })


def arr(series):
    """Convert a pandas Series (or any array-like) to a plain numpy float array."""
    return np.asarray(series, dtype=float)


def smooth(y, w=7):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="same")


def first_reach(mean_vals, steps, threshold):
    """First env-step (in millions) at which mean_vals ≥ threshold; else None."""
    idx = next((i for i, v in enumerate(mean_vals) if v >= threshold), None)
    return steps[idx] / 1e6 if idx is not None else None


# ── load data ─────────────────────────────────────────────────────────────────
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

# n_step = 5  — pick run 2026-04-03_11-32-13 (all 5 runs identical in length)
N5_DIR  = os.path.join(BASE, "PongNoFrameskip-v4-DQN-nstep5", "2026-04-03_11-32-13")
n5_mean = load_tfevents(N5_DIR, "test/episode_rewards_mean")
n5_std  = load_tfevents(N5_DIR, "test/episode_rewards_std")
n5_net  = load_tfevents(N5_DIR, "network/loss")

# Breakout & SpaceInvaders
bo_mean = load("results4.3.1/Breakout rewards mean.csv")
bo_std  = load("results4.3.1/Breakout rewards std.csv")
si_mean = load("results4.3.2/SpaceInvaders rewards mean.csv")
si_std  = load("results4.3.2/SpaceInvaders rewards std.csv")

# ── styles ────────────────────────────────────────────────────────────────────
CFG = {
    1: dict(color="royalblue",  dark="navy",        label="n_step=1  (1-step TD)"),
    2: dict(color="tomato",     dark="darkred",     label="n_step=2"),
    3: dict(color="seagreen",   dark="darkgreen",   label="n_step=3"),
    5: dict(color="darkorange", dark="saddlebrown", label="n_step=5"),
}

datasets = {
    1: (n1_mean, n1_std, n1_net),
    2: (n2_mean, n2_std, n2_net),
    3: (n3_mean, n3_std, n3_net),
    5: (n5_mean, n5_std, n5_net),
}

THRESHOLDS = [0, 10, 15, 19, 21]

# ── figure: 3 rows × 3 cols ───────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 18))
fig.suptitle(
    "DQN Analysis  |  n_step ∈ {1,2,3,5} on Pong  ·  Generalization to Breakout & SpaceInvaders  |  20M env steps",
    fontsize=13, fontweight="bold", y=0.995,
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

ax_reward = fig.add_subplot(gs[0, :2])   # Pong reward curves (spans 2 cols)
ax_miles  = fig.add_subplot(gs[0, 2])    # convergence milestones bar
ax_loss   = fig.add_subplot(gs[1, 0])    # TD-loss
ax_std    = fig.add_subplot(gs[1, 1])    # reward std
ax_speed  = fig.add_subplot(gs[1, 2])    # training speed
ax_bo     = fig.add_subplot(gs[2, 0])    # Breakout reward
ax_si     = fig.add_subplot(gs[2, 1])    # SpaceInvaders reward
ax_envbar = fig.add_subplot(gs[2, 2])    # cross-env final score


# ═══════════════════════════════════════════════════════════════════════════════
# Panel A — Pong reward curves  (n=1,2,3,5 on same axes)
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_reward
ax.set_title("[4.2]  Pong Reward Curves: n_step ∈ {1, 2, 3, 5}  (same axes)", fontsize=12, fontweight="bold")

milestones = {}
for n, (mean_df, std_df, _) in datasets.items():
    steps   = arr(mean_df["Step"])
    mean    = arr(mean_df["Value"])
    std     = arr(std_df["Value"])
    steps_M = steps / 1e6
    c = CFG[n]
    ax.fill_between(steps_M, mean - std, mean + std, alpha=0.12, color=c["color"])
    ax.plot(steps_M, mean,            color=c["color"], lw=0.8, alpha=0.5)
    ax.plot(steps_M, smooth(mean, 7), color=c["dark"],  lw=2.0, label=c["label"])
    milestones[n] = {t: first_reach(mean, steps, t) for t in THRESHOLDS}

ax.axhline(21,  color="black", lw=1.0, ls="--", alpha=0.5, label="Perfect score (21)")
ax.axhline(0,   color="gray",  lw=0.6, ls=":")
ax.axhline(-21, color="gray",  lw=0.6, ls=":")
ax.set_xlabel("Environment Steps (×10⁶)", fontsize=11)
ax.set_ylabel("Mean Episode Reward (10 epi.)", fontsize=11)
ax.legend(fontsize=9, loc="lower right")
ax.set_ylim(-23, 24)
ax.set_xlim(0, 20)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

final_means = {n: float(arr(datasets[n][0]["Value"])[-10:].mean()) for n in [1, 2, 3, 5]}
best_n = max(final_means, key=lambda n: final_means[n])
note = (
    "Final score (last-10 eval mean):\n"
    + "\n".join(f"  n={n}: {final_means[n]:.1f}" for n in [1, 2, 3, 5])
    + f"\n  ★ Best: n={best_n}"
)
ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=8.5, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85))


# ═══════════════════════════════════════════════════════════════════════════════
# Panel B — convergence milestones bar (4 groups for n=1,2,3,5)
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_miles
ax.set_title("Convergence Milestones\n(env-steps to first reach threshold)", fontsize=10, fontweight="bold")

thresholds_plot = [0, 15, 19]
x       = np.arange(len(thresholds_plot))
width   = 0.18
ns      = [1, 2, 3, 5]
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

for i, n in enumerate(ns):
    heights = [
        milestones[n][t] if milestones[n][t] is not None else 20.0
        for t in thresholds_plot
    ]
    bars = ax.bar(x + offsets[i], heights, width,
                  color=CFG[n]["color"], alpha=0.85, label=f"n={n}",
                  edgecolor="white", linewidth=0.5)
    for bar, h, t in zip(bars, heights, thresholds_plot):
        txt = f"{h:.1f}M" if milestones[n][t] is not None else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.25,
                txt, ha="center", va="bottom", fontsize=6)

ax.set_xticks(x)
ax.set_xticklabels([f"score≥{t}" for t in thresholds_plot], fontsize=9)
ax.set_ylabel("Steps to reach (M)", fontsize=9)
ax.set_ylim(0, 22)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel C — TD-loss curves
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_loss
ax.set_title("[4.2]  TD-Loss Curves", fontsize=10, fontweight="bold")

for n, (_, _, net_df) in datasets.items():
    vals  = arr(net_df["Value"])
    steps = arr(net_df["Step"])
    ds    = max(1, len(vals) // 200)     # keep ~200 points regardless of length
    ls    = steps[::ds] / 1e6
    lv    = vals[::ds]
    c     = CFG[n]
    ax.plot(ls, lv,              color=c["color"], lw=0.5, alpha=0.35)
    ax.plot(ls, smooth(lv, 20),  color=c["dark"],  lw=1.8, label=c["label"])

ax.set_xlabel("Gradient Steps (×10⁶)", fontsize=10)
ax.set_ylabel("TD Loss (MSE)", fontsize=10)
ax.set_yscale("log")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which="both")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel D — reward std (policy stability)
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_std
ax.set_title("Reward Std over Time  (Policy Stability)", fontsize=10, fontweight="bold")

for n, (mean_df, std_df, _) in datasets.items():
    steps_M = arr(mean_df["Step"]) / 1e6
    std     = arr(std_df["Value"])
    c       = CFG[n]
    ax.plot(steps_M, std,            color=c["color"], lw=0.6, alpha=0.4)
    ax.plot(steps_M, smooth(std, 7), color=c["dark"],  lw=1.8, label=c["label"])

ax.set_xlabel("Environment Steps (×10⁶)", fontsize=10)
ax.set_ylabel("Reward Std", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# Panel E — training speed (wall-clock time per 100k steps)
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_speed
ax.set_title("Training Speed  (Wall-clock / 100k env-steps)", fontsize=10, fontweight="bold")

for n, (mean_df, _, _) in datasets.items():
    wt      = arr(mean_df["Wall time"])
    steps_M = arr(mean_df["Step"])[1:] / 1e6
    dt_min  = np.diff(wt) / 60
    c = CFG[n]
    ax.plot(steps_M, dt_min,            color=c["color"], lw=0.6, alpha=0.4)
    ax.plot(steps_M, smooth(dt_min, 5), color=c["dark"],  lw=1.8, label=c["label"])

ax.set_xlabel("Environment Steps (×10⁶)", fontsize=10)
ax.set_ylabel("Minutes / 100k env-steps", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# Panel F — Breakout reward curve
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_bo
ax.set_title("[4.3]  BreakoutNoFrameskip-v4", fontsize=10, fontweight="bold")

steps_M = np.asarray(bo_mean["Step"].values,   dtype=float) / 1e6
mean    = np.asarray(bo_mean["Value"].values,  dtype=float)
std     = np.asarray(bo_std["Value"].values,   dtype=float)
ax.fill_between(steps_M, mean - std, mean + std, alpha=0.15, color="steelblue")
ax.plot(steps_M, mean,            color="steelblue", lw=0.8, alpha=0.5)
ax.plot(steps_M, smooth(mean, 7), color="darkblue",  lw=2.0, label="Breakout")
ax.set_xlabel("Environment Steps (×10⁶)", fontsize=10)
ax.set_ylabel("Mean Episode Reward", fontsize=10)
ax.set_xlim(0, 20)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="upper left")
final_bo = float(mean[-10:].mean())
ax.text(0.98, 0.05, f"Final (last-10): {final_bo:.1f}",
        transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85))


# ═══════════════════════════════════════════════════════════════════════════════
# Panel G — SpaceInvaders reward curve
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_si
ax.set_title("[4.3]  SpaceInvadersNoFrameskip-v4", fontsize=10, fontweight="bold")

steps_M = arr(si_mean["Step"]) / 1e6
mean    = arr(si_mean["Value"])
std     = arr(si_std["Value"])
ax.fill_between(steps_M, mean - std, mean + std, alpha=0.15, color="mediumpurple")
ax.plot(steps_M, mean,            color="mediumpurple", lw=0.8, alpha=0.5)
ax.plot(steps_M, smooth(mean, 7), color="indigo",       lw=2.0, label="SpaceInvaders")
ax.set_xlabel("Environment Steps (×10⁶)", fontsize=10)
ax.set_ylabel("Mean Episode Reward", fontsize=10)
ax.set_xlim(0, 20)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="upper left")
final_si = float(mean[-10:].mean())
ax.text(0.98, 0.05, f"Final (last-10): {final_si:.1f}",
        transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85))


# ═══════════════════════════════════════════════════════════════════════════════
# Panel H — cross-environment final score (log-scale bar, raw scores)
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_envbar
ax.set_title("[4.3]  Cross-Env Final Score\n(log scale; best n_step for Pong)", fontsize=10, fontweight="bold")

final_pong = float(final_means[best_n])
# Pong scores span -21..+21; shift so bars are always positive for log-scale
pong_shifted = final_pong + 22.0          # offset by 22 so min possible = 1
bo_shifted   = final_bo                   # already >> 0
si_shifted   = final_si                   # already >> 0

envs       = [f"Pong\n(n={best_n})", "Breakout", "SpaceInvaders"]
bar_heights = [pong_shifted, bo_shifted, si_shifted]
raw_scores  = [final_pong,   final_bo,  final_si]
bar_colors  = [CFG[best_n]["color"], "steelblue", "mediumpurple"]

bars = ax.bar(envs, bar_heights, color=bar_colors, alpha=0.85,
              edgecolor="white", linewidth=0.8)
for bar, raw in zip(bars, raw_scores):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.12,
            f"{raw:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_yscale("log")
ax.set_ylim(1, max(bar_heights) * 3)
ax.set_ylabel("Mean Episode Reward (log scale)", fontsize=9)
ax.text(0.5, -0.12,
        "* Pong bar = score + 22 (shifted for log scale; label shows raw score)",
        transform=ax.transAxes, fontsize=7, ha="center", color="gray")
ax.grid(True, alpha=0.3, axis="y", which="both")
ax.tick_params(axis="x", labelsize=9)


# ── save ──────────────────────────────────────────────────────────────────────
out = os.path.join(BASE, "4_2_nstep_comparison.png")
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
