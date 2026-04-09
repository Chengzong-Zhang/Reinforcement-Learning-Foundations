"""
Regenerate the Section 4.1 figure with three panels:
  Left  : Test Reward vs. Training Steps
  Middle: Average Q-Value (train/q_value_mean) vs. Training Steps
  Right : TD Loss vs. Training Steps
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE    = os.path.dirname(os.path.abspath(__file__))
PREFIX  = os.path.join(BASE, "2026-03-31_22-49-49")
RUN_DIR = os.path.join(os.path.dirname(BASE),
                       "PongNoFrameskip-v4-DQN", "2026-03-31_22-49-49")


def smooth(y, w=5):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="same")


def load_tfevents_tag(run_dir, tag):
    fpath = next(
        os.path.join(run_dir, f)
        for f in sorted(os.listdir(run_dir))
        if f.startswith("events.out.tfevents")
    )
    ea = EventAccumulator(fpath)
    ea.Reload()
    items = ea.scalars.Items(tag)
    steps  = np.array([e.step  for e in items], dtype=float)
    values = np.array([e.value for e in items], dtype=float)
    return steps, values


# ── load CSV data (reward) ─────────────────────────────────────────────────────
reward_df = pd.read_csv(f"{PREFIX}_episode_rewards_mean.csv")
std_df    = pd.read_csv(f"{PREFIX}_episode_reward_std.csv")
net_df    = pd.read_csv(f"{PREFIX}_network.csv")

steps_r   = reward_df["Step"].values / 1e6
rewards   = reward_df["Value"].values
stds      = std_df["Value"].values

# ── load q_value_mean from TFEvents ───────────────────────────────────────────
q_steps, q_values = load_tfevents_tag(RUN_DIR, "train/q_value_mean")
q_steps_M = q_steps / 1e6

# ── figure: 1 row × 3 cols ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("DQN on PongNoFrameskip-v4  (n_step=1,  20M env steps)",
             fontsize=13, fontweight="bold", y=1.02)

# ── Left: Test Reward ──────────────────────────────────────────────────────────
ax = axes[0]
ax.fill_between(steps_r, rewards - stds, rewards + stds,
                alpha=0.15, color="royalblue", label="±1 std")
ax.plot(steps_r, rewards,            color="royalblue", lw=1.5, alpha=0.5,
        label="DQN (test reward)")
ax.plot(steps_r, smooth(rewards, 10), color="navy",      lw=1.5, ls="--",
        alpha=0.8, label="Smoothed")
ax.axhline(20,    color="crimson",  ls="--", lw=1.2, label="Paper DQN avg = 20")
ax.axhline(21,    color="seagreen", ls=":",  lw=1.2, label="Perfect score = 21")
ax.axhline(-20.4, color="gray",     ls=":",  lw=1.0, label="Random = −20.4")

peak_idx = np.argmax(rewards)
ax.annotate(f"Peak = {rewards[peak_idx]:.0f}",
            xy=(steps_r[peak_idx], rewards[peak_idx]),
            xytext=(steps_r[peak_idx] - 5, rewards[peak_idx] - 4),
            fontsize=8, color="seagreen",
            arrowprops=dict(arrowstyle="->", color="seagreen", lw=0.8))

last10_mean = rewards[-10:].mean()
ax.text(0.02, 0.03,
        f"Last-10 mean: {last10_mean:.2f}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

ax.set_xlabel("Environment Steps (×10⁶)", fontsize=11)
ax.set_ylabel("Mean Episode Reward", fontsize=11)
ax.set_title("Test Reward", fontsize=11)
ax.legend(fontsize=7.5, loc="lower right")
ax.set_xlim(0, steps_r[-1])
ax.set_ylim(-22, 22)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

# ── Middle: Average Q-Value ───────────────────────────────────────────────────
ax = axes[1]
ds = max(1, len(q_steps_M) // 500)   # downsample to ~500 pts
ax.plot(q_steps_M[::ds], q_values[::ds],
        color="mediumorchid", lw=0.6, alpha=0.4, label="Q-mean (raw)")
ax.plot(q_steps_M[::ds], smooth(q_values[::ds], 30),
        color="purple",       lw=2.0,             label="Q-mean (smoothed)")
ax.set_xlabel("Environment Steps (×10⁶)", fontsize=11)
ax.set_ylabel("Average Q-Value", fontsize=11)
ax.set_title("Average Q-Value  (train/q_value_mean)", fontsize=11)
ax.legend(fontsize=8)
ax.set_xlim(0, q_steps_M[-1])
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

# ── Right: TD Loss ─────────────────────────────────────────────────────────────
ax = axes[2]
net_steps_M = net_df["Step"].values / 1e6
net_loss    = net_df["Value"].values
ax.plot(net_steps_M, net_loss,            color="tomato",  lw=0.6, alpha=0.5,
        label="Loss (raw)")
ax.plot(net_steps_M, smooth(net_loss, 50), color="darkred", lw=1.8,
        label="Loss (smoothed)")
ax.set_xlabel("Environment Steps (×10⁶)", fontsize=11)
ax.set_ylabel("TD Loss (MSE)", fontsize=11)
ax.set_title("Network TD Loss", fontsize=11)
ax.legend(fontsize=8)
ax.set_xlim(0, net_steps_M[-1])
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

plt.tight_layout()
out = os.path.join(BASE, "paper_figure_3panel.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()
