"""
Plot paper-quality training curves for DQN on PongNoFrameskip-v4.
Run from the outputs/results directory or adjust BASE_DIR below.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREFIX   = os.path.join(BASE_DIR, "2026-03-31_22-49-49")

reward_df = pd.read_csv(f"{PREFIX}_episode_rewards_mean.csv")
std_df    = pd.read_csv(f"{PREFIX}_episode_reward_std.csv")
net_df    = pd.read_csv(f"{PREFIX}_network.csv")

steps   = reward_df["Step"].values
rewards = reward_df["Value"].values
stds    = std_df["Value"].values

# ── smooth helper ──────────────────────────────────────────────────────────
def smooth(y, w=5):
    return np.convolve(y, np.ones(w)/w, mode="same")

# ── figure layout ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("DQN on PongNoFrameskip-v4  (Reproduced)",
             fontsize=13, fontweight="bold", y=1.01)

STEP_M = steps / 1e6   # convert to millions for x-axis

# ── Left: Test Reward ──────────────────────────────────────────────────────
ax = axes[0]
ax.fill_between(STEP_M, rewards - stds, rewards + stds,
                alpha=0.15, color="royalblue", label="±1 std")
ax.plot(STEP_M, rewards, color="royalblue", linewidth=1.8,
        label="DQN (ours, test)")
ax.plot(STEP_M, smooth(rewards, 10), color="navy", linewidth=1.0,
        linestyle="--", alpha=0.6, label="Smoothed")

ax.axhline(y=20,    color="crimson",    linestyle="--", linewidth=1.2, label="Paper DQN avg = 20")
ax.axhline(y=21,    color="seagreen",   linestyle=":",  linewidth=1.2, label="Perfect score = 21")
ax.axhline(y=-20.4, color="gray",       linestyle=":",  linewidth=1.0, label="Random = −20.4")

peak_idx = np.argmax(rewards)
ax.annotate(f"Peak = {rewards[peak_idx]:.0f}",
            xy=(STEP_M[peak_idx], rewards[peak_idx]),
            xytext=(STEP_M[peak_idx] - 4, rewards[peak_idx] - 3),
            fontsize=8, color="seagreen",
            arrowprops=dict(arrowstyle="->", color="seagreen", lw=0.8))

ax.set_xlabel("Environment Steps (×10⁶)", fontsize=11)
ax.set_ylabel("Mean Episode Reward", fontsize=11)
ax.set_title("Test Reward vs. Training Steps", fontsize=11)
ax.legend(fontsize=7.5, loc="lower right")
ax.set_xlim(0, STEP_M[-1])
ax.set_ylim(-22, 22)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

# ── Right: Network Loss ────────────────────────────────────────────────────
ax = axes[1]
net_steps = net_df["Step"].values / 1e6
net_loss  = net_df["Value"].values
ax.plot(net_steps, net_loss,   color="tomato", linewidth=0.6, alpha=0.5, label="Loss (raw)")
ax.plot(net_steps, smooth(net_loss, 50), color="darkred", linewidth=1.5, label="Loss (smoothed)")
ax.set_xlabel("Environment Steps (×10⁶)", fontsize=11)
ax.set_ylabel("TD Loss (MSE)", fontsize=11)
ax.set_title("Network Loss vs. Training Steps", fontsize=11)
ax.legend(fontsize=8)
ax.set_xlim(0, net_steps[-1])
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(BASE_DIR, "paper_figure.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
