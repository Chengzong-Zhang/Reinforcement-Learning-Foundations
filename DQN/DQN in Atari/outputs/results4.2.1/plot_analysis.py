import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

base = Path(__file__).parent

reward_mean = pd.read_csv(base / "reward_mean_n=3.csv")
reward_std  = pd.read_csv(base / "reward_std_n=3.csv")
network     = pd.read_csv(base / "network_n=3.csv")
time_df     = pd.read_csv(base / "time_n=3.csv")

steps   = reward_mean["Step"].values / 1e6        # unit: M steps
mean    = reward_mean["Value"].values
std     = reward_std["Value"].values

# ── smooth helper ──────────────────────────────────────────────────────────
def smooth(x, w=5):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")

# ── key milestones ─────────────────────────────────────────────────────────
idx_first_pos  = next((i for i, v in enumerate(mean) if v > 0),  None)
idx_first_15   = next((i for i, v in enumerate(mean) if v >= 15), None)
idx_first_21   = next((i for i, v in enumerate(mean) if v >= 21), None)
idx_converge   = next((i for i, v in enumerate(mean) if v >= 19), None)

# ── training time ──────────────────────────────────────────────────────────
t_start   = reward_mean["Wall time"].iloc[0]
t_end     = reward_mean["Wall time"].iloc[-1]
total_min = (t_end - t_start) / 60

# ── loss stats ─────────────────────────────────────────────────────────────
loss_steps  = network["Step"].values
loss_values = network["Value"].values

# ── figure layout ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
fig.suptitle("DQN n-step=3  |  PongNoFrameskip-v4  |  20M env steps", fontsize=15, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])   # reward curve full width
ax2 = fig.add_subplot(gs[1, 0])   # reward std
ax3 = fig.add_subplot(gs[1, 1])   # loss curve
ax4 = fig.add_subplot(gs[2, 0])   # reward distribution (histogram)
ax5 = fig.add_subplot(gs[2, 1])   # training speed (time per step)

# ─────────────────────────────────────────────────────────────────────────
# 1. Reward mean curve
# ─────────────────────────────────────────────────────────────────────────
ax1.fill_between(steps, mean - std, mean + std, alpha=0.2, color="steelblue", label="±1 std")
ax1.plot(steps, mean, color="steelblue", lw=1.2, alpha=0.5, label="raw")
ax1.plot(steps, smooth(mean, 7), color="navy", lw=2.0, label="smoothed (w=7)")
ax1.axhline(21, color="green",  lw=1.2, ls="--", label="max score (21)")
ax1.axhline(0,  color="gray",   lw=0.8, ls=":")
ax1.axhline(-21, color="red",   lw=1.2, ls="--", label="min score (-21)")

if idx_first_pos is not None:
    ax1.axvline(steps[idx_first_pos], color="orange", lw=1.2, ls="--",
                label=f"first >0 @ {steps[idx_first_pos]:.1f}M")
if idx_converge is not None:
    ax1.axvline(steps[idx_converge], color="purple", lw=1.2, ls="--",
                label=f"first ≥19 @ {steps[idx_converge]:.1f}M")

ax1.set_xlabel("Environment Steps (M)", fontsize=11)
ax1.set_ylabel("Mean Reward (10 episodes)", fontsize=11)
ax1.set_title("Training Reward Curve", fontsize=12)
ax1.legend(fontsize=8, loc="lower right")
ax1.set_ylim(-23, 24)
ax1.grid(True, alpha=0.3)

# ─────────────────────────────────────────────────────────────────────────
# 2. Reward std
# ─────────────────────────────────────────────────────────────────────────
ax2.plot(steps, std, color="coral", lw=1.2, alpha=0.6)
ax2.plot(steps, smooth(std, 7), color="darkred", lw=2.0)
ax2.set_xlabel("Environment Steps (M)", fontsize=10)
ax2.set_ylabel("Reward Std", fontsize=10)
ax2.set_title("Reward Standard Deviation", fontsize=11)
ax2.grid(True, alpha=0.3)

# ─────────────────────────────────────────────────────────────────────────
# 3. Network loss (down-sampled every 200 pts for readability)
# ─────────────────────────────────────────────────────────────────────────
ds = 200
ax3.plot(loss_steps[::ds] / 1e6, loss_values[::ds], color="gray", lw=0.8, alpha=0.5)
ax3.plot(loss_steps[::ds] / 1e6, smooth(loss_values[::ds], 15), color="darkgreen", lw=1.8)
ax3.set_xlabel("Gradient Steps (M)", fontsize=10)
ax3.set_ylabel("Loss", fontsize=10)
ax3.set_title("Network Loss", fontsize=11)
ax3.set_yscale("log")
ax3.grid(True, alpha=0.3, which="both")

# ─────────────────────────────────────────────────────────────────────────
# 4. Reward histogram — early / mid / late
# ─────────────────────────────────────────────────────────────────────────
n = len(mean)
phases = {
    "Early (0–33%)":  mean[:n//3],
    "Mid   (33–67%)": mean[n//3:2*n//3],
    "Late  (67–100%)":mean[2*n//3:],
}
colors = ["tomato", "gold", "mediumseagreen"]
for (label, vals), col in zip(phases.items(), colors):
    ax4.hist(vals, bins=20, alpha=0.6, label=label, color=col, density=True)
ax4.set_xlabel("Reward", fontsize=10)
ax4.set_ylabel("Density", fontsize=10)
ax4.set_title("Reward Distribution by Phase", fontsize=11)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# ─────────────────────────────────────────────────────────────────────────
# 5. Training speed — seconds per 100k env steps
# ─────────────────────────────────────────────────────────────────────────
wall_times = reward_mean["Wall time"].values
dt = np.diff(wall_times)          # seconds per 100k steps
ax5.plot(steps[1:], dt / 60, color="slateblue", lw=1.2, alpha=0.6)
ax5.plot(steps[1:], smooth(dt / 60, 7), color="indigo", lw=2.0)
ax5.set_xlabel("Environment Steps (M)", fontsize=10)
ax5.set_ylabel("Minutes / 100k steps", fontsize=10)
ax5.set_title("Training Speed", fontsize=11)
ax5.grid(True, alpha=0.3)

# ─────────────────────────────────────────────────────────────────────────
# Stats box
# ─────────────────────────────────────────────────────────────────────────
late_mean = mean[2*n//3:]
stats_text = (
    f"Total training time : {total_min:.0f} min  ({total_min/60:.1f} h)\n"
    f"Final reward (mean±std) : {late_mean.mean():.2f} ± {late_mean.std():.2f}\n"
    f"Peak reward  : {mean.max():.1f}  @ {steps[mean.argmax()]:.1f}M steps\n"
    f"First reward > 0  : {steps[idx_first_pos]:.1f}M steps\n"
    f"First reward ≥ 19 : {steps[idx_converge]:.1f}M steps\n"
    f"Scores = 21 (perfect) : {(mean == 21).sum()} / {len(mean)} evals"
)
fig.text(0.5, 0.01, stats_text, ha="center", va="bottom", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

out_path = base / "analysis_n=3.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
