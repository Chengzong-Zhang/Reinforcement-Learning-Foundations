"""Generate the Ant/Hopper extended-environment figure for the TRPO report.

Pulls test/episode_rewards_mean curves from all Ant-v5-TRPO and Hopper-v5-TRPO
runs under outputs_server_latest/outputs/, smooths them, plots one panel per
environment with mean (across runs) and shaded min-max band. Saves PDF into
figures/exploration/.
"""

from __future__ import annotations

import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tbparse import SummaryReader

BASE = r"c:/coding/py/RL/outputs_server_latest/outputs"
OUT = r"c:/coding/py/RL/TRPO/TRPO实现/figures/exploration/extended_envs_rewards.pdf"

ENVS = {
    "Ant-v5": "Ant-v5-TRPO",
    "Hopper-v5": "Hopper-v5-TRPO",
}

REWARD_TAG = "test/episode_rewards_mean"


def load_runs(env_dir: str):
    runs = []
    for run_dir in sorted(glob(os.path.join(BASE, env_dir, "*"))):
        cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
        if not os.path.isfile(cfg_path):
            continue
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        algo = cfg.get("algorithm", {})
        traj = bool(algo.get("collect_traj", False))
        norm = bool(algo.get("advan_norm", True))
        df = SummaryReader(run_dir).scalars
        if df.empty:
            continue
        sub = df[df["tag"] == REWARD_TAG].sort_values("step")
        if sub.empty:
            continue
        runs.append(
            {
                "path": run_dir,
                "traj": traj,
                "norm": norm,
                "steps": sub["step"].to_numpy(),
                "reward": sub["value"].to_numpy(),
            }
        )
    return runs


def resample(steps: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, steps, values, left=np.nan, right=np.nan)


def smooth(x: np.ndarray, w: int = 9) -> np.ndarray:
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def plot():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    palette = {
        ("rollout", True): "#1f77b4",
        ("rollout", False): "#d62728",
        ("trajectory", True): "#2ca02c",
        ("trajectory", False): "#ff7f0e",
    }
    label = {
        ("rollout", True): "rollout, norm=True",
        ("rollout", False): "rollout, norm=False",
        ("trajectory", True): "trajectory, norm=True",
        ("trajectory", False): "trajectory, norm=False",
    }

    for ax, (env_name, env_dir) in zip(axes, ENVS.items()):
        runs = load_runs(env_dir)
        # Group runs by config
        groups: dict[tuple[str, bool], list[dict]] = {}
        for r in runs:
            mode = "trajectory" if r["traj"] else "rollout"
            groups.setdefault((mode, r["norm"]), []).append(r)

        for cfg, items in groups.items():
            # Build a common step grid from the overlap range
            max_step = min(r["steps"][-1] for r in items)
            grid = np.linspace(0, max_step, 200)
            curves = np.stack(
                [resample(r["steps"], r["reward"], grid) for r in items]
            )
            mean = np.nanmean(curves, axis=0)
            lo = np.nanmin(curves, axis=0)
            hi = np.nanmax(curves, axis=0)
            color = palette[cfg]
            ax.plot(
                grid,
                smooth(mean),
                color=color,
                label=f"{label[cfg]}  (n={len(items)})",
                linewidth=1.8,
            )
            if len(items) > 1:
                ax.fill_between(grid, smooth(lo), smooth(hi), color=color, alpha=0.18)

        ax.set_title(env_name)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("test episode reward")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT)
    print("Saved:", OUT)

    # Report numerical summaries
    print("\n=== final-window stats (last 10% of steps) ===")
    for env_name, env_dir in ENVS.items():
        runs = load_runs(env_dir)
        for r in runs:
            tail = r["reward"][-max(1, len(r["reward"]) // 10):]
            mode = "trajectory" if r["traj"] else "rollout"
            print(
                f"{env_name:12s} mode={mode:10s} norm={str(r['norm']):5s} "
                f"final={tail.mean():8.1f} best={r['reward'].max():8.1f} "
                f"steps={r['steps'][-1]:>10d}  run={os.path.basename(r['path'])}"
            )


if __name__ == "__main__":
    plot()
