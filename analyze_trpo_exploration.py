"""Analyze the completed TRPO exploration experiments and generate report figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
LOG_ROOT = ROOT / "outputs_server_latest"
OUTPUT_DIR = ROOT / "TRPO" / "TRPO实现" / "figures" / "exploration"

ENVIRONMENTS = ("HalfCheetah", "Humanoid", "Walker2d")

ADVANTAGE_RUNS = {
    ("HalfCheetah", True): LOG_ROOT
    / "HalfCheetah-v5-TRPO-rollout"
    / "2026-05-23_10-24-09",
    ("Humanoid", True): LOG_ROOT
    / "Humanoid-v5-TRPO-rollout"
    / "2026-05-23_10-24-04",
    ("Walker2d", True): LOG_ROOT
    / "Walker2d-v5-TRPO-rollout"
    / "2026-05-23_10-24-16",
    ("HalfCheetah", False): LOG_ROOT
    / "HalfCheetah-v5-TRPO-rollout-advnorm-false"
    / "2026-06-11_12-11-13",
    ("Humanoid", False): LOG_ROOT
    / "Humanoid-v5-TRPO-rollout-advnorm-false"
    / "2026-06-11_21-33-36",
    ("Walker2d", False): LOG_ROOT
    / "Walker2d-v5-TRPO-rollout-advnorm-false"
    / "2026-06-13_01-32-28",
}

TRAJECTORY_RUNS = {
    "HalfCheetah": LOG_ROOT
    / "HalfCheetah-v5-TRPO-trajectory-samples"
    / "2026-06-13_15-12-20",
    "Humanoid": LOG_ROOT
    / "Humanoid-v5-TRPO-trajectory-samples"
    / "2026-06-14_00-43-34",
    "Walker2d": LOG_ROOT
    / "Walker2d-v5-TRPO-trajectory-samples"
    / "2026-06-14_03-14-30",
}

TEST_REWARD = "test/episode_rewards_mean"
KL = "train/actor/new_kl"
SURROGATE = "train/actor/new_surrogate_target"
COLLECTED = "train/samples/collected_transitions"
EFFECTIVE = "train/samples/effective_transitions"
UTILIZATION = "train/samples/utilization"
VALUE_LOSS = "train/value/loss"
BACKTRACK = "train/actor/backtrack_step"
STEPSIZE = "train/actor/stepsize"


def load_run(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    event_files = list(run_dir.glob("events.out.tfevents*"))
    if len(event_files) != 1:
        raise RuntimeError(f"Expected one event file under {run_dir}, found {len(event_files)}")
    accumulator = EventAccumulator(str(event_files[0]))
    accumulator.Reload()
    return {
        tag: (
            np.asarray([event.step for event in accumulator.Scalars(tag)], dtype=float),
            np.asarray([event.value for event in accumulator.Scalars(tag)], dtype=float),
        )
        for tag in accumulator.Tags().get("scalars", [])
    }


def moving_average(values: np.ndarray, window: int = 10) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def save_figure(figure: plt.Figure, name: str) -> None:
    figure.savefig(OUTPUT_DIR / f"{name}.pdf")
    plt.close(figure)


def sample_metrics(
    trajectory_runs: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    rollout_runs: dict[tuple[str, bool], dict[str, tuple[np.ndarray, np.ndarray]]],
) -> list[dict[str, float | str]]:
    rows = []
    for environment in ENVIRONMENTS:
        trajectory = trajectory_runs[environment]
        trajectory_reward_steps, trajectory_rewards = trajectory[TEST_REWARD]
        rollout_reward_steps, rollout_rewards = rollout_runs[(environment, True)][TEST_REWARD]
        matched_rollout_reward = np.interp(
            trajectory_reward_steps[-1], rollout_reward_steps, rollout_rewards
        )
        rows.append(
            {
                "environment": environment,
                "trajectory_collected_mean": np.mean(trajectory[COLLECTED][1]),
                "trajectory_effective_mean": np.mean(trajectory[EFFECTIVE][1]),
                "trajectory_utilization_mean": np.mean(trajectory[UTILIZATION][1]),
                "rollout_collected": 16000.0,
                "rollout_effective": 16000.0,
                "rollout_utilization": 1.0,
                "trajectory_final_step": trajectory_reward_steps[-1],
                "trajectory_reward_at_final_step": trajectory_rewards[-1],
                "rollout_reward_at_matched_step": matched_rollout_reward,
                "rollout_final_reward": rollout_rewards[-1],
            }
        )
    return rows


def plot_advantage_rewards(
    runs: dict[tuple[str, bool], dict[str, tuple[np.ndarray, np.ndarray]]],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, environment in zip(axes, ENVIRONMENTS):
        for normalized, color in ((True, "tab:blue"), (False, "tab:orange")):
            steps, values = runs[(environment, normalized)][TEST_REWARD]
            axis.plot(
                steps / 1e6,
                values,
                color=color,
                linewidth=1.8,
                label=f"advan_norm={str(normalized).lower()}",
            )
        axis.set_title(f"{environment}-v5")
        axis.set_xlabel("Interaction steps (million)")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("Mean test reward")
    axes[-1].legend()
    save_figure(figure, "advantage_normalization_rewards")


def plot_advantage_stability(
    runs: dict[tuple[str, bool], dict[str, tuple[np.ndarray, np.ndarray]]],
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for column, environment in enumerate(ENVIRONMENTS):
        for normalized, color in ((True, "tab:blue"), (False, "tab:orange")):
            label = f"advan_norm={str(normalized).lower()}"
            kl_steps, kl_values = runs[(environment, normalized)][KL]
            axes[0, column].plot(
                kl_steps / 1e6,
                moving_average(kl_values),
                color=color,
                linewidth=1.5,
                label=label,
            )
            surrogate_steps, surrogate_values = runs[(environment, normalized)][SURROGATE]
            finite = np.isfinite(surrogate_values)
            axes[1, column].plot(
                surrogate_steps[finite] / 1e6,
                np.maximum(np.abs(surrogate_values[finite]), 1e-12),
                color=color,
                linewidth=1.2,
                label=label,
            )
        axes[0, column].axhline(0.01, color="black", linestyle="--", linewidth=1)
        axes[0, column].set_title(f"{environment}-v5")
        axes[0, column].set_xlabel("Interaction steps (million)")
        axes[0, column].grid(alpha=0.3)
        axes[1, column].set_xlabel("Interaction steps (million)")
        axes[1, column].set_yscale("log")
        axes[1, column].grid(alpha=0.3)
    axes[0, 0].set_ylabel("New KL (10-point moving average)")
    axes[1, 0].set_ylabel("|New surrogate objective| (log scale)")
    axes[0, -1].legend()
    save_figure(figure, "advantage_normalization_stability")


def plot_sample_utilization(sample_rows: list[dict[str, float | str]]) -> None:
    x = np.arange(len(ENVIRONMENTS))
    width = 0.25
    trajectory_collected = np.asarray(
        [float(row["trajectory_collected_mean"]) for row in sample_rows]
    )
    trajectory_effective = np.asarray(
        [float(row["trajectory_effective_mean"]) for row in sample_rows]
    )
    rollout_effective = np.asarray([float(row["rollout_effective"]) for row in sample_rows])
    trajectory_utilization = np.asarray(
        [float(row["trajectory_utilization_mean"]) for row in sample_rows]
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].bar(x - width, trajectory_collected, width, label="Trajectory collected")
    axes[0].bar(x, trajectory_effective, width, label="Trajectory effective")
    axes[0].bar(x + width, rollout_effective, width, label="Rollout collected/effective")
    axes[0].set_ylabel("Transitions per policy update")
    axes[0].set_xticks(x, ENVIRONMENTS)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x - width / 2, trajectory_utilization * 100, width, label="Trajectory")
    axes[1].bar(x + width / 2, np.full(len(x), 100.0), width, label="Rollout")
    axes[1].set_ylabel("Transition utilization (%)")
    axes[1].set_xticks(x, ENVIRONMENTS)
    axes[1].set_ylim(0, 110)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()
    save_figure(figure, "sample_utilization")


def plot_trajectory_rollout_rewards(
    trajectory_runs: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    rollout_runs: dict[tuple[str, bool], dict[str, tuple[np.ndarray, np.ndarray]]],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, environment in zip(axes, ENVIRONMENTS):
        trajectory_steps, trajectory_rewards = trajectory_runs[environment][TEST_REWARD]
        rollout_steps, rollout_rewards = rollout_runs[(environment, True)][TEST_REWARD]
        axis.plot(
            trajectory_steps / 1e6,
            trajectory_rewards,
            linewidth=1.8,
            label="Trajectory",
        )
        axis.plot(
            rollout_steps / 1e6,
            rollout_rewards,
            linewidth=1.8,
            label="Rollout",
        )
        axis.set_title(f"{environment}-v5")
        axis.set_xlabel("Interaction steps (million)")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("Mean test reward")
    axes[-1].legend()
    save_figure(figure, "trajectory_rollout_rewards")


def plot_training_diagnostics(
    runs: dict[tuple[str, bool], dict[str, tuple[np.ndarray, np.ndarray]]],
) -> None:
    """3 rows × 3 cols: critic value loss / backtrack steps / step size,
    comparing advan_norm true vs false on rollout runs."""
    figure, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    for column, environment in enumerate(ENVIRONMENTS):
        for normalized, color in ((True, "tab:blue"), (False, "tab:orange")):
            label = f"advan_norm={str(normalized).lower()}"
            run = runs[(environment, normalized)]

            vl_steps, vl_values = run[VALUE_LOSS]
            finite = np.isfinite(vl_values)
            axes[0, column].plot(
                vl_steps[finite] / 1e6,
                np.maximum(vl_values[finite], 1e-3),
                color=color, linewidth=1.2, label=label, alpha=0.85,
            )

            bt_steps, bt_values = run[BACKTRACK]
            axes[1, column].plot(
                bt_steps / 1e6, moving_average(bt_values, 20),
                color=color, linewidth=1.4, label=label,
            )

            ss_steps, ss_values = run[STEPSIZE]
            finite = np.isfinite(ss_values) & (ss_values > 0)
            axes[2, column].plot(
                ss_steps[finite] / 1e6,
                np.maximum(ss_values[finite], 1e-6),
                color=color, linewidth=1.2, label=label, alpha=0.85,
            )

        axes[0, column].set_title(f"{environment}-v5")
        axes[0, column].set_yscale("log")
        axes[2, column].set_yscale("log")
        for row in range(3):
            axes[row, column].set_xlabel("Interaction steps (million)")
            axes[row, column].grid(alpha=0.3)
    axes[0, 0].set_ylabel("Critic value loss (log)")
    axes[1, 0].set_ylabel("Backtrack steps (20-pt avg)")
    axes[2, 0].set_ylabel("Step size (log)")
    axes[0, -1].legend()
    save_figure(figure, "training_diagnostics")


def diagnostic_summary(
    runs: dict[tuple[str, bool], dict[str, tuple[np.ndarray, np.ndarray]]],
) -> None:
    print("\n=== Training diagnostic summary ===")
    for environment in ENVIRONMENTS:
        for normalized in (True, False):
            run = runs[(environment, normalized)]
            vl = run[VALUE_LOSS][1]
            bt = run[BACKTRACK][1]
            ss = run[STEPSIZE][1]
            vl_finite = vl[np.isfinite(vl)]
            ss_finite = ss[np.isfinite(ss) & (ss > 0)]
            print(
                f"{environment:12s} norm={str(normalized):5s} "
                f"value_loss mean={vl_finite.mean():.2f} max={vl_finite.max():.2e} "
                f"backtrack mean={bt.mean():.2f} max={int(bt.max())} "
                f"stepsize mean={ss_finite.mean():.3e}"
            )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    advantage_runs = {key: load_run(path) for key, path in ADVANTAGE_RUNS.items()}
    trajectory_runs = {key: load_run(path) for key, path in TRAJECTORY_RUNS.items()}

    sample_rows = sample_metrics(trajectory_runs, advantage_runs)

    plot_advantage_rewards(advantage_runs)
    plot_advantage_stability(advantage_runs)
    plot_sample_utilization(sample_rows)
    plot_trajectory_rollout_rewards(trajectory_runs, advantage_runs)
    plot_training_diagnostics(advantage_runs)
    diagnostic_summary(advantage_runs)
    print(f"Wrote analysis artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
