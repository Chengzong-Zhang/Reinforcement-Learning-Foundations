from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

run_dir = r'c:/coding/py/RL/DQN/DQN in Atari/outputs/PongNoFrameskip-v4-DQN-nstep5/2026-04-03_11-32-13'
fpath = next(
    os.path.join(run_dir, f)
    for f in sorted(os.listdir(run_dir))
    if f.startswith("events.out.tfevents")
)
ea = EventAccumulator(fpath)
ea.Reload()
print("Tags:", ea.Tags()['scalars'])

for tag in ["test/episode_rewards_mean", "network/loss"]:
    items = ea.scalars.Items(tag)
    print(f"\n{tag}: count={len(items)}")
    print(f"  first: step={items[0].step}, val={items[0].value:.4f}")
    print(f"  last:  step={items[-1].step}, val={items[-1].value:.4f}")
