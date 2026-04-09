from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

run_dir = os.path.join(os.path.dirname(__file__),
                       "PongNoFrameskip-v4-DQN", "2026-03-31_22-49-49")
f = next(os.path.join(run_dir, x) for x in os.listdir(run_dir)
         if x.startswith("events.out"))
ea = EventAccumulator(f)
ea.Reload()
print("Tags:", ea.scalars.Keys())
