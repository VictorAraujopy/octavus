"""
Watch the trained octopus: loads octavus.pt and lets the brain drive the body in the viewer, the way training sees
it: every move is a draw (what the brain believes in most plus its random tries).
The red ball shows the episode's direction (it stays 1 m ahead of the octopus that way).
The brain is reloaded at the start of every episode, so you can keep this open while training runs
and see it improve without restarting.

Run (on macOS the viewer needs mjpython):  uv run mjpython world_octavus/watch_octopus.py
"""

from pathlib import Path

import numpy as np
import torch

from brain_octavus.brain import Octavus_arms_brain
from world_octavus.environment import OctopusEnv

CHECKPOINT = Path(__file__).resolve().parents[1] / "octavus.pt"


def load_latest(brain):
    try:
        brain.load_state_dict(torch.load(CHECKPOINT))
        return True
    except RuntimeError as error:
        if "size mismatch" in str(error):
            raise SystemExit(f"{CHECKPOINT.name} was trained on a different body: train again with the current one")
        return False  # caught while the trainer was writing it: keep the previous weights
    except (FileNotFoundError, EOFError):
        return False


if not CHECKPOINT.exists():
    raise SystemExit(f"no {CHECKPOINT.name} yet: run the training first (uv run brain_octavus/train.py)")

env = OctopusEnv(render_mode="human")
brain = Octavus_arms_brain()
load_latest(brain)
x, _ = env.reset()
speeds = []

while env.viewer is None or env.viewer.is_running():
    with torch.no_grad():
        draw_y, _ = brain.act(torch.tensor(x, dtype=torch.float32))  # a draw, like every move in training
    x, _, terminated, truncated, info = env.step(draw_y.numpy())
    speeds.append(info["speed"])
    if terminated or truncated:
        print(f"{np.mean(speeds):+.3f} m/s in the direction, {sum(speeds) * env.dt:+.2f} m in {len(speeds) * env.dt:.0f} s"
              f"{', flipped over' if info['flipped'] else ''}")
        load_latest(brain)
        x, _ = env.reset()
        speeds = []
env.close()
