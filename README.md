# octavus

A virtual octopus: a common octopus (*Octopus vulgaris*) simulated in MuJoCo, with a brain that learns to move and live like an octopus through reinforcement learning in PyTorch.

## Status

Restructuring. What exists now is the body in the water:

- **8 soft arms**, 16 segments each in 4 sections. Each section has 4 muscles (bend up/down, bend sideways, twist, stretch/shorten) and suckers, with a touch sensor on every segment.
- **Octopus muscle:** each muscle loses force the faster it shortens (Hill's curve, with speeds measured in octopus arms). The brain sends postures (0 = limp), and the arm's own reflex turns them into force.
- **Sea water:** buoyancy, drag and lift, slippery skin (the suckers do the gripping).
- **Siphon jet:** squirts in pulses and tires after ~5 s of full jet, like a real octopus whose heart stops while it swims.
- **Energy:** the metabolic cost in watts.
- **Real sizes:** 2.9 kg, 60 cm arms tapering to ~4 mm, a 13 cm mantle and a web, all taken from published measurements (sources in the code).

The first version, one end-to-end PPO brain, lives in the git history (up to `9ccad0e`).

## Running

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14.

```bash
uv sync

# watch the octopus flail at random (on macOS the viewer needs mjpython)
uv run mjpython world_octavus/environment.py

# rebuild octopus.xml after changing the body in build_octopus.py
uv run python world_octavus/build_octopus.py

# open just the body in the MuJoCo viewer (the path must be absolute)
uv run python -m mujoco.viewer --mjcf="$PWD/world_octavus/octopus.xml"
```
