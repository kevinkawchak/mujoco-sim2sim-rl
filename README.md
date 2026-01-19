# MuJoCo Sim-to-Sim RL

This project trains a Humanoid agent using Proximal Policy Optimization (PPO) in MuJoCo. It implements Domain Randomization to facilitate Sim-to-Sim transfer.

## Files
- `train_baseline.py`: Standard training on Humanoid-v4.
- `train_dr_physics.py`: Training with randomized friction, mass, and timestep.
- `train_dr_sensors.py`: Training with sensor noise and delays.
- `train_dr_terrain.py`: Training on randomized slopes.
- `eval_sim_transfer.py`: Zero-shot evaluation of MuJoCo policy in PyBullet.

## Usage
1. Install dependencies: `pip install "gymnasium[mujoco]" stable-baselines3 pybullet`
2. Run playback: `python play_mujoco.py humanoid_physics_dr`

## Videos
- `humanoid_baseline.mp4`: Standard training on Humanoid-v4 [Video](https://drive.google.com/drive/folders/1etwV8i9aJuS9NeHBnQIZQjk_-ldd6ZRC?usp=sharing).
- `humanoid_physics_dr.mp4`: Training with randomized friction, mass, and timestep.
- `humanoid_sensor_dr.mp4`: Training with sensor noise and delays.
- `humanoid_terrain_dr.mp4`: Training on randomized slopes.

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18304513-blue)](https://doi.org/10.5281/zenodo.18304513)
