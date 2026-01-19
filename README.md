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
