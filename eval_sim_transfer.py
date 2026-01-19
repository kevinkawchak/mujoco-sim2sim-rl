import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import PPO
import os

# 1. Register PyBullet environments with Gymnasium
# Note: PyBullet standard envs use the old Gym API. 
# We use a wrapper or try to load it directly if installed.
try:
    import pybullet_envs
    # PyBullet often registers envs like 'HumanoidBulletEnv-v0'
except ImportError:
    print("PyBullet not installed. Run: pip install pybullet")
    exit()

model_path = "models/PPO/humanoid_physics_dr.zip"

if not os.path.exists(model_path):
    print(f"Model not found at {model_path}. Please ensure Stage 3 ran successfully.")
    exit()

print(f"Loading MuJoCo-trained model from {model_path}...")
model = PPO.load(model_path)

print("Setting up PyBullet Target Environment...")
# We use 'HumanoidBulletEnv-v0'. 
# Note: Sim-to-Sim usually fails without an 'Adapter' layer because 
# MuJoCo and Bullet define joint orders differently.
# We run this to demonstrate the TRANSFER attempt.
try:
    # We use apply_api_compatibility=True to handle old Gym vs Gymnasium differences
    env_target = gym.make("HumanoidBulletEnv-v0", render_mode="human", apply_api_compatibility=True)
except Exception as e:
    print(f"\nError loading PyBullet Env: {e}")
    print("NOTE: PyBullet support in Gymnasium is experimental.")
    print("If this fails, it confirms the environment backend gap.")
    exit()

obs, _ = env_target.reset()
print("\n--- STARTING SIM-TO-SIM EVALUATION ---")
print("Watch the simulation window.")
print("The robot may fall immediately due to API/Physics mismatch (Expected in Zero-Shot).")
print("Press Ctrl+C in terminal to stop.")

try:
    total_reward = 0
    # Run for 1000 steps
    for i in range(1000):
        # Predict action using MuJoCo policy
        action, _ = model.predict(obs, deterministic=True)
        
        # Step the PyBullet environment
        obs, reward, terminated, truncated, info = env_target.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished at step {i}. Reward: {total_reward:.2f}")
            obs, _ = env_target.reset()
            total_reward = 0
            
except KeyboardInterrupt:
    print("\nStopped by user.")

env_target.close()
