import gymnasium as gym
from stable_baselines3 import PPO
import os

# Create directories for saving models
models_dir = "models/PPO"
log_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# 1. Load the Standard Humanoid Environment (Source Domain)
env = gym.make('Humanoid-v4', render_mode=None)

# 2. Define the Policy (PPO)
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

# 3. Train
print("Starting Baseline Training...")
# We use 100,000 steps for a quick test. Real walking usually takes >1M steps.
model.learn(total_timesteps=100_000) 

# 4. Save
model.save(f"{models_dir}/humanoid_baseline")
print("Baseline Saved.")
env.close()
