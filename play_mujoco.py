import gymnasium as gym
from stable_baselines3 import PPO
import sys
import os

# Default to baseline if no argument provided
model_name = sys.argv[1] if len(sys.argv) > 1 else "humanoid_baseline"
model_path = f"models/PPO/{model_name}.zip"

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    print("Available models:")
    os.system("ls models/PPO")
    exit()

print(f"Loading {model_name}...")
# Render mode 'human' pops up the window
env = gym.make('Humanoid-v4', render_mode="human")
model = PPO.load(model_path)

obs, _ = env.reset()
print("Press Ctrl+C to stop playback.")

try:
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
except KeyboardInterrupt:
    print("\nClosing...")
    env.close()
