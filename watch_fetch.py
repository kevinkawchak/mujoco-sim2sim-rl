import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
import time

ENV_ID = "FetchReach-v3"

env = gym.make(ENV_ID, render_mode="human")
model = SAC.load("sac_her_fetchreach")

obs, info = env.reset(seed=0)
for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
    time.sleep(1/60)

env.close()
