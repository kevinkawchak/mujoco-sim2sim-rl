import gymnasium as gym
import numpy as np
from collections import deque
from stable_baselines3 import PPO

class SensorNoiseDelayWrapper(gym.ObservationWrapper):
    def __init__(self, env, delay_steps=2, noise_std=0.05):
        super().__init__(env)
        self.noise_std = noise_std
        self.delay_steps = delay_steps
        # Buffer to simulate latency
        self.obs_buffer = deque(maxlen=delay_steps + 1)

    def observation(self, obs):
        # 1. Add Gaussian Noise
        noise = np.random.normal(0, self.noise_std, size=obs.shape)
        noisy_obs = obs + noise
        
        # 2. Simulate Delay
        self.obs_buffer.append(noisy_obs)
        if len(self.obs_buffer) < self.delay_steps + 1:
            return self.obs_buffer[0] # Return oldest available
        return self.obs_buffer[0] # Return delayed frame

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs_buffer.clear()
        self.obs_buffer.append(obs)
        return obs, info

env = gym.make('Humanoid-v4')
env = SensorNoiseDelayWrapper(env, delay_steps=1, noise_std=0.02)

model = PPO('MlpPolicy', env, verbose=1)
print("Starting Sensor DR Training...")
model.learn(total_timesteps=100_000)
model.save("models/PPO/humanoid_sensor_dr")
print("Sensor DR Model Saved.")
