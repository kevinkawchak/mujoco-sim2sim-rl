import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

class TerrainRandomizer(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.model = self.unwrapped.model

    def reset(self, **kwargs):
        # Fix: Select index first, then get the axis vector
        axes_options = [[1,0,0], [0,1,0]]
        idx = np.random.randint(0, len(axes_options))
        axis = axes_options[idx]
        
        # Tilt angle: approx +/- 3 degrees
        tilt_angle = np.random.uniform(-0.05, 0.05) 
        
        # Calculate Quaternion from Axis-Angle
        c = np.cos(tilt_angle/2)
        s = np.sin(tilt_angle/2)
        
        # Apply tilt to floor (Geom 0 is usually the floor in humanoid.xml)
        self.model.geom_quat[0] = [c, s*axis[0], s*axis[1], 0]

        return self.env.reset(**kwargs)

env = gym.make('Humanoid-v4', render_mode=None)
env = TerrainRandomizer(env)
model = PPO('MlpPolicy', env, verbose=1)

print("Starting Terrain (Slope) Training...")
model.learn(total_timesteps=100_000)
model.save("models/PPO/humanoid_terrain_dr")
print("Terrain DR Model Saved.")
