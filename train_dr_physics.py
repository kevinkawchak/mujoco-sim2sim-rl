import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

class PhysicsRandomizerWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.base_model = self.unwrapped.model
        
        # Store original values to perturb around
        self.orig_friction = self.base_model.geom_friction.copy()
        self.orig_body_mass = self.base_model.body_mass.copy()

    def reset(self, **kwargs):
        # 1. Randomize Friction (0.5x to 1.5x)
        friction_scale = np.random.uniform(0.5, 1.5)
        self.base_model.geom_friction[:] = self.orig_friction * friction_scale

        # 2. Randomize Torso/Limb Mass (0.8x to 1.2x)
        mass_scale = np.random.uniform(0.8, 1.2, size=self.base_model.body_mass.shape)
        self.base_model.body_mass[:] = self.orig_body_mass * mass_scale
        
        # 3. Randomize Timestep (Crucial for Sim-to-Sim)
        # Nominal is 0.002, we jitter it slightly
        self.base_model.opt.timestep = 0.002 * np.random.uniform(0.9, 1.1)

        return self.env.reset(**kwargs)

# Setup Environment with Wrapper
env = gym.make('Humanoid-v4', render_mode=None)
env = PhysicsRandomizerWrapper(env)

# Train
model = PPO('MlpPolicy', env, verbose=1)
print("Starting Physics DR Training...")
model.learn(total_timesteps=100_000)
model.save("models/PPO/humanoid_physics_dr")
print("Physics DR Model Saved.")
