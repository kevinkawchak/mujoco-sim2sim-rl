import gymnasium as gym
from stable_baselines3 import PPO
import time

class PhysicsPerturbWrapper(gym.Wrapper):
    def __init__(self, env, gravity_scale=1.15, damping_scale=1.25, actuator_scale=0.9):
        super().__init__(env)
        self.gravity_scale = gravity_scale
        self.damping_scale = damping_scale
        self.actuator_scale = actuator_scale

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        model = self.env.unwrapped.model
        model.opt.gravity[:] = model.opt.gravity[:] * self.gravity_scale
        model.dof_damping[:] = model.dof_damping[:] * self.damping_scale
        if model.nu > 0 and model.actuator_gainprm.shape[0] == model.nu:
            model.actuator_gainprm[:, 0] = model.actuator_gainprm[:, 0] * self.actuator_scale
        return obs, info

ENV_ID = "HalfCheetah-v5"

env = gym.make(ENV_ID, render_mode="human")
env = PhysicsPerturbWrapper(env)

model = PPO.load("ppo_halfcheetah_target")

obs, info = env.reset(seed=2)
for _ in range(3000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
    time.sleep(1/60)

env.close()
