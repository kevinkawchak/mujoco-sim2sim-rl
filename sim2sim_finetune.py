import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class PhysicsPerturbWrapper(gym.Wrapper):
    """
    Sim-to-sim: modify MuJoCo physics at reset time.
    This keeps the task "the same" but changes dynamics -> requires policy adaptation.
    """
    def __init__(self, env, gravity_scale=1.2, damping_scale=1.3, actuator_scale=0.85):
        super().__init__(env)
        self.gravity_scale = gravity_scale
        self.damping_scale = damping_scale
        self.actuator_scale = actuator_scale

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Access MuJoCo model through Gymnasium env
        model = self.env.unwrapped.model

        # gravity: model.opt.gravity is a 3-vector (x,y,z)
        model.opt.gravity[:] = model.opt.gravity[:] * self.gravity_scale

        # joint damping: model.dof_damping is per-DoF
        model.dof_damping[:] = model.dof_damping[:] * self.damping_scale

        # actuator strength: model.actuator_gainprm is per-actuator, params vary by actuator type
        # We'll scale the first gain term as a simple strength proxy.
        if model.nu > 0 and model.actuator_gainprm.shape[0] == model.nu:
            model.actuator_gainprm[:, 0] = model.actuator_gainprm[:, 0] * self.actuator_scale

        return obs, info


ENV_ID = "HalfCheetah-v5"

def make_target_env():
    env = gym.make(ENV_ID)
    env = PhysicsPerturbWrapper(env, gravity_scale=1.15, damping_scale=1.25, actuator_scale=0.9)
    return env

# Vec env from callables
env = make_vec_env(make_target_env, n_envs=4, seed=123)

# Load source policy and continue training in new sim
model = PPO.load("ppo_halfcheetah_source", env=env, verbose=1)
model.learn(total_timesteps=200_000)
model.save("ppo_halfcheetah_target")

print("Saved -> ppo_halfcheetah_target.zip")
