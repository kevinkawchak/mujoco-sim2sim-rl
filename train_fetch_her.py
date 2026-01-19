import gymnasium as gym
import gymnasium_robotics  # registers envs
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

ENV_ID = "FetchReach-v3"

env = gym.make(ENV_ID)

model = SAC(
    policy="MultiInputPolicy",  # required for dict observations
    env=env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ),
    verbose=1,
    buffer_size=200_000,
    learning_starts=10_000,
)

model.learn(total_timesteps=200_000)
model.save("sac_her_fetchreach")
print("Saved -> sac_her_fetchreach.zip")
