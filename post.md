## [LinkedIn 19Jan26 Post](https://www.linkedin.com/posts/kevin-kawchak-38b52a4a_robust-policy-transfer-via-domain-randomization-activity-7419101862227755008-DLdl)

Robust Policy Transfer via Domain Randomization: A Sim-to-Sim Reinforcement Learning Study using MuJoCo and a Humanoid Agent. 

Proximal policy optimization (PPO) agents were applied across four distinct dynamic randomizations with the goal of generating forward momentum.
-Humanoid Control: Standard MuJoCo Humanoid-v4 physics (fixed friction, gravity, mass).
-Humanoid Physics: Randomized friction (0.5x–1.5x), body mass (torso/limbs 0.8x–1.2x), simulation timestep jitter.
-Humanoid Sensor: Injection of Gaussian noise into observations and artificial latency (frame delays).
-Humanoid Terrain: Randomization of the ground plane orientation (tilts and slopes).

Reinforcement learning state-action-reward: AI must learn the precise coordination of torques to maintain balance.
1. State Space: Proprioceptive data: joint positions (angles), joint velocities, center-of-mass inertia, and actuator forces. 
2. Action Space: The policy outputs continuous motor torques applied to the humanoid’s joints (e.g., hips, knees, ankles). 
3. Reward Function: The shaping function encourages locomotion while penalizing energy waste. 

TensorBoard Results:
Episode Length: Start: ~21 steps per episode, End: ~62 steps per episode. (Agent over doubled survival time)
Mean Reward: Start: ~61, End: ~306. (Agents achieved forward motion by over 5x)
 
1. Humanoid Control: The Baseline agent achieved the highest reward in generating forward momentum, but failed zero-shot transfer tests. 
2. Humanoid Physics: The Physics agent stood the longest (6 seconds total), first falling back then falling forward (like falling on ice). 
3. Humanoid Sensor: This agent exhibited "stiff" control behavior, apparent in its recorded video.

This study serves as a foundational proof-of-concept for the Sim-to-Real pipeline (operating real robots), demonstrating that "imperfect" training simulations are, paradoxically, the key to perfect real-world performance. Videos are available through the GitHub repository.

kevinkawchak. “GitHub - Kevinkawchak/Mujoco-Sim2sim-Rl: Sim-To-Sim Reinforcement Learning with MuJoCo and Stable-Baselines3.” GitHub, 19 Jan. 2026, https://lnkd.in/g5_Yr5jd.
