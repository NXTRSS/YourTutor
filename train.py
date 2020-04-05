from pathlib import Path

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, TRPO


from environment import StudentEnv
from reporting import setup_logging

USE_RANDOM = False

env = StudentEnv(subjects_number=2)
# if Path('ppo2.zip').exists():
#     model = PPO2.load('ppo2')
if Path('ppo2_skills_estimation.zip').exists():
    model = PPO2.load('ppo2_skills_estimation')
else:
    model = PPO2(MlpPolicy, env, verbose=1, gamma=0.9)
    model.learn(total_timesteps=250000)
    model.save('ppo2_new')

if USE_RANDOM:
    setup_logging('application_random.log')
else:
    setup_logging('application_trpo.log')

for ep in range(5):
    obs = env.reset()
    for i in range(20000):
        if USE_RANDOM:
            action = env.action_space.sample()
        else:
            action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(i)
        env.render()
        if done:
            break
