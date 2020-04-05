from pathlib import Path

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.trpo_mpi import TRPO

from environment import StudentEnv
from reporting import setup_logging


env = StudentEnv(subjects_number=6)
if Path('ppo2.zip').exists():
    model = PPO2.load('ppo2')
else:
    model = TRPO(MlpPolicy, env, verbose=1, gamma=0.8, timesteps_per_batch=2048)
    model.learn(total_timesteps=500000)
    model.save('ppo2_4')

setup_logging('application.log')

obs = env.reset()
for i in range(20000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    print(i)
    env.render()
    if done:
        break

# def read_logs(path):
#     df = pd.read_json(path, lines=True)
#     subjects = pd.DataFrame(df.skills.tolist(),
#                             columns=[f'subject_{i+1}' for i in range(len(df.skills[0]))])
#     return pd.concat((df, subjects), axis=1).drop(columns=['skills'])