from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.rl_env import ScantEnv, GymScantEnv

config_path = '../data/config.json'
base_env = ScantEnv(config_path)
gym_env = GymScantEnv(base_env)


vec_env = DummyVecEnv([lambda: gym_env])


model = PPO(
    "MlpPolicy",          # 기본 MLP 구조
    vec_env,
    verbose=1,
    learning_rate=1e-4,
    gamma=0.95,
    n_steps=512,
    batch_size=64,
)

model.learn(total_timesteps=20000)
model.save("ppo_scantling_tbar")
