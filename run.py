import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from rl.rl_env import ScantlingOptEnv


def train_scantling_env(config_path):
    env = ScantlingOptEnv(config_path=config_path, max_steps=20)
    env = Monitor(env)

    if os.path.exists("./logs/best_model/best_model.zip"):
        print("Load model")
        model = PPO.load("./logs/best_model/best_model.zip", device="cuda")
        model.set_env(env)
    else:
        print("New model")
        model = PPO(
            "MlpPolicy",
            env,
            device="cuda",
            verbose=1,
        )

    eval_callback = EvalCallback(
        env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/eval/",
        eval_freq=20,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=20,
        save_path="./logs/checkpoints/",
        name_prefix="sac_scantling"
    )

    model.learn(
        total_timesteps=200000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    model.save("scantling_sac_final")

    return model


if __name__ == "__main__":
    train_scantling_env('data/config.json')