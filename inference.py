import os
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor

from rl.rl_env import ScantlingOptEnv


def inference_scantling(model_path, config_path, max_steps=50):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"{model_path}")

    model = PPO.load(model_path, device="cuda")

    env = ScantlingOptEnv(config_path=config_path, max_steps=max_steps)
    env = Monitor(env)

    obs, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        print(f"[Step {step_count}] Action: {action}, Reward: {reward}")

    print("\n========== 추론 종료 ==========")
    print(f"총 스텝 수     : {step_count}")
    print(f"총 누적 리워드 : {total_reward}")

    return total_reward


if __name__ == "__main__":
    inference_scantling(
        model_path="./logs/best_model/best_model.zip",
        config_path="data/config.json",
        max_steps=50
    )
