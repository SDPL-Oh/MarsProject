import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    """
    간단한 2D 그리드 환경:
    - 에이전트는 (0, 0)에서 시작하여 (4, 4) 목표에 도달해야 함
    - 4개의 방향으로 움직일 수 있음
    - 목표 도달 시 보상 +1, 그렇지 않으면 -0.1
    """
    def __init__(self):
        super().__init__()
        self.grid_size = 5
        self.action_space = spaces.Discrete(4)  # 0:상, 1:하, 2:좌, 3:우
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32
        )

        self.agent_pos = None
        self.goal_pos = np.array([4, 4], dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        return self.agent_pos.copy(), {}

    def step(self, action):
        if action == 0:
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)  # Up
        elif action == 1:
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)  # Down
        elif action == 2:
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)  # Left
        elif action == 3:
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)  # Right

        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        truncated = False  # 시간 제한 조건 없음
        reward = 1.0 if terminated else -0.1

        observation = self.agent_pos.astype(np.int32).copy()
        return observation, reward, terminated, truncated, {}

    def render(self):
        grid = np.full((self.grid_size, self.grid_size), '-', dtype=str)
        grid[tuple(self.goal_pos)] = 'G'
        grid[tuple(self.agent_pos)] = 'A'
        print("\n".join(" ".join(row) for row in grid))
        print("------")


# from shimmy.gymnasium_compatibility import GymV26CompatibilityV0
from stable_baselines3 import PPO

# 환경 래핑
env = GridWorldEnv()
# wrapped_env = GymV26CompatibilityV0(env)

# 모델 생성 및 학습
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# 학습된 모델로 테스트
obs, _ = env.reset()  # ✅ 튜플 언팩
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)  # ✅ Gymnasium은 5개의 return 값
    env.render()

    if done:
        print("🎉 목표에 도달!")
        break

