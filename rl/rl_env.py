import gym
from gym import spaces
import numpy as np

class ShipStrakeScantEnv(gym.Env):
    def __init__(self, strakes_df, scant_df):
        super().__init__()
        self.strakes = strakes_df.copy()
        self.scant = scant_df.copy()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        self.reset()

    def _extract_state(self):
        try:
            avg_thickness = self.strakes["Thk"].astype(float).mean()
        except:
            avg_thickness = 0.0
        try:
            avg_width = self.strakes["Width"].astype(float).mean()
        except:
            avg_width = 0.0

        # scant features
        try:
            avg_spacing = self.scant["Spacing"].astype(float).mean()
        except:
            avg_spacing = 0.0
        try:
            avg_web = self.scant["Web"].astype(float).mean()
        except:
            avg_web = 0.0
        try:
            avg_flange = self.scant["Flange"].astype(float).mean()
        except:
            avg_flange = 0.0

        return np.array([avg_thickness, avg_width, avg_spacing, avg_web, avg_flange], dtype=np.float32)

    def step(self, action):
        delta_thickness, delta_spacing = action

        # strake 두께 조정
        if "Thk" in self.strakes.columns:
            self.strakes["Thk"] = self.strakes["Thk"].astype(float) + delta_thickness

        # scant spacing 조정
        if "Spacing" in self.scant.columns:
            self.scant["Spacing"] = self.scant["Spacing"].astype(float) + delta_spacing

        # 보상 계산
        reward = self._evaluate_design()

        self.state = self._extract_state()
        done = False
        return self.state, reward, done, {}

    def _evaluate_design(self):
        reward = 0.0
        try:
            avg_thickness = self.strakes["Thk"].astype(float).mean()
            reward -= avg_thickness  # 두께 줄일수록 보상
        except:
            pass
        try:
            avg_spacing = self.scant["Spacing"].astype(float).mean()
            reward += avg_spacing * 0.1  # 간격 늘릴수록 보상
        except:
            pass

        # Rule 위반 시 페널티 (예: 두께 < 0, spacing < 0)
        if (avg_thickness < 0) or (avg_spacing < 0):
            reward -= 100.0

        return reward

    def reset(self):
        self.state = self._extract_state()
        return self.state

