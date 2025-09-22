import gym
from gym import spaces
import numpy as np


class ShipStrakeScantEnv(gym.Env):
    def __init__(self, csv_dict):
        super().__init__()
        self.csv_dict = csv_dict

        # 전체 action 후보: 각 CSV 행을 하나의 action으로 정의
        self.actions = []
        for name, df in csv_dict.items():
            for idx, row in df.iterrows():
                self.actions.append((name, idx))  # (종류, index)

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        self.reset()

    def _extract_state(self):
        try:
            avg_thickness = self.csv_dict["strake"]["Thk"].astype(float).mean()
        except:
            avg_thickness = 0.0
        try:
            avg_width = self.csv_dict["strake"]["Width"].astype(float).mean()
        except:
            avg_width = 0.0
        try:
            avg_spacing = self.csv_dict["tbar"]["Spacing"].astype(float).mean()
        except:
            avg_spacing = 0.0
        try:
            avg_web = self.csv_dict["tbar"]["Web"].astype(float).mean()
        except:
            avg_web = 0.0
        try:
            avg_flange = self.csv_dict["tbar"]["Flange"].astype(float).mean()
        except:
            avg_flange = 0.0

        return np.array([avg_thickness, avg_width, avg_spacing, avg_web, avg_flange], dtype=np.float32)

    def step(self, action_idx):
        # action 선택
        comp_name, row_idx = self.actions[action_idx]
        row = self.csv_dict[comp_name].iloc[row_idx]

        # 예시: strake 두께를 선택된 값으로 변경
        if comp_name == "strake" and "Thk" in row:
            self.csv_dict["strake"]["Thk"] = float(row["Thk"])

        # 예시: tbar spacing 적용
        if comp_name == "tbar" and "Spacing" in row:
            self.csv_dict["tbar"]["Spacing"] = float(row["Spacing"])

        # 보상 계산
        reward = self._evaluate_design()
        self.state = self._extract_state()
        done = False
        return self.state, reward, done, {}

    def _evaluate_design(self):
        """
        보상 = -총 무게
        무게는 각 부재별 치수를 단순화해서 근사 계산
        """
        total_weight = 0.0

        # Strake (판재)
        try:
            total_weight += (self.csv_dict["strake"]["Thk"].astype(float) *
                             self.csv_dict["strake"]["Width"].astype(float)).sum()
        except:
            pass

        # T-bar
        try:
            total_weight += (self.csv_dict["tbar"]["Web"].astype(float) *
                             self.csv_dict["tbar"]["Spacing"].astype(float)).sum()
            total_weight += (self.csv_dict["tbar"]["Flange"].astype(float) *
                             self.csv_dict["tbar"]["Spacing"].astype(float)).sum()
        except:
            pass

        # Flat
        try:
            total_weight += (self.csv_dict["flat"]["Thk"].astype(float) *
                             self.csv_dict["flat"]["Width"].astype(float)).sum()
        except:
            pass

        # Bulb
        try:
            total_weight += (self.csv_dict["bulb"].select_dtypes(include=[np.number])).sum().sum()
        except:
            pass

        # Angle
        try:
            total_weight += (self.csv_dict["angle"].select_dtypes(include=[np.number])).sum().sum()
        except:
            pass

        # 보상은 무게가 작을수록 큼
        reward = -total_weight
        return reward

    def reset(self):
        self.state = self._extract_state()
        return self.state

