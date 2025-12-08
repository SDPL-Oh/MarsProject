import os
import shutil
import json
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from tqdm import tqdm

from utils.parser import parse_ma2, update_stiff_scant_in_ma2
from utils.mars import run_mars, evaluate_rule, compute_margin
from utils.processing import group_stiff, update_group_value


def load_action_data(config):
    tbar_df = pd.read_csv(config["tbar"])
    return tbar_df


class ScantlingOptEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config_path, max_steps=40):
        super().__init__()

        with open(config_path, "r") as f:
            self.config = json.load(f)

        parsed = parse_ma2(self.config["input_path"])
        self.df_scant = parsed["stiff scant"]
        self.df_stiff_loc = parsed["stiff loc"]
        self.df_stiff_new = group_stiff(self.df_scant, self.df_stiff_loc)

        self.hweb_list = [350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775]
        self.tweb_list = [11, 11.5, 12, 12.25, 12.5, 13, 13.5, 14, 14.25, 14.5, 15, 15.5, 16]
        self.hflan_list = [125, 150, 175, 200]
        self.tflan_list = [round(x, 2) for x in np.arange(11.5, 29.5 + 0.001, 0.25)]

        self.action_space = spaces.MultiDiscrete([
            59,
            len(self.hweb_list),
            len(self.tweb_list),
            len(self.hflan_list),
            len(self.tflan_list)
        ])

        df_eval = self._parse_and_eval()
        margin = compute_margin(df_eval)
        obs = self._get_observation(margin)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

        self.max_steps = max_steps
        self.current_step = 0
        self.prev_weight = self._compute_weight(self.df_stiff_new)
        # self.prev_weight = None
        self.prev_fail = None
        self.selected_group = 0

    def reset(self, seed=None, options=None):
        input_path = self.config["input_path"]
        if os.path.exists(input_path):
            os.remove(input_path)

        temp_path = self.config["temp_path"]
        shutil.copy(temp_path, input_path)

        parsed = parse_ma2(input_path)
        self.df_scant = parsed["stiff scant"]
        self.df_stiff_loc = parsed["stiff loc"]
        self.df_stiff_new = group_stiff(self.df_scant, self.df_stiff_loc)
        self.current_step = 0
        self.prev_weight = self._compute_weight(self.df_stiff_new)
        self.prev_fail = 0

        df_eval = self._parse_and_eval()
        margin = compute_margin(df_eval)
        obs = self._get_observation(margin)
        return obs.astype(np.float32), {}

    def step(self, action):
        self.current_step += 1

        target_group = action[0]
        selected_spec = [
            self.hweb_list[action[1]],
            self.tweb_list[action[2]],
            self.hflan_list[action[3]],
            self.tflan_list[action[4]]
        ]

        g_bool = (self.df_stiff_new["group"] == target_group)
        g_idx = np.where(g_bool)[0]
        cols = [4, 5, 6, 7]
        for col_idx, spec in zip(cols, selected_spec):
            self.df_stiff_new.iloc[g_idx, col_idx] = spec

        update_stiff_scant_in_ma2(
            self.config["input_path"],
            self.config["input_path"],
            self.df_stiff_new.iloc[:, 2:]
        )

        df_eval_new = self._parse_and_eval()
        margin = compute_margin(df_eval_new)
        reward, terminated = self._compute_reward(margin, target_group)
        df_fail = df_eval_new[df_eval_new["pass"] == False]
        print(f'[step {self.current_step}], reward: {round(reward, 2)}, modify group: {target_group}, fail: {df_fail.shape[0]}')

        truncated = self.current_step >= self.max_steps
        obs = self._get_observation(margin)
        self.selected_group = target_group

        return obs, reward, terminated, truncated, {}

    def _parse_and_eval(self):
        _, _, df_stiffener = run_mars(self.config)
        return evaluate_rule(df_stiffener)

    def _get_observation(self, df_margin):
        df_margin = df_margin.reset_index()
        df_margin.columns = ["panel", "stiffener", "margin"]
        df_margin["stiffener"] = df_margin["stiffener"].astype(int)

        # columns: [Ipan, type, group, stiff_index, hweb, tweb, hflan, tflan]
        obs_cols = [2, 3, 0, 1, 4, 5, 6, 7]
        df_obs = self.df_stiff_new.iloc[:, obs_cols].copy()
        df_obs = df_obs[df_obs["Type"] == str(4)]
        df_obs["panel"] = df_obs["Ipan"]
        df_obs["stiffener"] = df_obs["stiff_index"]
        df_obs = df_obs.merge(df_margin, on=["panel", "stiffener"], how="left").astype(float)
        df_obs = df_obs.dropna(subset=["margin"])

        df_obs["hweb_idx"] = df_obs["hweb"].apply(lambda x: self.hweb_list.index(x))
        df_obs["tweb_idx"] = df_obs["tweb"].apply(lambda x: self.tweb_list.index(x))
        df_obs["hflan_idx"] = df_obs["hflan"].apply(lambda x: self.hflan_list.index(x))
        df_obs["tflan_idx"] = df_obs["tflan"].apply(lambda x: self.tflan_list.index(x))
        cols = ["group", "hweb_idx", "tweb_idx", "hflan_idx", "tflan_idx", "margin"]

        return df_obs[cols].to_numpy(dtype=np.float32)

    def _find_group(self, panel, stiff):
        row = self.df_stiff_new[
            (self.df_stiff_new["Ipan"] == str(panel)) &
            (self.df_stiff_new["stiff_index"] == int(stiff))
        ]
        return row["group"].iloc[0]

    def _compute_weight(self, df):
        cols = [4, 5, 6, 7]
        vals = df.iloc[:, cols].apply(pd.to_numeric, errors="coerce")
        return vals.sum(axis=1).sum()

    def _compute_reward(self, margins, group):
        curr_fail = (margins < 0).sum()
        fail_change = self.prev_fail - curr_fail
        reward = 0

        current_w = self._compute_weight(self.df_stiff_new)
        delta_w = self.prev_weight - current_w
        w_ratio = delta_w / self.prev_weight

        if curr_fail > 0:
            reward -= curr_fail
            reward += 10 * fail_change
            if self.selected_group == group:
                reward += 10
        else:
            # reward += 15
            reward += 0.01 * delta_w

        terminated = (curr_fail == 0) and (0.1 <= w_ratio)

        self.prev_fail = curr_fail
        if self.prev_weight > current_w:
            self.prev_weight = current_w

        return reward, terminated

    def render(self):
        print(f"[Step {self.current_step}] Weight={self._compute_weight(self.df_stiff_new):.2f}")