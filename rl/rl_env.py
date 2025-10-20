import json
import pandas as pd
import numpy as np

from utils.parser import parse_ma2
from utils.mars import run_mars, evaluate_rule
from utils.processing import update_group_value


class ScantEnv:
    def __init__(self, config_path, max_steps=50):
        with open(config_path, "r") as f:
            config = json.load(f)

        input_file = config["input_path"]
        parsed = parse_ma2(input_file)

        self.df_scant = parsed["stiff scant"]
        self.df_stiff_loc = parsed["stiff loc"]
        self.config_path = config_path
        self.max_steps = max_steps
        self.current_step = 0
        self.prev_weight = self._compute_weight(self.df_scant)

    def reset(self):
        self.current_step = 0
        self.state = self.df_scant.copy()
        self.prev_weight = self._compute_weight(self.state)
        return self._get_observation()

    def _get_observation(self):
        obs_cols = ["Ipan", "group", "hweb", "tweb", "hflan", "tflan"]
        return self.state[obs_cols].to_numpy()

    def step(self, actions):
        self._apply_actions(actions)

        df_result = self._parse_and_evaluate_results()

        reward = self._compute_reward(df_result)
        done = self._check_done(df_result)

        self.current_step += 1
        obs = self._get_observation()

        return obs, reward, done, {}

    def _apply_actions(self, actions):
        for i, act in enumerate(actions):
            group_idx = act["group"]
            param = act["param"]
            delta = act["delta"]
            panel = act.get("panel", 1)
            old_val = self.df_scant.loc[self.df_scant["GroupIndex"] == group_idx, param].iloc[0]
            new_val = max(old_val + delta, 0)
            self.df_scant = update_group_value(self.df_scant, panel, 1, param, new_val)

    def _parse_and_evaluate_results(self):
        df_global, df_panel, df_stiffener = run_mars(self.config_path)
        df_eval = evaluate_rule(df_stiffener)
        return df_eval

    def _compute_weight(self, df):
        weight = (
            df["hweb"].fillna(0)
            + df["tweb"].fillna(0)
            + df["hflan"].fillna(0)
            + df["tflan"].fillna(0)
        ).sum()
        return weight

    def _compute_reward(self, df_eval):
        has_false = (df_eval["Pass"] == False).any()
        current_weight = self._compute_weight(self.df_scant)

        if has_false:
            reward = -1
            done = True
        else:
            if current_weight < self.prev_weight:
                reward = +2
            elif np.isclose(current_weight, self.prev_weight, atol=1e-3):
                reward = +1
            else:
                reward = +0.5
            done = True
        self.prev_weight = current_weight
        return reward, done