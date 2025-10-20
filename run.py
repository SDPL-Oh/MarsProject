import pandas as pd

from utils.parser import parse_ma2
from rl.rl_env import ScantEnv
from rl.rl_model import train_agent


if __name__ == "__main__":
    file_path = "data/example.txt"

    parsed = parse_ma2("../data/change.ma2")

    csv_dict = {
        "angle": pd.read_csv("data/angle.CSV"),
        "tbar": pd.read_csv("data/tbar.CSV"),
        "strake": pd.read_csv("data/strake.CSV"),
        "flat": pd.read_csv("data/flat.CSV"),
        "bulb": pd.read_csv("data/bulb.CSV"),
    }

    env = ScantEnv(csv_dict)

    trained_agent = train_agent(env, episodes=200)
