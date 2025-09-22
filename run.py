from utils.parser import parse_ma2_sections

import pandas as pd
from rl.rl_env import ShipStrakeScantEnv
from rl.rl_model import train_agent

if __name__ == "__main__":
    file_path = "data/example.txt"

    sections = parse_ma2_sections(file_path)

    # 노드 데이터프레임 변환 후 시각화 (추가 개발 필요)
    # nodes_df = parse_nodes(sections["nodes"])
    # plot_nodes(nodes_df)


    # CSV 불러오기
    csv_dict = {
        "angle": pd.read_csv("data/angle.CSV"),
        "tbar": pd.read_csv("data/tbar.CSV"),
        "strake": pd.read_csv("data/strake.CSV"),
        "flat": pd.read_csv("data/flat.CSV"),
        "bulb": pd.read_csv("data/bulb.CSV"),
    }

    # 환경 초기화
    env = ShipStrakeScantEnv(csv_dict)

    # 학습 실행
    trained_agent = train_agent(env, episodes=200)
