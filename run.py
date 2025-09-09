from utils.parser import parse_ma2_sections, parse_panels
from utils.visualization import plot_nodes
import pandas as pd

if __name__ == "__main__":
    file_path = "data/example.txt"

    # 섹션별 파싱
    sections = parse_ma2_sections(file_path)

    # 패널 정보 DataFrame 변환
    panels_df = parse_panels(sections["panels"])
    print(panels_df.head())

    # 노드 데이터프레임 변환 후 시각화 (추가 개발 필요)
    # nodes_df = parse_nodes(sections["nodes"])
    # plot_nodes(nodes_df)
