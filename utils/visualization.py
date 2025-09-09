import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_stiffeners(df, node_x="Ipan", node_y="Type"):
    """
    df : build_stiffener_groups로 만든 DataFrame
    node_x, node_y : 보강재 좌표 컬럼 (실제 stiffener geometry에 맞게 변경)
    """

    if df.empty:
        print("DataFrame is empty")
        return

    # panel-group 조합별 그룹핑
    grouped = df.groupby(["panel", "group"])
    cmap = cm.get_cmap("tab20", len(grouped))  # 색상 팔레트

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, ((panel, group), gdf) in enumerate(grouped):
        color = cmap(i)

        # 보강재 위치 플롯 (여기서는 node_x, node_y 기준으로 점 찍기)
        xs = gdf[node_x].astype(float)
        ys = gdf[node_y].astype(float)

        ax.scatter(xs, ys, label=f"Panel {panel}, Group {group}", color=color, s=50)

        # 연결선이 필요하다면
        ax.plot(xs, ys, color=color, linewidth=1, alpha=0.7)

    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Stiffener groups by panel & group")
    plt.show()


if __name__ == "__main__":
    from parser import parse_ma2
    import pandas as pd

    pd.set_option("display.max_rows", None)
    parsed = parse_ma2("../data/sample.ma2")

    print(parsed["stiffener_groups"])
    plot_stiffeners(parsed["stiffener_groups"])
