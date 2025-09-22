import os
import re
import json
import subprocess

import pandas as pd


def run_mars(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    batch_file = config["batch_path"]
    output_file = config["output_path"]

    process = subprocess.run(batch_file, shell=True, capture_output=True, text=True)

    if process.returncode != 0:
        print("[MARS ERROR] 실행 실패")
        print(process.stderr)
        return False

    # 결과 파일 확인
    if not os.path.exists(output_file):
        print("[MARS ERROR] 결과 파일 없음:", output_file)
        return False

    return parse_output_file(output_file)


def parse_output_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    kv_pattern = re.compile(r'([A-Za-z0-9/ ,\.]+):\s+([-0-9.E\s]+)')
    kv_matches = kv_pattern.findall(text)

    rows = []
    for k, v in kv_matches:
        nums = re.findall(r'-?\d+\.\d+(?:E[+-]?\d+)?', v)
        rows.append([k.strip()] + nums)

    max_len = max(len(r) for r in rows)
    columns = ["Key"] + [f"Value{i}" for i in range(1, max_len)]
    df_global = pd.DataFrame(rows, columns=columns)

    panel_pattern = re.compile(r'(Panel:\s+\d+\s+Strake:\s+\d+)(.*?)(?=Panel:|\Z)', re.S)
    panel_rows = []
    for block in panel_pattern.findall(text):
        header, body = block
        panel_num = re.search(r'Panel:\s+(\d+)', header).group(1)
        strake_num = re.search(r'Strake:\s+(\d+)', header).group(1)

        for line in body.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                key = parts[0].replace(":", "")
                actual = parts[1]
                rule = parts[2] if len(parts) > 2 else None
                case = parts[3] if len(parts) > 3 else None
                panel_rows.append([panel_num, strake_num, key, actual, rule, case])

    df_panel = pd.DataFrame(panel_rows, columns=["Panel", "Strake", "Item", "Actual", "Rule", "Case"])

    return df_global, df_panel


if __name__ == "__main__":

    bat_file = "../data/config.json"
    df_global, df_panel = run_mars(bat_file)
    print(df_panel)