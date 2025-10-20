import os
import re
import json
import subprocess

import pandas as pd

stark_condition_map = {
    "Gross Thick.": "ge",
    "Sig. Nor.": "ge",
    "Tau. Nor.": "ge",
    "Sig. Comb.": "ge",
    "Tau. Comb.": "le",
    "Net Load Thick.": "le",
    "Net Mini Thick.": "le",
    "Eta Buck.": "le"
}

stiff_condition_map = {
    "Gross W.": "ge",
    "Gross Mini Thick.": "ge",
    "Sig. Nor.": "le",
    # "Sig. Comb.": "le",
    "Net Load W.": "ge",
    "Net Load Ash.": "ge",
    "Net Test W.": "ge",
    "Net Test Ash.": "ge",
    "Net Mini Thick.": "ge",
    "Net Mini Tflange.": "ge",
    "Net Mini BFlange": "ge",
    "Eta Bu. Over.": "le",
    "Eta Bu. Sti.": "le",
    # "Fatigue Life": "ge",
}


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

    block_pattern = re.compile(r'(Panel:\s+\d+\s+(?:Strake|Stiffener):\s+\d+)(.*?)(?=Panel:|\Z)', re.S)
    panel_rows = []
    stiffener_rows = []

    for header, body in block_pattern.findall(text):
        panel_num = re.search(r'Panel:\s+(\d+)', header).group(1)
        is_stiffener = "Stiffener" in header

        if is_stiffener:
            stiffener_num = re.search(r'Stiffener:\s+(\d+)', header).group(1)
        else:
            strake_num = re.search(r'Strake:\s+(\d+)', header).group(1)

        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("Actual"):
                continue

            match = re.match(r'(.+?:)\s+([\d.E+-]+)\s+([\d.E+-]+)?(?:\s+(.+))?', line)
            if not match:
                continue

            item = match.group(1).strip()
            actual = match.group(2)
            rule = match.group(3)

            if is_stiffener:
                stiffener_rows.append([panel_num, stiffener_num, item.replace(":", ""), actual, rule])
            else:
                panel_rows.append([panel_num, strake_num, item.replace(":", ""), actual, rule])

    df_panel = pd.DataFrame(panel_rows, columns=["panel", "Strake", "item", "actual", "rule"])
    df_stiffener = pd.DataFrame(stiffener_rows, columns=["panel", "stiffener", "item", "actual", "rule"])

    return df_global, df_panel, df_stiffener


def evaluate_rule(df, mode="stiff"):
    if mode == "stiff":
        condition_map = stiff_condition_map
    else:
        condition_map = stark_condition_map

    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df["rule"]   = pd.to_numeric(df["rule"], errors="coerce")

    results = []
    for _, row in df.iterrows():
        item = row["item"].strip().rstrip(":")
        actual, rule = row["actual"], row["rule"]

        if pd.isna(actual) or pd.isna(rule):
            results.append(True)
            continue

        if item not in condition_map:
            results.append(True)
            continue

        cond = condition_map[item]
        if cond == "ge":
            results.append(actual >= rule)
        elif cond == "le":
            results.append(actual <= rule)
        else:
            results.append(True)

    df["pass"] = results
    return df


if __name__ == "__main__":
    bat_file = "../data/config.json"
    df_global, df_panel, df_stiffener = run_mars(bat_file)
    # print(df_panel)
    # print(df_stiffener)

    df_result = evaluate_rule(df_stiffener)
    df_fail = df_result[df_result["pass"] == False]

    print(df_fail[["panel", "stiffener", "item", "actual", "rule", "pass"]])
