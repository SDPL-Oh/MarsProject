import re
import pandas as pd
from collections import defaultdict


def parse_ma2_sections(file_path: str):
    sections = defaultdict(list)
    current_section = None
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            match = re.match(r"-+\s*([\w\s]+)\s*-+", line)
            if match:
                current_section = match.group(1).strip().lower()
                continue
            if current_section:
                sections[current_section].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def parse_table(section_text: str) -> pd.DataFrame:
    lines = [l for l in section_text.splitlines() if l.strip()]
    if not lines:
        return pd.DataFrame()

    header_candidates = [l for l in lines if l.startswith("*")]
    if not header_candidates:
        return pd.DataFrame()

    header_line = None
    for cand in header_candidates:
        if "Name" in cand and "Bending" in cand:
            header_line = cand
            break

    if header_line is None:
        header_line = max(header_candidates, key=lambda x: len(re.split(r"\s+", x.strip())))

    header = re.split(r"\s+", header_line.replace("*", "").strip())

    data_lines = [l for l in lines if not l.startswith("*")]

    rows = []
    for l in data_lines:
        row = re.split(r"\s+", l.strip())
        if len(row) < len(header):
            row += [None] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[:len(header)]
        rows.append(row)

    return pd.DataFrame(rows, columns=header)


def parse_key_values(section_text: str) -> dict:
    lines = [l.strip() for l in section_text.splitlines() if l.strip()]
    result = {}
    item_counter = 1
    for l in lines:
        if l.startswith("*"): l = l.replace("*", "").strip()
        if ":" in l:
            key, val = l.split(":",1)
            result[key.strip().lower().replace(" ","_")] = val.strip()
        else:
            result[f"item_{item_counter}"] = l
            item_counter += 1
    return result


def parse_version(text: str) -> dict:
    return parse_key_values(text)


def parse_bsd(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith("*")]
    data = {}
    try:
        data["version"] = lines[0]
        data["ship_name"] = lines[1]
        data["yard"] = lines[2]
        data["project"] = lines[3]
        data["ship_type"] = lines[4]
        data["section"] = lines[5]
        data["rule_set_1"] = lines[6].split()
        data["rule_set_2"] = lines[7].split()
        data["rule_set_3"] = lines[8].split()
        data["principal_dimensions"] = lines[9].split()
        data["coefficients"] = lines[10].split()
        data["yield_strengths"] = lines[20].split()
        data["elastic_modulus"] = lines[21].split()
        data["section_moments"] = lines[26].split()
    except IndexError:
        pass
    return data


def parse_compartments(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    compartments = {}
    current = None
    stage = 0

    for l in lines:
        if l == "*":
            if current:
                compartments[current["name"]] = current
                current = None
                stage = 0
            continue

        if stage == 0 and not l[0].isdigit():
            current = {"name": l}
            stage = 1
            continue

        if current is None:
            continue

        if stage == 1:
            current["nodes_panels"] = l.split()
            stage = 2
            continue

        if stage == 2:
            current["type_options"] = l.split()
            stage = 3
            continue

        if stage == 3:
            current["vcog"] = l.split()
            stage = 4
            continue

        if stage == 4:
            current["load_type"] = l.split()
            stage = 5
            continue

        if stage == 5:
            current["load_values"] = l.split()
            stage = 6
            continue

        if stage == 6:
            current["filling_levels"] = l.split()
            stage = 7
            continue

        if stage == 7:
            current["end_flags"] = l.split()
            stage = 8
            continue

        if stage >= 8:
            key = f"extra_{stage}"
            current[key] = l.split()
            stage += 1

    if current:
        compartments[current["name"]] = current

    return compartments


def parse_main(text: str) -> dict:
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith("*")]
    return {"entries": lines}


SECTION_PARSERS = {
    "version": parse_version,
    "bsd": parse_bsd,
    "main": parse_main,
    "panels": parse_table,
    "nodes": parse_table,
    "strakes": parse_table,
    "stiff loc": parse_table,
    "stiff scant": parse_table,
    "stiff scant bis": parse_table,
    "special span": parse_table,
    "deck load": parse_table,
    "compartments": parse_compartments,
    "fatigue": parse_table,
}


def build_stiffener_groups(panels_df: pd.DataFrame, stiff_loc_df: pd.DataFrame, scant_df: pd.DataFrame) -> pd.DataFrame:
    groups = []

    if panels_df.empty or stiff_loc_df.empty or scant_df.empty:
        return pd.DataFrame(columns=["panel", "group", "stiffeners"])

    n_group = 0
    n_stiff = 0
    for i, prow in panels_df.iterrows():
        panel_id = i
        Nstigr = int(prow.get("Nstigr", 0))

        for group_id in range(Nstigr):
            Nstiff = int(stiff_loc_df["Nstiff"].iloc[group_id])

            for stiff_id in range(Nstiff):
                idx = n_stiff + stiff_id
                if idx >= len(scant_df):
                    break
                features = scant_df.iloc[idx].to_dict()
                row = {"panel": panel_id, "group": n_group + group_id}
                row.update(features)
                groups.append(row)

            n_stiff += Nstiff
        n_group += Nstigr

    return pd.DataFrame(groups)


def parse_ma2(file_path: str) -> dict:
    sections = parse_ma2_sections(file_path)
    parsed = {}

    for sec_name, content in sections.items():
        func = SECTION_PARSERS.get(sec_name, parse_key_values)
        parsed[sec_name] = func(content)

    if "panels" in parsed and "stiff loc" in parsed and "stiff scant" in parsed:
        panel_df = parsed["panels"] if isinstance(parsed["panels"], pd.DataFrame) else pd.DataFrame()
        stiff_df = parsed["stiff loc"] if isinstance(parsed["stiff loc"], pd.DataFrame) else pd.DataFrame()
        scant_df = parsed["stiff scant"] if isinstance(parsed["stiff scant"], pd.DataFrame) else pd.DataFrame()
        parsed["stiffener_groups"] = build_stiffener_groups(panel_df, stiff_df, scant_df)

    return parsed


def update_stiff_scant_in_ma2(input_ma2: str, output_ma2: str, new_df: pd.DataFrame):
    with open(input_ma2, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    pattern = r"(------------------ STIFF SCANT\s+------------------)(.*?)(?=------------------|\Z)"
    match = re.search(pattern, text, flags=re.S)

    if not match:
        raise ValueError("STIFF SCANT section not found in file")

    header_line = "* " + "\t".join(new_df.columns)
    body_lines = []

    for _, row in new_df.iterrows():
        values = [str(v) for v in row.values if v not in [None, ""]]
        formatted = " " + " \t ".join(values) + "  "
        formatted = formatted.replace(" -", "-")
        body_lines.append(formatted)
    body_lines.append("*\n")

    new_section = match.group(1) + "\n" + header_line + "\n" + "\n".join(body_lines)

    new_text = text[:match.start()] + new_section + text[match.end():]

    with open(output_ma2, "w", encoding="utf-8") as f:
        f.write(new_text)

    print(f"[INFO] Updated STIFF SCANT section written to {output_ma2}")



if __name__ == "__main__":
    parsed = parse_ma2("../data/sample.ma2")

    # print("=== VERSION ===")
    # print(parsed["version"])
    # print("=== BSD ===")
    # print(parsed["bsd"])
    print("=== stiff loc ===")
    print(parsed["stiff loc"])
    # print("=== NODES ===")
    # print(parsed["nodes"].head())
    # print("=== strakes ===")
    # print(parsed["strakes"].head())
    print("=== stiff scant ===")
    print(parsed["stiff scant"])

    # df = parsed["stiff scant"]
    # df.iat[0, 2] = 600
    #
    # update_stiff_scant_in_ma2("../data/sample.ma2", "../data/change.ma2", df)

    # print(parsed["stiffener_groups"].head(15))

    # with open("../data/sample.txt", "r", encoding="utf-8") as f:
    #     text = f.read()
    # parsed_blocks = parse_blocks(text)
    #
    # print("=== Panel/Strake ===")
    # print(parsed_blocks["panel_strake"].head())
    #
    # print("\n=== Panel/Stiffener ===")
    # print(parsed_blocks["panel_stiffener"].head())