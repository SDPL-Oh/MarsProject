def group_stiff(df_scant, df_stiff_loc):
    group_indices = []
    stiff_indices = []
    global_group_idx = 1

    df_stiff_loc["Panel"] = df_stiff_loc["Panel"].astype(int)
    for panel_num, panel_df in df_stiff_loc.sort_values("Panel").groupby("Panel"):

        panel_stiff_indices = []
        stiff_counter = 1
        for group_idx, (_, row) in enumerate(panel_df.iterrows()):
            try:
                nstiff = int(row["Nstiff"])

            except ValueError:
                nstiff = 0
            group_indices.extend([global_group_idx] * nstiff)
            global_group_idx += 1
            panel_stiff_indices.extend(list(range(stiff_counter, stiff_counter + nstiff)))
            stiff_counter += nstiff

        stiff_indices.extend(panel_stiff_indices)

    min_len = min(len(df_scant), len(group_indices))
    df_scant = df_scant.iloc[:min_len].copy()

    df_scant.insert(0, "group", group_indices[:min_len])
    df_scant.insert(1, "stiff_index", stiff_indices[:min_len])

    return df_scant


def update_group_value(df_scant, group_idx, column_name, new_value):
    cols = df_scant.columns.tolist()
    target_idx_list = [i for i, c in enumerate(cols) if c.strip() == column_name]

    if not target_idx_list:
        print(f"[WARN] column '{column_name}' not found in df_scant")
        return df_scant

    target_col_idx = target_idx_list[0]
    mask = (df_scant["group"] == group_idx)

    if not mask.any():
        print(f"[WARN] No rows found for group {group_idx}")
        return df_scant

    df_scant.iloc[mask.values, target_col_idx] = new_value
    return df_scant



if __name__ == "__main__":
    from utils.parser import parse_ma2
    parsed = parse_ma2("../data/sample.ma2")

    loc = parsed["stiff loc"]
    scant = parsed["stiff scant"]

    df_stiff_scant_new = group_stiff(scant, loc)
    print(df_stiff_scant_new.to_string())

    # df_scant = update_group_value(df_stiff_scant_new, 1, 12, "tweb", 18.5)
    # print(df_scant)

