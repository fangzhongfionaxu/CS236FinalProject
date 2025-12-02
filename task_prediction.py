import csv
import random
import os
import pandas as pd
from prophet import Prophet
import numpy as np

def _last_reward_col(df):
    cols = [c for c in df.columns if c.startswith("reward_run")]
    if cols:
        return sorted(cols)[-1]
    if "reward" in df.columns:
        return "reward"
    raise RuntimeError("No reward column found in CSV")

def predict_reward_at(vertex, t_minute, df_path="project_files/tasklog_reward.csv", reward_col=None):
    """
    Very simple: uses the 'minute' column as time (integer minutes).
    - vertex: value in VERTEX column
    - t_minute: integer minutes (same units as 'minute' column)
    Returns predicted reward (float) for the task at the same VERTEX at minute t_minute.
    """
    df = pd.read_csv(df_path)
    if "minute" not in df.columns or "VERTEX" not in df.columns:
        raise RuntimeError("CSV must contain 'minute' and 'VERTEX' columns")

    reward_col = reward_col or _last_reward_col(df)

    group = df[df["VERTEX"] == vertex].copy()
    if group.empty:
        return float("nan")

    y = pd.to_numeric(group[reward_col], errors="coerce").dropna()
    if y.empty:
        return float("nan")
    if len(y) == 1:
        return float(y.iloc[-1])

    # Build datetimes from minute integers (fixed epoch)
    origin = pd.Timestamp("1970-01-01")
    ds = origin + pd.to_timedelta(group.loc[y.index, "minute"].astype(int), unit="m")
    ts = pd.DataFrame({"ds": ds, "y": y.values})

    m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
    m.fit(ts)

    t_dt = origin + pd.to_timedelta(int(t_minute), unit="m")
    future = pd.DataFrame({"ds": [t_dt]})
    forecast = m.predict(future)
    return float(forecast["yhat"].iloc[0])

def read_tasks_and_write_rewards(input_fname, output_fname, appear_time_fixed):
    """
    Read CSV that has columns USERID,VERTEX,TIME,minute and write a copy with
    an added 'reward' column (random here). Uses minute column only.
    Returns list of task dicts.
    """
    tasks = []

    with open(input_fname, "r", newline="") as fin:
        reader = csv.reader(fin)
        header = next(reader)

        # find indices (fall back to expected positions)
        try:
            vertex_idx = header.index("VERTEX")
        except ValueError:
            vertex_idx = 1
        try:
            minute_idx = header.index("minute")
        except ValueError:
            minute_idx = 3

        new_header = header + ["reward"]

        with open(output_fname, "w", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(new_header)

            task_id = 0
            for row in reader:
                # skip short/malformed rows
                if len(row) <= minute_idx or row[minute_idx] == "":
                    continue
                try:
                    target_time = int(row[minute_idx])
                except ValueError:
                    continue

                reward = random.randint(1, 100)
                appear_time = target_time - appear_time_fixed

                writer.writerow(row + [reward])

                task = {
                    'task_id': task_id,
                    'location': int(row[vertex_idx]),
                    'appear_time': appear_time,
                    'target_time': target_time,
                    'reward': reward
                }
                tasks.append(task)
                task_id += 1

    return tasks

def append_rewards(input_fname, appear_time_fixed):
    """
    Append one new reward_run_* column to CSV using the 'minute' column.
    """
    with open(input_fname, "r", newline="") as fin:
        rows = list(csv.reader(fin))
        if not rows:
            print(f"{input_fname} is empty")
            return
        header = rows[0]
        data_rows = rows[1:]

    # find minute column index
    try:
        minute_idx = header.index("minute")
    except ValueError:
        minute_idx = 3

    existing_reward_count = sum(1 for h in header if h.lower().startswith("reward"))
    new_reward_col = f"reward_run_{existing_reward_count + 1}"

    for row in data_rows:
        if len(row) < len(header):
            row.extend([""] * (len(header) - len(row)))

        if minute_idx >= len(row) or row[minute_idx] == "":
            row.append("")  # cannot compute reward without minute
            continue

        try:
            target_time = int(row[minute_idx])
        except ValueError:
            row.append("")
            continue

        appear_time = target_time - appear_time_fixed
        reward = random.randint(1, 100)
        row.append(str(reward))

    header.append(new_reward_col)

    with open(input_fname, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        writer.writerows(data_rows)

    print(f"Appended {new_reward_col} to {input_fname}")

print("Writing file to:", os.path.abspath("project_files/tasklog_reward.csv"))

read_tasks_and_write_rewards(
    "project_files/tasklog.csv",
    "project_files/tasklog_reward.csv",
    appear_time_fixed=20
)

append_rewards("project_files/tasklog_reward.csv", appear_time_fixed=20)
append_rewards("project_files/tasklog_reward.csv", appear_time_fixed=20)

# Example usage (when run as script)
if __name__ == "__main__":
    pred = predict_reward_at(45, 1161)
    print(pred)

