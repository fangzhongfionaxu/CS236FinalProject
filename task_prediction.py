import csv
import random
import os
import pandas as pd
from prophet import Prophet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#predict for ech location 
#predict the reward at given time

def _last_reward_col(df):
    cols = [c for c in df.columns if c.startswith("reward_run")]
    if cols:
        return sorted(cols)[-1]
    if "reward" in df.columns:
        return "reward"
    raise RuntimeError("No reward column found in CSV")
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

def predict_reward_at_range(vertex, t1, t2, df_path="project_files/tasklog_reward.csv", reward_col=None):
    """
    Predict rewards for a single vertex for all integer minutes from t1 to t2 .
    Returns a DataFrame with columns ['minute', 'predicted_reward'].
    """
    predictions = []
    
    for t in range(t1, t2 + 1):
        pred = predict_reward_at(vertex, t, df_path=df_path, reward_col=reward_col)
        predictions.append({"minute": t, "predicted_reward": pred})
    pred_df = pd.DataFrame(predictions)
    print(pred_df)
    return pred_df

def predict_all_vertices(t_minute, df_path="project_files/tasklog_reward.csv", reward_col=None):
    """
    predict_reward_at all vertices: run prophet for all location
    - vertex: value in VERTEX column
    - t_minute: integer minutes (same units as 'minute' column)
    Returns dataframe of predicted reward for task at all vertices at minute t_minute.
    """
    df = pd.read_csv(df_path)

    if "minute" not in df.columns or "VERTEX" not in df.columns:
        raise RuntimeError("CSV must contain 'minute' and 'VERTEX' columns")

    reward_col = reward_col or _last_reward_col(df)

    origin = pd.Timestamp("1970-01-01")
    target_dt = origin + pd.to_timedelta(int(t_minute), unit="m")

    predictions = []

    for vertex, group in df.groupby("VERTEX"):
        y = pd.to_numeric(group[reward_col], errors="coerce").dropna()

        if len(y) == 0:
            continue

        if len(y) == 1:
            pred = float(y.iloc[-1])
            
        else:
            ds = origin + pd.to_timedelta(
                group.loc[y.index, "minute"].astype(int),
                unit="m"
            )

            ts = pd.DataFrame({"ds": ds, "y": y.values})

            m = Prophet(
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False
            )
            m.fit(ts)

            future = pd.DataFrame({"ds": [target_dt]})
            forecast = m.predict(future)
            pred = max(0,float(forecast["yhat"].iloc[0]))

        predictions.append({
            "VERTEX": vertex,
            "predicted_reward": pred
        })

    return pd.DataFrame(predictions)

def get_historical_rewards_by_location(location, t1, t2, df_path="project_files/tasklog_reward.csv", reward_col=None):
    """
    Get historical rewards for a single vertex over a range of minutes [t1, t2].
    Returns a DataFrame with columns ['minute', 'historical_reward'] suitable for plotting.
    """
    df = pd.read_csv(df_path)
    
    if "minute" not in df.columns or "VERTEX" not in df.columns:
        raise RuntimeError("CSV must contain 'minute' and 'VERTEX' columns")
    
    reward_col = reward_col or _last_reward_col(df)
    
    # Filter for this vertex and the time range
    vertex_df = df[df["VERTEX"] == location].copy()
    vertex_df = vertex_df[(vertex_df["minute"] >= t1) & (vertex_df["minute"] <= t2)]
    
    # Ensure numeric rewards and drop NaNs
    vertex_df[reward_col] = pd.to_numeric(vertex_df[reward_col], errors="coerce")
    vertex_df = vertex_df.dropna(subset=[reward_col])
    
    # Create a DataFrame with the same structure as predicted
    hist_df = vertex_df[["minute", reward_col]].rename(columns={reward_col: "historical_reward"})
    
    # If some minutes in range are missing, you can optionally fill them with 0
    all_minutes = pd.DataFrame({"minute": range(t1, t2 + 1)})
    hist_df = pd.merge(all_minutes, hist_df, on="minute", how="left").fillna(0)
    print(hist_df[hist_df["historical_reward"] != 0])
    return hist_df

def graph_by_location(location,t1,t2):
    """line graph of comparison of true and predicted rewards at the same location time ranging form t1 to t2"""
    hist_df = get_historical_rewards_by_location(location, t1, t2)
    pred_df = predict_reward_at_range(location, t1, t2)

    mse = mean_squared_error(hist_df["historical_reward"], pred_df["predicted_reward"])
    print(f"MSE for Vertex {location} over minutes {t1}-{t2}: {mse:.4f}")

    plt.figure(figsize=(12, 5))
    plt.plot(pred_df["minute"], pred_df["predicted_reward"], label="Predicted", color="skyblue", marker='o')
    plt.plot(hist_df["minute"], hist_df["historical_reward"], label="Historical", color="gold", marker='x')
    plt.xlabel("Minute")
    plt.ylabel("Reward")
    plt.title(f"Predicted vs Historical Rewards for Vertex {location}")
    plt.legend()
    plt.grid(True)
    plt.show()

def get_historical_rewards_by_time(time, df_path= "project_files/tasklog_reward.csv", reward_col=None) -> pd.DataFrame:
    """
    Extract historical rewards for all locations at a specific time (minute).
    
    Returns a DataFrame with columns ['VERTEX', 'historical_reward'].
    """
    df = pd.read_csv(df_path)
    
    if "minute" not in df.columns or "VERTEX" not in df.columns:
        raise RuntimeError("CSV must contain 'minute' and 'VERTEX' columns")
    
    # Determine reward column
    reward_col = reward_col or _last_reward_col(df)
    
    hist = []
    for vertex, group in df.groupby("VERTEX"):
        group = group[group["minute"] == time]
        y = pd.to_numeric(group[reward_col], errors="coerce").dropna()
        reward = float(y.iloc[-1]) if len(y) > 0 else 0  # set 0 if no data
        hist.append({"VERTEX": vertex, "historical_reward": reward})
    
    return pd.DataFrame(hist)


def graph_by_time(time: int):
    """
    Bar Chart Graph predicted rewards for all vertices at a given time.
    - x-axis: vertex (sorted ascending)
    - y-axis: reward
    - color: blue for predicted, yellow for historical
    - prints MSE in terminal
    """

    # Get data internally
    predictions = predict_all_vertices(time)  # must return ['VERTEX', 'predicted_reward']
    historical_rewards = get_historical_rewards_by_time(time)  # ['VERTEX', 'historical_reward']

    # Merge predicted and historical rewards
    merged = pd.merge(
        predictions,
        historical_rewards,
        on="VERTEX",
        how="outer"
    ).fillna(0)

    merged = merged.sort_values("VERTEX")

    vertices = merged["VERTEX"].astype(str)
    predicted = merged["predicted_reward"]
    historical = merged["historical_reward"]

    # Compute MSE
    mse = mean_squared_error(historical, predicted)
    print(f"MSE for all vertices at time {time}: {mse:.4f}")

    # Plot
    x = np.arange(len(vertices))
    width = 0.35

    plt.figure(figsize=(14, 6))
    plt.bar(x - width / 2, predicted, width, label="Predicted", color="skyblue")
    plt.bar(x + width / 2, historical, width, label="Historical", color="gold")

    plt.xlabel("Vertex (Location ID)")
    plt.ylabel("Reward")
    plt.title(f"Predicted vs Historical Rewards at Time = {time}")
    plt.xticks(x, vertices, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()


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
   
    graph_by_time(1227)
    graph_by_location(34,1210,1240)

