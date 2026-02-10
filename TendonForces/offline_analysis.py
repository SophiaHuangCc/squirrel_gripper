import os
import re
import pandas as pd
import numpy as np
import wandb

# Initialize wandb project
run = wandb.init(project="tendonforces-offline-ranking", job_type="analysis")

# Set your results directory
results_dir = "runs/exp7"  # Change to your experiment directory
CONTACT_LOG_RE = re.compile(r"^contact_log_(\d{8}_\d{6})_(.+)\.csv$")
MASTER_LOG_RE = re.compile(r"^master_log_(\d{8}_\d{6})_(.+)\.npz$")


def parse_params_from_name(name):
    params = {}
    if not name:
        return params
    patterns = {
        "T": r"(?:^|_)T(?P<T>-?\d+(\.\d+)?)",
        "R": r"(?:^|_)R(?P<R>-?\d+(\.\d+)?)",
        "JS": r"(?:^|_)JS(?P<JS>-?\d+(\.\d+)?)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, name)
        if match:
            params[key] = float(match.group(key))
    return params


def find_video_for_run(run_id, suffix, run_dir):
    expected_name = f"output_{run_id}_{suffix}.mp4"
    expected_path = os.path.join(run_dir, expected_name)
    if os.path.isfile(expected_path):
        return expected_path
    mp4s = [f for f in os.listdir(run_dir) if f.endswith(f"_{suffix}.mp4")]
    if mp4s:
        mp4s.sort(key=lambda f: os.path.getmtime(os.path.join(run_dir, f)), reverse=True)
        print(f"No exact match for {expected_name} in {run_dir}, using {mp4s[0]}")
        return os.path.join(run_dir, mp4s[0])
    print(f"No video found for run_id={run_id} suffix={suffix} in {run_dir}")
    return None


def extract_metrics_from_npz(npz_path):
    metrics = {}
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            for key in data.files:
                if not (key.startswith("metric_") or key in ("final_grasp_score", "final_score")):
                    continue
                value = data[key]
                if np.size(value) != 1:
                    continue
                scalar = value.reshape(-1)[0]
                if isinstance(scalar, np.generic):
                    scalar = scalar.item()
                if isinstance(scalar, (int, float, bool, np.integer, np.floating)):
                    metrics[key] = scalar
    except Exception as exc:
        print(f"Failed to read metrics from {npz_path}: {exc}")
    return metrics


def find_video_for_csv(csv_path):
    csv_dir = os.path.dirname(csv_path)
    csv_base = os.path.basename(csv_path)
    mp4s = [f for f in os.listdir(csv_dir) if f.endswith(".mp4")]
    if not mp4s:
        return None
    match = CONTACT_LOG_RE.match(csv_base)
    if match:
        run_id, suffix = match.groups()
        expected_name = f"output_{run_id}_{suffix}.mp4"
        expected_path = os.path.join(csv_dir, expected_name)
        if os.path.isfile(expected_path):
            return expected_path
        suffix_matches = [f for f in mp4s if f.endswith(f"_{suffix}.mp4")]
        if suffix_matches:
            suffix_matches.sort(key=lambda f: os.path.getmtime(os.path.join(csv_dir, f)), reverse=True)
            print(f"No exact match for {expected_name} in {csv_dir}, using {suffix_matches[0]}")
            return os.path.join(csv_dir, suffix_matches[0])
        print(f"No video found for {csv_base} (expected {expected_name})")
        return None

    csv_stem = os.path.splitext(csv_base)[0]
    if csv_stem.startswith("contact_log_"):
        candidate = f"output_{csv_stem[len('contact_log_'):]}.mp4"
        candidate_path = os.path.join(csv_dir, candidate)
        if os.path.isfile(candidate_path):
            return candidate_path

    # Fallback to newest video only when no naming rule applies.
    mp4s.sort(key=lambda f: os.path.getmtime(os.path.join(csv_dir, f)), reverse=True)
    print(f"No naming rule match for {csv_base}, using newest video: {mp4s[0]}")
    return os.path.join(csv_dir, mp4s[0])


def build_results_table(df):
    if df.empty:
        return wandb.Table(columns=[])
    table_cols = [col for col in df.columns if col != "video_path"]
    if "video" not in table_cols:
        table_cols.append("video")
    table = wandb.Table(columns=table_cols)
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        video_path = row_dict.get("video_path")
        if video_path and os.path.isfile(video_path):
            row_dict["video"] = wandb.Video(video_path, fps=4, format="mp4")
        else:
            row_dict["video"] = None
        table.add_data(*[row_dict.get(col) for col in table_cols])
    return table

# Scan for all result master logs (metrics live in master_log_*.npz)
data = []
for root, dirs, files in os.walk(results_dir):
    for f in files:
        if f.endswith(".npz"):
            match = MASTER_LOG_RE.match(f)
            if not match:
                continue
            run_id, suffix = match.groups()
            npz_path = os.path.join(root, f)
            row = {
                "run_id": run_id,
                "suffix": suffix,
            }
            row.update(parse_params_from_name(suffix))
            row.update(extract_metrics_from_npz(npz_path))
            csv_path = os.path.join(root, f"contact_log_{run_id}_{suffix}.csv")
            if os.path.isfile(csv_path):
                row["csv_path"] = csv_path
            video_path = find_video_for_run(run_id, suffix, root)
            if video_path:
                row["video_path"] = os.path.abspath(video_path)
            data.append(row)

df_all = pd.DataFrame(data)

# Log full dataframe as a table, including videos where available
run.log({"all_results": build_results_table(df_all)})

# Find best design and top-10 for each metric
metric_cols = [
    col for col in df_all.columns
    if col.startswith("metric_") or col in ("final_grasp_score", "final_score")
]
for metric in metric_cols:
    if metric in df_all:
        metric_series = pd.to_numeric(df_all[metric], errors="coerce")
        if metric_series.notna().any():
            best_idx = metric_series.idxmax()
            best_csv = df_all.loc[best_idx].get("csv_path")
            run.log({
                f"best_{metric}": metric_series.loc[best_idx],
                f"best_{metric}_csv": best_csv,
            })
            print(f"Best {metric}: {metric_series.loc[best_idx]} at {best_csv}")

            top10 = df_all.assign(_metric=metric_series).sort_values("_metric", ascending=False).head(10)
            top10 = top10.drop(columns=["_metric"])
            run.log({f"top10_{metric}": build_results_table(top10)})

run.finish()