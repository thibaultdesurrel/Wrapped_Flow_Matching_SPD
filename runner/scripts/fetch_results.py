import os
import re
import pandas as pd

PATH_TO_LOGS = os.path.join(os.path.dirname(__file__), "..", "logs")
PATH_TO_RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(PATH_TO_RESULTS, exist_ok=True)

TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

all_results = []

for experiment in sorted(os.listdir(PATH_TO_LOGS)):
    path_to_experiment = os.path.join(PATH_TO_LOGS, experiment)
    if not os.path.isdir(path_to_experiment):
        continue

    # Experiment name format: dataset_model (model may contain underscores)
    exp_infos = experiment.split("_", 1)
    dataset, model = exp_infos[0], exp_infos[1]

    subdirs = sorted(os.listdir(path_to_experiment))

    # Detect structure: flat (timestamp directly under experiment) or grouped (group_X/timestamp)
    if any(TIMESTAMP_RE.match(d) for d in subdirs):
        # Flat structure: experiment/timestamp/eval
        runs_and_paths = [("group_0", path_to_experiment)]
    else:
        # Grouped structure: experiment/group_X/timestamp/eval
        runs_and_paths = [
            (run, os.path.join(path_to_experiment, run))
            for run in subdirs
            if os.path.isdir(os.path.join(path_to_experiment, run))
        ]

    for run, path_to_run in runs_and_paths:
        # Inside each run, there's a timestamp directory
        timestamps = [
            d
            for d in os.listdir(path_to_run)
            if os.path.isdir(os.path.join(path_to_run, d)) and TIMESTAMP_RE.match(d)
        ]
        if not timestamps:
            continue

        # Use the most recent timestamp
        timestamp = sorted(timestamps)[-1]
        path_to_timestamp = os.path.join(path_to_run, timestamp)
        path_to_eval = os.path.join(path_to_timestamp, "eval")

        if not os.path.isdir(path_to_eval):
            print(f"Warning: no eval dir in {path_to_timestamp}, skipping")
            continue

        row = {"dataset": dataset, "model": model, "run": run}

        # Training time
        training_time_path = os.path.join(path_to_timestamp, "training_time.log")
        if os.path.isfile(training_time_path):
            with open(training_time_path) as f:
                row["training_time_s"] = float(f.read().strip().rstrip("s"))

        # Sampling time
        sampling_time_path = os.path.join(path_to_eval, "sampling_time.log")
        if os.path.isfile(sampling_time_path):
            with open(sampling_time_path) as f:
                row["sampling_time_s"] = float(f.read().strip().rstrip("s"))

        # Classification metrics: flatten each row as "Comparison_Metric"
        df_cls = pd.read_csv(os.path.join(path_to_eval, "classification_metrics.csv"))
        for _, r in df_cls.iterrows():
            prefix = r["Comparison"]
            for col in df_cls.columns[1:]:
                row[f"cls_{prefix}_{col}"] = r[col]

        # Quality metrics
        df_qual = pd.read_csv(os.path.join(path_to_eval, "quality_metrics.csv"))
        for _, r in df_qual.iterrows():
            prefix = r["Comparison"]
            for col in df_qual.columns[1:]:
                row[f"qual_{prefix}_{col}"] = r[col]

        # SPD constraints
        df_spd = pd.read_csv(os.path.join(path_to_eval, "spd_constraints.csv"))
        for _, r in df_spd.iterrows():
            prefix = r["Subset"]
            for col in df_spd.columns[1:]:
                row[f"spd_{prefix}_{col}"] = r[col]

        all_results.append(row)

df_all = pd.DataFrame(all_results)
output_path = os.path.join(PATH_TO_RESULTS, "all_results.csv")
df_all.to_csv(output_path, index=False)
print(f"Saved {len(df_all)} results to {output_path}")
