"""Aggregate the synthetic CFM benchmark into a mean +/- std summary table.

Scans logs/ for the per-run metrics written by evaluate_synthetic.py
(``<exp>/seed_<seed>/<timestamp>/eval/synthetic_metrics.csv``), and for each
(dataset, model) aggregates across seeds the three test-set metrics:

    pct_spd : % generated test samples that are SPD ("in M")
    ll      : mean test log-likelihood (per-run ll_mean averaged over seeds)
    w2      : Wasserstein-2 distance (true test x1 vs generated test)

For each seed only the most recent timestamp is used.

Output: results/synthetic_summary.csv (also printed to stdout).

Usage:
    cd runner
    uv run scripts/make_synthetic_table.py
"""

import os
from glob import glob

import pandas as pd

HERE = os.path.dirname(__file__)
PATH_TO_LOGS = os.path.join(HERE, "..", "logs")
PATH_TO_RESULTS = os.path.join(HERE, "..", "results")
OUTPUT = os.path.join(PATH_TO_RESULTS, "synthetic_summary.csv")

DATASETS = ["figure8", "swiss_roll"]
MODELS = ["wrappedcfm", "geodesiccfm", "diffeocfm", "euclideancfm"]


def split_experiment(name: str):
    """Split an experiment name like 'swiss_roll_wrappedcfm' -> (dataset, model)."""
    for ds in DATASETS:
        if name.startswith(ds + "_"):
            return ds, name[len(ds) + 1 :]
    return None, None


def collect_runs() -> pd.DataFrame:
    rows = []
    for exp in sorted(os.listdir(PATH_TO_LOGS)):
        exp_dir = os.path.join(PATH_TO_LOGS, exp)
        if not os.path.isdir(exp_dir):
            continue
        dataset, model = split_experiment(exp)
        if dataset is None or model not in MODELS:
            continue

        for seed_dir in sorted(glob(os.path.join(exp_dir, "seed_*"))):
            # most recent timestamp dir under this seed that has the metrics file
            candidates = sorted(
                glob(os.path.join(seed_dir, "*", "eval", "synthetic_metrics.csv"))
            )
            if not candidates:
                continue
            df = pd.read_csv(candidates[-1])
            r = df.iloc[0].to_dict()
            r["dataset"] = dataset
            r["model"] = model
            rows.append(r)

    return pd.DataFrame(rows)


def main():
    df = collect_runs()
    if df.empty:
        print(f"No synthetic_metrics.csv found under {os.path.abspath(PATH_TO_LOGS)}.")
        print("Run scripts/run_synthetic_benchmark.sh first.")
        return

    # Use the per-run ll_mean as each run's log-likelihood point estimate.
    df = df.rename(columns={"ll_mean": "ll"})
    metric_cols = ["pct_spd", "ll", "w2"]

    grouped = df.groupby(["dataset", "model"])
    summary = grouped[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary["n_seeds"] = grouped.size()
    summary = summary.reset_index()

    # Order rows by the canonical dataset/model order.
    summary["dataset"] = pd.Categorical(summary["dataset"], DATASETS, ordered=True)
    summary["model"] = pd.Categorical(summary["model"], MODELS, ordered=True)
    summary = summary.sort_values(["dataset", "model"]).reset_index(drop=True)

    cols = ["dataset", "model", "n_seeds"]
    for name in metric_cols:
        cols += [f"{name}_mean", f"{name}_std"]
    summary = summary[cols]

    os.makedirs(PATH_TO_RESULTS, exist_ok=True)
    summary.to_csv(OUTPUT, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved summary ({len(summary)} rows) to {os.path.abspath(OUTPUT)}")


if __name__ == "__main__":
    main()
