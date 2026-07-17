"""Time sampling of 1000 points from each trained CFM checkpoint and report
the mean sampling time per dataset/method as a LaTeX table.

Unlike training_times.py, there is no pre-existing log to parse: each run's
checkpoint is loaded and timed live. Results are cached to a CSV next to this
script so re-running the report doesn't re-time every checkpoint.
"""

import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

_RUNNER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RUNNER_DIR))
sys.path.insert(0, str(_RUNNER_DIR / "src"))

from utils.loading import load_model  # noqa: E402

LOGS_DIR = _RUNNER_DIR / "logs"
CACHE_CSV = Path(__file__).resolve().parent / "sampling_times.csv"
METHODS = ["diffeocfm", "euclideancfm", "geodesiccfm", "wrappedcfm"]
N_SAMPLES = 1000
N_WARMUP = 8


def parse_run_name(name):
    for method in METHODS:
        if name.endswith("_" + method):
            return name[: -len("_" + method)], method
    return None, None


def time_sampling(model, num_classes, n, device):
    labels = torch.randint(0, num_classes, (n,), device=device)
    y = F.one_hot(labels, num_classes=num_classes).to(next(model.parameters()).dtype)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    x0 = model.sample_x0(batch_size=n, int_labels=labels)
    _ = model.sample_ode(x0, y)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start


def measure_all():
    rows = []
    run_dirs = sorted(LOGS_DIR.glob("*/seed_*/*/"))
    for i, run_dir in enumerate(run_dirs):
        if not (run_dir / "training_time.log").exists():
            continue
        run_name = run_dir.parents[1].name
        seed = run_dir.parents[0].name
        dataset, method = parse_run_name(run_name)
        if dataset is None:
            continue

        print(f"[{i + 1}/{len(run_dirs)}] {run_name}/{seed} ...", end=" ", flush=True)
        model, cfg = load_model(run_dir)
        device = next(model.parameters()).device
        num_classes = cfg.model.num_classes

        with torch.no_grad():
            time_sampling(model, num_classes, N_WARMUP, device)  # warmup
            elapsed = time_sampling(model, num_classes, N_SAMPLES, device)

        print(f"{elapsed:.3f}s")
        rows.append({"dataset": dataset, "method": method, "seed": seed, "time_s": elapsed})

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(CACHE_CSV, index=False)
    return df


def main():
    if CACHE_CSV.exists():
        df = pd.read_csv(CACHE_CSV)
        print(f"Loaded cached sampling times from {CACHE_CSV}")
    else:
        df = measure_all()

    times = defaultdict(list)
    for row in df.itertuples():
        times[(row.dataset, row.method)].append(row.time_s)

    datasets = sorted({d for d, _ in times})
    methods = [m for m in METHODS if any(m == meth for _, meth in times)]

    stats = {}
    for key, vals in times.items():
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        stats[key] = (mean, std, len(vals))

    method_labels = {
        "diffeocfm": "DiffeoCFM",
        "euclideancfm": "EuclideanCFM",
        "geodesiccfm": "GeodesicCFM",
        "wrappedcfm": "WrappedCFM",
    }

    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\begin{tabular}{l" + "c" * len(methods) + "}")
    print(r"\toprule")
    print("Dataset & " + " & ".join(method_labels.get(m, m) for m in methods) + r" \\")
    print(r"\midrule")
    for dataset in datasets:
        row = [dataset.replace("_", r"\_")]
        row_means = {
            method: stats[(dataset, method)][0]
            for method in methods
            if (dataset, method) in stats
        }
        best_method = min(row_means, key=row_means.get) if row_means else None
        for method in methods:
            key = (dataset, method)
            if key in stats:
                mean, std, _ = stats[key]
                cell = f"{mean:.3f} \\pm {std:.3f}"
                if method == best_method:
                    cell = f"\\mathbf{{{mean:.3f}}} \\pm {std:.3f}"
                row.append(f"${cell}$")
            else:
                row.append("--")
        print(" & ".join(row) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(rf"\caption{{Mean sampling time for {N_SAMPLES} samples per dataset and method (seconds, mean $\pm$ std).}}")
    print(r"\label{tab:sampling_times}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
