"""Aggregate training times from runner/logs and report mean per dataset/method."""

import re
from collections import defaultdict
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
METHODS = ["diffeocfm", "euclideancfm", "geodesiccfm", "wrappedcfm"]
TIME_RE = re.compile(r"([\d.]+)s")


def parse_run_name(name):
    for method in METHODS:
        if name.endswith("_" + method):
            return name[: -len("_" + method)], method
    return None, None


def main():
    times = defaultdict(list)

    for log_file in LOGS_DIR.glob("*/seed_*/*/training_time.log"):
        run_name = log_file.parents[2].name
        dataset, method = parse_run_name(run_name)
        if dataset is None:
            print(f"Skipping unrecognized run name: {run_name}")
            continue

        text = log_file.read_text().strip()
        match = TIME_RE.search(text)
        if not match:
            print(f"Skipping unparseable file: {log_file}")
            continue

        times[(dataset, method)].append(float(match.group(1)))

    datasets = sorted({d for d, _ in times})
    methods = [m for m in METHODS if any(m == meth for _, meth in times)]

    stats = {}
    for key, vals in times.items():
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        stats[key] = (mean / 60, std / 60, len(vals))  # minutes

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
                cell = f"{mean:.1f} \\pm {std:.1f}"
                if method == best_method:
                    cell = f"\\mathbf{{{mean:.1f}}} \\pm {std:.1f}"
                row.append(f"${cell}$")
            else:
                row.append("--")
        print(" & ".join(row) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Mean training time per dataset and method (minutes, mean $\pm$ std).}")
    print(r"\label{tab:training_times}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
