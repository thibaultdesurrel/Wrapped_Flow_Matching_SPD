import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.ticker as mticker

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": "cm",
        "text.usetex": True,
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 12.5,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "axes.linewidth": 0.8,
    }
)

summary = pd.read_csv("synthetic_summary.csv", skipinitialspace=True)
for c in ("dataset", "model"):
    summary[c] = summary[c].str.strip()

wg = summary[summary["dataset"].str.startswith("wg_mixture_k")].copy()
wg["kappa"] = wg["dataset"].str.replace("wg_mixture_k", "", regex=False).astype(float)

MODELS = ["wrappedcfm", "geodesiccfm", "diffeocfm", "euclideancfm"]
STYLE = {
    "wrappedcfm": dict(
        label="WrappedCFM (ours)",
        color="#1f77b4",
        marker="o",
        ls="-",
        lw=2.2,
        ms=5.5,
        zorder=5,
    ),
    "diffeocfm": dict(
        label="DiffeoCFM",
        color="#ff7f0e",
        marker="^",
        ls="-.",
        lw=2.0,
        ms=5.5,
        zorder=4,
    ),
    "geodesiccfm": dict(
        label="RiemannianCFM",
        color="#2ca02c",
        marker="s",
        ls="--",
        lw=1.3,
        ms=4.2,
        zorder=3,
    ),
    "euclideancfm": dict(
        label="EuclideanCFM",
        color="#d62728",
        marker="D",
        ls=":",
        lw=1.4,
        ms=4.5,
        zorder=2,
    ),
}
# tiny multiplicative x-offsets so overlapping error bars stay legible
OFF = {"wrappedcfm": 1.0, "geodesiccfm": 0.93, "diffeocfm": 1.08, "euclideancfm": 0.86}

Y_TOP = 1.9

fig, ax = plt.subplots(figsize=(4.5, 3.15))

for model in MODELS:
    m = wg[wg["model"] == model].sort_values("kappa")
    style = STYLE[model].copy()
    marker = style.pop("marker")
    ax.errorbar(
        m["kappa"] * OFF[model],
        m["w2_mean"],
        yerr=m["w2_std"],
        marker=marker,
        capsize=2.5,
        capthick=0.9,
        elinewidth=0.9,
        markeredgecolor="white",
        markeredgewidth=0.5,
        **style,
    )

ax.set_xscale("log")
ax.set_xlabel(r"condition number $\kappa$ of the class centers")
ax.set_ylabel(r"$W_2$ distance ($\downarrow$)")
ax.set_xticks([10, 100, 1000, 10000])
ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())
ax.set_xlim(8, 1.3e4)

ax.set_yticks([1.0, 1.2, 1.4, 1.6, 1.8])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
ax.set_ylim(1.0, Y_TOP)

ax.grid(True, which="major", axis="both", color="0.85", lw=0.6, zorder=0)
ax.grid(True, which="minor", axis="x", color="0.92", lw=0.4, zorder=0)
ax.set_axisbelow(True)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color("0.3")

handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    ncol=1,
    frameon=False,
    handletextpad=0.5,
    labelspacing=0.8,
    handlelength=1.8,
)

# EuclideanCFM leaves the frame almost immediately; call out its actual
# (off-chart) W2 value at kappa=1e3 and kappa=1e4 with an up arrow.
euclid = wg[(wg["model"] == "euclideancfm") & (wg["kappa"] >= 1000)].sort_values(
    "kappa"
)
for _, row in euclid.iterrows():
    ax.annotate(
        f"{row['w2_mean']:.1f}" + r"$\uparrow$",
        xy=(row["kappa"] * OFF["euclideancfm"], Y_TOP * 0.97),
        ha="center",
        va="top",
        fontsize=12.5,
        color="#d62728",
    )

fig.tight_layout(pad=0.3)
fig.savefig("conditioning_w2.pdf", bbox_inches="tight")
plt.show()
