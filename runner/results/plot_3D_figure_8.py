import torch
import sys
from pathlib import Path

_RUNNER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RUNNER_DIR))
sys.path.insert(0, str(_RUNNER_DIR / "src"))

from src.utils.loading import load_model
from src.datamodules.synthetic import SwissRollDataModule, Figure8DataModule

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",  # Utilisation de la police serif
        "font.serif": "cm",  # Utilisation de la police Computer Modern
        "text.usetex": True,  # Utilisation de LaTeX pour le rendu du texte
    }
)

model_wrappedcfm, cfg_wrappedcfm = load_model(
    _RUNNER_DIR / "logs" / "figure8_wrappedcfm" / "seed_0" / "2026-07-14_12-33-42"
)

N = 2000
x0 = model_wrappedcfm.sample_x0(N)
# the model was trained single-class: labels are one-hot(0, num_classes=1) = 1.0,
# so the conditioning input must be ones (zeros is out-of-distribution -> inflated samples)
y = torch.ones(N, 1, device=model_wrappedcfm.device)

x1_tilde = model_wrappedcfm.sample_ode(x0, y)
x1_tilde_np = x1_tilde.detach().cpu().numpy()
dm = Figure8DataModule(
    n_samples=N * 5 // 4,
    batch_size=100,
    num_workers=0,
    recenter=cfg_wrappedcfm["datamodule"]["recenter"],
    seed=cfg_wrappedcfm["seed"],
)
dm.setup()
x1 = dm.train_ds.tensors[0].detach().cpu().numpy()
fig = plt.figure(figsize=(12, 6))

# ==========================================
# 1. Setup the Cone Data
# ==========================================
# We define the cone boundary where ac = b^2
# Using a slightly denser grid (100) for a smoother look
a_range = np.linspace(0, 6, 300)
c_range = np.linspace(0, 6, 300)
A, C = np.meshgrid(a_range, c_range)
B = np.sqrt(A * C)  # The boundary values for b


def plot_spd_cone(ax):
    """
    Plots the boundary of the SPD cone (ac = b^2) as a semi-transparent surface.
    """
    # Plot top half of the cone (b is positive)
    # rstride/cstride=5 controls the downsampling of the grid (higher = coarser)
    # linewidth=0 removes the black mesh lines for a smooth "glass" look
    ax.plot_surface(
        A,
        B,
        C,
        color="lightseagreen",
        alpha=0.2,
        rstride=20,
        cstride=20,
        linewidth=0,
        antialiased=True,
    )

    # Plot bottom half of the cone (b is negative)
    ax.plot_surface(
        A,
        -B,
        C,
        color="lightseagreen",
        alpha=0.2,
        rstride=20,
        cstride=20,
        linewidth=0,
        antialiased=True,
    )

    # Optional: Plot wireframe on top for better depth perception (faint lines)
    # ax.plot_wireframe(A, B, C, color='darkslategray', alpha=0.1, rstride=10, cstride=10)
    # ax.plot_wireframe(A, -B, C, color='darkslategray', alpha=0.1, rstride=10, cstride=10)


# =============
# First Column
# =============
ax1 = fig.add_subplot(2, 3, 1, projection="3d")
plot_spd_cone(ax1)
# Plot data points
ax1.scatter(x1[:, 0, 0], x1[:, 0, 1], x1[:, 1, 1], c="b", s=2)  # , depthshade=True)
ax1.set_title("Front view", fontsize=20, y=0.95)
ax1.set_axis_off()
ax1.set_aspect("equal", "box")
# Adjust view angle if needed
ax1.view_init(elev=40, azim=25)

ax2 = fig.add_subplot(2, 3, 4, projection="3d")
plot_spd_cone(ax2)
# Plot data points
ax2.scatter(
    x1_tilde_np[:, 0, 0],
    x1_tilde_np[:, 0, 1],
    x1_tilde_np[:, 1, 1],
    c="red",
    s=2,
    depthshade=True,
)
ax2.set_axis_off()
ax2.set_aspect("equal", "box")
# Adjust view angle if needed
ax2.view_init(elev=40, azim=25)


# =============
# Second Column
# =============
ax1 = fig.add_subplot(2, 3, 2, projection="3d")
plot_spd_cone(ax1)
# Plot data points
ax1.scatter(x1[:, 0, 0], x1[:, 0, 1], x1[:, 1, 1], c="b", s=2)  # , depthshade=True)
ax1.set_title("Top view", fontsize=20, y=0.95)
ax1.set_axis_off()
ax1.set_aspect("equal", "box")
# Adjust view angle if needed
ax1.view_init(elev=30, azim=0, roll=0)

ax2 = fig.add_subplot(2, 3, 5, projection="3d")
plot_spd_cone(ax2)
ax2.set_title("Figure-8", fontsize=25, y=-0.01)
# Plot data points
ax2.scatter(
    x1_tilde_np[:, 0, 0],
    x1_tilde_np[:, 0, 1],
    x1_tilde_np[:, 1, 1],
    c="red",
    s=2,
    depthshade=True,
)
ax2.set_axis_off()
ax2.set_aspect("equal", "box")
# Adjust view angle if needed
ax2.view_init(elev=30, azim=0, roll=0)


# =============
# Third Column
# =============
ax1 = fig.add_subplot(2, 3, 3, projection="3d")
plot_spd_cone(ax1)
# Plot data points
ax1.scatter(x1[:, 0, 0], x1[:, 0, 1], x1[:, 1, 1], c="b", s=2)  # , depthshade=True)
ax1.set_title("Side view", fontsize=20, y=0.95)
ax1.set_axis_off()
ax1.set_aspect("equal", "box")
# Adjust view angle if needed
ax1.view_init(elev=0, azim=90, roll=-45)

ax2 = fig.add_subplot(2, 3, 6, projection="3d")
plot_spd_cone(ax2)
# Plot data points
ax2.scatter(
    x1_tilde_np[:, 0, 0],
    x1_tilde_np[:, 0, 1],
    x1_tilde_np[:, 1, 1],
    c="red",
    s=2,
    depthshade=True,
)
ax2.set_axis_off()
ax2.set_aspect("equal", "box")
# Adjust view angle if needed
ax2.view_init(elev=0, azim=90, roll=-45)

# plt.tight_layout()
plt.subplots_adjust(hspace=-0.27, wspace=-0.4)

plt.savefig("figure_8_in_cone.pdf", dpi=300, bbox_inches="tight")
plt.show()
