"""
    Plots the safe distance graphs used in the thesis where the three worst worst-case agents are shown.
"""

from pathlib import Path
import pandas as pd
from rlpyt.projects.safe.experiments.scripts.plot.plot_settings import TUM_BLUE, TUM_GREEN, TUM_ORANGE, set_plotting_theme
import seaborn as sns
import matplotlib.pyplot as plt

set_plotting_theme(1, fontsize=11)

paths = [
    Path('/home/pillmayerc/mth/viz/DEU_LocationAUpper-27_29_T-1/safe_distance_data_DEU_LocationAUpper-27_29_T-1.csv'),
    Path('/home/pillmayerc/mth/viz/DEU_LocationAUpper-35_19_T-1/safe_distance_data_DEU_LocationAUpper-35_19_T-1.csv'),
    Path('/home/pillmayerc/mth/viz/DEU_LocationDLower-7_9_T-1/safe_distance_data_DEU_LocationDLower-7_9_T-1.csv')
]

ranges = [
    "`t` > 104 and `t` <= 129",
    "`t` > 127 and `t` <= 152",
    "`t` > 184 and `t` <= 209"
]

labels = ["a)", "b)", "c)"]

colors = [TUM_BLUE, TUM_GREEN, TUM_ORANGE]

fig, ax = plt.subplots(figsize=(5.8, 2))

for path, range, label, color in zip(paths, ranges, labels, colors):

    benchmark_id = path.parent.name
    save_path = path.parent.joinpath(f'safe_distance_plot.pdf')
    df = pd.read_csv(path)
    df["t"] = df.index + 1
    df = df.query(range)
    df["t"] -= df["t"].max()
    df["violation"] = (df["safe_dist_lead"] - df["dist_lead"]).clip(lower=-1)

    sns.lineplot(
        data=df,
        ax=ax,
        x="t",
        y="violation",
        drawstyle='steps-pre',
        color=color,
        label=label
    )

    ax.set_ylabel(r"$d_\mathrm{safe} - d_\mathrm{lead}$")
    ax.set_xlabel(r"$t_\mathrm{rel}$")
    ax.legend()
    sns.move_legend(ax, loc="lower center", ncol = 3, bbox_to_anchor=(0.5, 1.05), frameon=False)

    fig.tight_layout()
    fig.savefig(save_path)