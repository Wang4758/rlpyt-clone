"""
    Used to create the barplot that shows rule compliance rates.
"""

from io import StringIO
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rlpyt.projects.safe.experiments.scripts.plot.plot_settings import (
    TUM_BLUE, TUM_GREEN, TUM_ORANGE, set_plotting_theme, TUM_LIGHTBLUE
)

PAPER = True

if not PAPER:
    data = r"""Variant,R_G1,R_G2,R_G3,R_G0
Unconstrained,0.14,0.00,0.28,0.00
Human,0.90,0.99,0.65,0.42
"$\alpha=0.5, d=7.5$",0.94,0.99,0.97,0.91
"$\alpha=0.9, d=7.5$",0.95,0.95,0.97,0.89
    """
else:
    data = r"""Variant,R_G1,R_G2,R_G3,R_G0
"$\alpha=0.9$ (WCSAC)",0.96,0.94,1.0,0.91
"$\alpha=0.5$ (WCSAC)",0.96,0.94,0.99,0.91
"$\alpha=0.9$ (ours)",0.95,0.95,0.97,0.89
"$\alpha=0.5$ (ours)",0.94,0.99,0.97,0.91
Human,0.90,0.99,0.65,0.42
Unconstrained,0.14,0.00,0.28,0.00
    """

io = StringIO(data)

df_initial: pd.DataFrame = pd.read_csv(io)
df = df_initial.melt(id_vars=["Variant"], value_vars=["R_G1", "R_G2", "R_G3", "R_G0"], value_name="Compliance")
df.rename(columns={"variable": "Rule"}, inplace=True)

set_plotting_theme(
    df["Variant"].nunique(), 
    palette_name="tab10" if PAPER else "deep", 
    fontsize=14 if PAPER else 11,
    font = "Times" if PAPER else "Palatino"
)

# 5.8... is the \textwidth of LaTeX
# with the golden ratio the height should be 3.6...
figsize = (5.84036, 3.6)

g = sns.catplot(
    data=df, 
    x="Rule", 
    y="Compliance", 
    hue="Variant",
    kind="bar", 
    height=figsize[1],
    aspect=figsize[0] / figsize[1],
)

sns.move_legend(g, "center", bbox_to_anchor=(.4, -0.07), title=None, frameon=False, ncol=2)

if not PAPER:
    hatches = ["", "...", "\\\\", "\\\\\\", "...","", "xx"]
    colors = [TUM_BLUE, TUM_ORANGE, TUM_GREEN, TUM_GREEN, TUM_GREEN]
else:
    hatches = ["\\", "\\\\\\", "\\", "\\\\\\", "...","", "xx"]
    colors = [TUM_ORANGE, TUM_ORANGE, TUM_BLUE, TUM_BLUE, TUM_GREEN, TUM_LIGHTBLUE]

patch_colors = list()
for patch in g.ax.patches:
    patch_color = patch._facecolor
    if patch_color not in patch_colors:
        patch_colors.append(patch_color)
    color_idx = patch_colors.index(patch_color)
    patch.set_hatch(hatches[color_idx % len(hatches)])
    patch.set_facecolor(colors[color_idx])
    patch.set_edgecolor("black")

for patch in g.legend.legendHandles:
    color_idx = patch_colors.index(patch._facecolor)
    patch.set_hatch(hatches[color_idx % len(hatches)])
    patch.set_facecolor(colors[color_idx])
    patch.set_edgecolor("black")

# annotate bar values: https://stackoverflow.com/a/68323374
if not PAPER or True:
    ax = g.ax
    for container in ax.containers:
        ax.bar_label(container, rotation='vertical', label_type='edge', padding=4, fontsize=11)

g.tight_layout()
g.savefig(f"/home/pillmayerc/mth/plots/traffic_rule_conformance_overview{'_paper' if PAPER else ''}.pdf")
