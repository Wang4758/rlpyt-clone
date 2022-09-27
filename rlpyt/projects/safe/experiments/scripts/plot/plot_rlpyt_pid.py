""" Creates the big 4x4 pi-tuning plot from the thesis """

from argparse import ArgumentParser
import pathlib
from rlpyt.projects.safe.experiments.scripts.plot.plot_settings import set_plotting_theme
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import logging
from itertools import product
import json

logging.basicConfig(level=logging.INFO)

def load_config(progress_path: pathlib.Path):
    config_path = progress_path.parent.joinpath('params.json')
    with open(config_path, 'r') as f:
        config: dict = json.load(f)
    return config

def load_data(path: pathlib.Path):

    dfs = list()

    csvs = list(path.glob('**/progress.csv'))
    for progress_csv in tqdm(csvs, desc='Loading progress files'):
        df = pd.read_csv(progress_csv)[['IsGoalReachedAverage', 'CostAverage', 'costPenaltyAverage', 'Diagnostics/CumSteps']]
        config = load_config(progress_csv)
        df['CostAverage'] = df['CostAverage'].rolling(5).mean()
        df['$K_I$'] = config['algo']['pid_Ki']
        df['$K_P$'] = config['algo']['pid_Kp']

        dfs.append(df)

    return pd.concat(dfs).reset_index()

def plot_from_dir(path: pathlib.Path):
    
    abs_cost_name = '$|$Cost$-d|$'

    df = load_data(path)
    df[abs_cost_name] = ((df["CostAverage"] - 2.0)).abs().ewm(alpha=0.05, adjust=True).mean()
    # df = df[df['$K_I$'] != 1.0]

    hue_src = "$K_I$"
    col_src = "$K_P$"

    rows = ['IsGoalReachedAverage', 'CostAverage', abs_cost_name, 'costPenaltyAverage']
    nrows = len(rows)
    cols = df[col_src].unique()
    cols.sort()
    ncols = len(cols)
    
    ncolors = df[hue_src].nunique()

    palette_name='crest'
    palette = set_plotting_theme(ncolors, palette_name=palette_name)

    fig, ax = plt.subplots(nrows, ncols, figsize=(5.84036, 6.5), sharey='row')

    for rowidx, colidx in product(range(nrows), range(ncols)):
        row = rows[rowidx]
        col = cols[colidx]
        data = df[df[col_src] == col]

        current_ax: plt.Axes = ax[rowidx][colidx]
        sns.lineplot(
            data=data,
            hue=hue_src,
            x="Diagnostics/CumSteps",
            y=row,
            ax=ax[rowidx][colidx],
            estimator='mean',
            ci=None,
            palette=palette,
            legend=False,
            linestyle='-',
        )

        if row == 'CostAverage':
            current_ax.set_ylim(bottom=0.0)
            current_ax.axhline(2.0, linestyle=":", color="black")

        if row == 'IsGoalReachedAverage':
            current_ax.set_ylim(bottom=0.0, top=1.0)

        if colidx > 0:
            current_ax.set_ylabel("")
        else:
            nice_y_label = dict(
                IsGoalReachedAverage = 'Goal Rate',
                CostAverage = "Cost",
                costPenaltyAverage = "$\lambda$"
            )
            c = current_ax.get_ylabel()
            current_ax.set_ylabel(nice_y_label.get(c, c))

        if rowidx == 0:
            current_ax.set_title(f'{col_src} = {col}')
        if rowidx < len(rows) - 1:
            current_ax.set_xlabel("")
        else:
            current_ax.set_xlabel("Steps")
    
    savepath = pathlib.Path("/home/pillmayerc/mth/plots/cr_tune_pi_aio.pdf")
    fig.tight_layout()
    fig.savefig(savepath)

    legend_path = savepath.parent / f"{savepath.stem}_legend.pdf"
    palette = set_plotting_theme(len(palette), fontsize=10, palette_name=palette_name)

    fig, ax = plt.subplots(figsize=(5.84036, 0.4))
    patches = []
    variants = df[hue_src].unique()
    variants.sort()
    for i, variant in enumerate(variants):
        p = mlines.Line2D([0],[0],color=palette[i], label=f'{hue_src} = {variant}', lw=1.5)
        # p = mpatches.Patch(color=palette[i], label=f'{hue_src} = {variant}')
        patches.append(p)
    l = ax.legend(handles=patches, ncol=df[hue_src].nunique(), loc="center", frameon=False)
    plt.axis('off')
    plt.savefig(legend_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", help="Plot from this directory", type=pathlib.Path, default='/home/pillmayerc/mth/data_to_keep/00_thesis_data/cr_pi_all_in_one_new')
    args = parser.parse_args()

    plot_from_dir(
        pathlib.Path(args.d)
    )
