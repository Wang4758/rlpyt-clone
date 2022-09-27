"""
    Creates the KDE plots and CVAR cost tables for evaluating worst-case agents with different alphas
"""

import argparse
import ast
import itertools
import json
import math
import pathlib
import re
from argparse import ArgumentParser
from itertools import chain
from tkinter.filedialog import askopenfile
from typing import Dict, List, Union
import gzip

import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame
from rlpyt.projects.safe.experiments.scripts.plot.plot_settings import get_nice_height, set_plotting_theme
import seaborn as sns


def get_nth_parent(path: pathlib.Path, n):
    for _ in range(n):
        path = path.parent
    return path


def load_datasets(
    input_dir: pathlib.Path,
    drop_actions=True,
    glob_pattern="**/eval_results*.json*",
    skip_pattern=None,
    worst_case_fraction=0.1,
):

    dfs = []
    dfs_worst = []
    cost_evals = []

    for elem in input_dir.glob(glob_pattern):

        if "plot_exclude" in elem.as_posix():
            print("Skipping", elem.name)
            continue

        if skip_pattern:
            if re.match(skip_pattern, elem.name):
                print("Skipping", elem.name)
                continue
        
        print("Reading", elem.as_posix())
        variant_name = elem.parent.name

        m = re.match(r'eval_results(\d+)_new', elem.name.replace('.json.gz', ''))
        run_id = int(m.group(1))

        # quick fix for the starter agents wcsac implementation...
        if 'wc_alpha' not in variant_name:
            variant_name = elem.parent.parent.name
            
            if "wc_alpha" not in variant_name:
                # not wc training...
                variant_name = elem.parent.name

        if elem.suffix == ".gz":
            with gzip.open(elem, "rb") as f:
                df = pd.read_json(f)
        elif elem.suffix == ".json":
            df = pd.read_json(elem)
        else:
            raise RuntimeError(f"Unexpected suffix caught in glob {elem.suffix}")

        if "Actions" in df.columns and drop_actions:
            df.drop(columns=["Actions"], inplace=True)

        if "wc_alpha" in variant_name:
            alpha = float(variant_name.replace('wc_alpha', ''))
            variant_name = r"$\alpha = " + f"{alpha}" + r"$"
        else:
            alpha = 1.0

        cost_evals.append(eval_costs(df, alpha))

        df["Variant"] = variant_name
        df["RunId"] = run_id
        if "wc_alpha" in variant_name:
            df["Variant"] = pd.Categorical(
                df["Variant"], [r"$\alpha = 1.0$", r"$\alpha = 0.9$", r"$\alpha = 0.5$", r"$\alpha = 0.1$"]
            )

        if worst_case_fraction < 0.001:
            print(f'Automatic worst case fraction {alpha}')
            load_fraction = alpha
        else:
            load_fraction = worst_case_fraction


        # Select worst 10% or 20% of trajectories to better see the effect by removing many of the 0 cost examples
        n_rows = len(df.index)
        dfw = (
            df.sort_values("Cost").tail(int(n_rows * load_fraction)).reset_index()
        )

        dfs.append(df)
        dfs_worst.append(dfw)

    df = pd.concat(dfs)
    dfw = pd.concat(dfs_worst)
    dfc = pd.concat(cost_evals)
    dfw.reset_index()
    df.reset_index()
    dfc.reset_index()

    # See how many violations per category there are:
    costs = df[['Cost', 'Variant', 'BenchmarkId', "RunId"]].copy()
    costs['Violated'] = df['Cost'] > 2
    costs.drop(columns=['BenchmarkId', 'Cost', 'RunId'], inplace=True)
    #print(costs.groupby(['Variant']).size())
    violations_count = costs.groupby(['Variant', 'Violated']).size()
    violations_count = violations_count.reset_index().rename(columns={0: 'Counts'})
    violations_count['Relative'] = violations_count['Counts'] / 2673
    #print(violations_count.round(2))


    dfc = dfc.groupby('alpha').mean().sort_values('alpha', ascending=False)

    return df, dfw, dfc, violations_count

def eval_costs(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    
    print(df.reset_index(drop=True))
    
    cost_levels = [1.0, .9, .5, .1]
    cost_names = ['EC', 'C0.9', 'C0.5', 'C0.1']
    costs = []

    n_rows = len(df.index)
    for cost_level in cost_levels:
        v = df.sort_values("Cost")["Cost"].tail(int(n_rows * cost_level)).mean()
        costs.append(v)

    res = pd.DataFrame(data={n: [v] for n,v in zip(cost_names + ['alpha'], costs + [alpha])})
    if "IsGoalReached" in df.columns:
        res['GR'] = df["IsGoalReached"].mean()
    else:
        res['ER'] = df["Return"].mean()
    return res


def create_termination_table(df: pd.DataFrame, save_path: pathlib.Path):
    grouped = df.groupby("Variant")
    print(grouped.count())
    averages = grouped.mean()
    averages.to_latex(save_path)
    print(averages)


def plot(df: pd.DataFrame, save_path: pathlib.Path, hist_x_limits=None):
    set_plotting_theme(df["Variant"].nunique(), palette_name='crest')

    fig, ax = plt.subplots(
        1, 1, 
        figsize=get_nice_height()
    )
    ax = [ax]

    remove_cols = set(df.columns) - {"Cost", "Variant"}
    df = df.drop(columns=list(remove_cols)).reset_index().sort_values("Variant", ascending=False)

    # sns.histplot(
    #     ax=ax[0],
    #     data=df,
    #     x="Cost",
    #     hue="Variant",
    #     multiple="dodge",
    #     # shrink=.8,
    #     # element='poly',
    #     kde=True,
    #     stat="density",
    # )

    sns.kdeplot(
        ax=ax[0],
        data=df,
        x="Cost",
        hue="Variant",
        # cumulative=True,
        fill=True,
        linewidth=1,
        alpha=.5,
    )

    if hist_x_limits:
        ax[0].set_xlim(left=hist_x_limits[0], right=hist_x_limits[1])
    else:
        ax[0].set_xlim(left=0, right=100)

    # sns.boxplot(ax=ax[1], data=df, x="Variant", y="Cost", showfliers=True, flierprops=dict(marker='+'))

    plt.tight_layout()

    plt.savefig(save_path.as_posix())


def main(
    input_dir: pathlib.Path,
    output_dir,
    glob_pattern,
    skip_pattern,
    outfile_prefix,
    worst_case_fraction,
    hist_x_limits
):

    df, df_worst, dfc, violations_count = load_datasets(
        input_dir,
        glob_pattern=glob_pattern,
        skip_pattern=skip_pattern,
        worst_case_fraction=worst_case_fraction,
    )

    violations_count.to_latex(output_dir.joinpath(outfile_prefix + 'violations.tex'), float_format='%.2f', index=False)
    dfc.to_latex(output_dir.joinpath(outfile_prefix + 'cost_table.tex'), float_format='%.2f')

    plot(df_worst, output_dir.joinpath(outfile_prefix + "costplot.pdf"),hist_x_limits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-d", type=pathlib.Path)
    parser.add_argument("--output_dir", "-o", type=pathlib.Path)
    parser.add_argument("--glob_pattern", default="**/eval_results*_new.json*")
    parser.add_argument("--skip_pattern", default=None)
    parser.add_argument("--outfile_prefix", type=str, default="")
    parser.add_argument("--worst_case_fraction", type=float, default=.05)
    parser.add_argument("--hist_x_limits", default=None, nargs="+", type=float)

    args = parser.parse_args()
    args = args.__dict__
    args["output_dir"] = args.get("output_dir", None) or args["input_dir"]

    main(**args)
