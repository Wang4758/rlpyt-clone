""" Main plotting script for training graphs. 

    Saves legend separately.
"""

import ast
import itertools
import json
import math
import pathlib
import re
from argparse import ArgumentParser
from itertools import chain
from typing import Dict, List, Union
import logging
from functools import cmp_to_key

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from rlpyt.projects.safe.experiments.scripts.plot.plot_settings import set_plotting_theme
import seaborn as sns

COST_LIMITS = set()
COST_SCALES = set()
logging.basicConfig(format='[%(levelname)s]:%(message)s')
LOGGER = logging.getLogger(name='plotter')
LOGGER.setLevel('INFO')

VARIANT_NAME_MAP = {
    "cr_cpo00_new": "CPO",
    "Falseclipcosto0.5costema": "$EMA_C$=0.5",
    "Falseclipcosto0costema": "$EMA_C$=0",
    "Trueclipcosto0.5costema": "PPO-$J_C$, $EMA_C$=0.5",
    "Trueclipcosto0costema": "PPO-$J_C$, $EMA_C$=0",
    "7.5clcrm_dgae_0.9wc_7.5cl": r"$\alpha=0.9, c_{\mathrm{limit}}=7.5$",
}


def camel_case_split(identifier):
    matches = re.finditer(
        ".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", identifier
    )
    return [m.group(0) for m in matches]


def niceify_variant(variant_name):

    # print(f"niceify {variant_name}")

    if variant_name in VARIANT_NAME_MAP:
        return VARIANT_NAME_MAP.get(variant_name)

    m = re.match(r"^(\d+(\.\d+)?)Kp0pi$", variant_name)
    if m:
        return f"{m.group(1)} Kp"

    m = re.match(r"^(\d+)Kp(\d+\.\d+)Ki$", variant_name)
    if m:
        return f"{m.group(1)} Kp {m.group(2)} Ki"

    m = re.match(r"(\d+(\.\d+)?)wc_alpha(.*)", variant_name)
    if m:
        if not ('wcsac' in variant_name or 'ours' in variant_name):
            res = f'$\\alpha={m.group(1)}$'
        else:
            if 'wcsac' in variant_name:
                res = f'$\\alpha={m.group(1)}$ (WCSAC)'
            else:
                res = f'$\\alpha={m.group(1)}$ (ours)'

        return res 

    return variant_name

def squeeze_list(ll):
    return [item for sublist in ll for item in sublist]

def get_nice_name(s: str):
    nicename = s.capitalize()
    if 'c_var_valueAverage' == s:
        nicename = 'CostVariance Value'
    elif s.endswith("Average"):
        nicename = s[: -len("Average")]
        parts = squeeze_list([x.split('_') for x in camel_case_split(nicename)])
        nicename = " ".join((part.capitalize() for part in parts))
    elif s == "Diagnostics/CumSteps":
        nicename = "Steps"
    elif s == "NViolations":
        nicename = r"\#Violations"
    return nicename


def search_dict(d: dict, key: str):
    if key in d:
        return d[key]
    else:
        for value in filter(lambda v: isinstance(v, dict), d.values()):
            res = search_dict(value, key)
            if res:
                return res


def get_nth_parent(path: pathlib.Path, n):
    for _ in range(n):
        path = path.parent
    return path

def should_be_excluded(path: pathlib.Path):
    return 'plot_exclude' in path.absolute().as_posix()

def find_datasets(
    path: pathlib.Path, variant_name_depth=2, starter_agents=False, include_path=False
) -> Dict:
    datasets = {}
    
    for progress_file_path in path.glob('**/progress.csv'):
        if should_be_excluded(progress_file_path): # or 'run_1' in progress_file_path.as_posix():
            LOGGER.info(f'Excluding {progress_file_path.as_posix()}')
            continue

        variant_name = ""
        for ip in range(variant_name_depth):
            pn = get_nth_parent(progress_file_path.parent, ip + 1).name
            if "commonroad" in pn:
                break
            variant_name += pn

        # variant_name += progress_file_path.parent.stem

        with open(progress_file_path.parent.joinpath("params.json"), "r") as f:
            config = json.load(f)
            cost_limit = search_dict(config, "cost_limit")
            cost_scale = search_dict(config, 'cost_scale')
            COST_LIMITS.add(cost_limit)
            COST_SCALES.add(cost_scale)

        if variant_name == "":
            variant_name = search_dict(config, 'id')

        variant_name = niceify_variant(variant_name)

        if variant_name not in datasets:
            datasets[variant_name] = []
        datasets[variant_name].append(progress_file_path)

    return datasets

def convert_tensorboard_naming(df: pd.DataFrame):
    if "Cost/Average" not in df.columns and "CostAverage" in df.columns:
        # assume normal format
        return df
    rename_dict = dict()
    for col in df.columns:
        if "Diagnostics" not in col:
            rename_dict[col] = col.replace('/','')
    
    LOGGER.info(f"Converting {len(rename_dict)} columns from tensorboard naming.")
    return df.rename(columns=rename_dict)

def ensure_costrate_present(df: pd.DataFrame, timestepsName):
    if 'costrate' not in df.columns:
        if not ("cumCostAverage" in df.columns):
            LOGGER.warning("computing cumCostAverage from costAverage")
            df["cumCostAverage"] = df["CostAverage"].cumsum()
        df["costrate"] = df["cumCostAverage"] / df[timestepsName]
    
    return df.reset_index()

def apply_smoothing(df: pd.DataFrame, metrics: list, ewm: float = None, smooth: int = 0):
    df = df.copy()
    for metric in metrics:
        if metric not in df.columns:
            LOGGER.info(f"WARNING: Metric {metric} not found. Filling with zeros")
            df[metric] = float('NaN')
        elif ewm is not None:
            # ewm overrides normal smoothing...
            # use tensorboard setting for smoothing factor
            df[metric] = df[metric].ewm(alpha=1.0 - ewm, adjust=True).mean()
        elif (smooth or 1) > 1:
            df[metric] = df[metric].rolling(smooth, min_periods=2, center=True).mean()
    return df

def clamp_delta(df: pd.DataFrame):
    df = df.copy()
    if 'deltaAverage' in df:
        df["deltaAverage"] = df["deltaAverage"].clip(lower = 0.0, upper=None)
    return df

def load_datasets(
    datasets: Dict,
    metrics: List,
    timestepsName="Diagnostics/CumSteps",
    smooth=None,
    ewm=None,
) -> pd.DataFrame:

    dfs = []

    for variant_name, results in datasets.items():
        for result in results:
            with open(result, "r") as csv:
                
                is_csv = result.as_posix().endswith(".csv")
                df = pd.read_csv(csv, delimiter="," if is_csv else "\t")
                
                df = convert_tensorboard_naming(df)
                df = ensure_costrate_present(df, timestepsName)
                df = clamp_delta(df)
                df = apply_smoothing(df, metrics, ewm, smooth)

                df["Variant"] = variant_name # + result.parent.name
                if "cumCostMax" not in df:
                    df["cumCostMax"] = np.nan
                dfs.append(df[metrics + [timestepsName, "cumCostMax", "Variant"]])

    big_s_df = pd.concat(dfs).reset_index()
    return big_s_df


def plot_from_dir(
    path: pathlib.Path,
    output_path: Union[pathlib.Path, None],
    metrics: List,
    xsource: str,
    smooth=25,
    show_costrate=False,
    legend_pos=(0, 0),
    plots_per_row=3,
    y_limits=dict(),
    show_variance=False,
    show_legend=False,
    palette_name="tab10",
    variant_name_depth=2,
    variant_sort_fun=None,
    line_styles: Dict =dict(),
    linewidth=2.5,
    linestyle='-',
    ewm=None,
    column_aliases=dict(),
    legend_fontsize=10,
    legend_cols=1,
    use_min_x_limit=False
):

    assert not show_costrate, "Obsolete settings, pls use costrate as a metric. It is always calculated."

    use_line_styles = len(line_styles) > 0

    LOGGER.info("Gathering datasets...")
    datasets = find_datasets(path, variant_name_depth=variant_name_depth)
    if len(datasets) == 0:
        LOGGER.warning(f'Directory {path.as_posix()} is empty')
    [LOGGER.info(f"{vn}: {len(r)} runs") for vn, r in datasets.items()]
    if len(COST_LIMITS) > 1:
        LOGGER.warning(f"found more than 1 cost limits: {COST_LIMITS}")
    df = load_datasets(datasets,metrics,timestepsName=xsource, smooth=smooth, ewm=ewm)
    if use_line_styles:
        # the style should just be some identifier (seaborn will create the actual line styles after the groups...)
        df['Style'] = df['Variant'].apply(lambda v: line_styles.get(v, 0))

    variants_ordered = list(sorted(df["Variant"].unique()))
    if variant_sort_fun is not None:
        variants_ordered.sort(key = variant_sort_fun)
        
    n_variants = len(variants_ordered)

    df.drop(columns=["index"], inplace=True)

    palette = set_plotting_theme(n_variants, palette_name=palette_name)

    plots_sum = len(metrics)
    plots_x = min(plots_sum, plots_per_row)
    plots_y = math.ceil(plots_sum / plots_x)

    if use_min_x_limit:
        tmp = df[["Variant", xsource]].groupby("Variant").max().reset_index()
        min_x = tmp[xsource].min()

    if plots_y == 1:
        graph_width = 5.84036 / 3
        graph_height = graph_width * 0.9
        fig, axes = plt.subplots(
            plots_y, plots_x, figsize=(plots_x * graph_width, plots_y * graph_height + 0.25), sharex=True
        )
    else:
        graph_width = 5.84036 / 3
        graph_height = graph_width * 0.75
        fig, axes = plt.subplots(
            plots_y, plots_x, figsize=(plots_x * graph_width, plots_y * graph_height + 0.4), sharex=True
        )

    for idx, metric in enumerate(metrics):

        row, col = idx // plots_per_row, idx % plots_per_row
        ax = axes[row, col] if plots_y > 1 else axes[idx]

        sns.lineplot(
            ax=ax,
            data=df,
            x=xsource,
            y=metric,
            hue="Variant",
            hue_order=variants_ordered,
            legend=(legend_pos == (row, col)),
            estimator='mean',
            ci="sd" if show_variance else None,
            linestyle = linestyle,
            linewidth = linewidth,
            alpha = 1.0
        )

        if metric in column_aliases:
            yname = column_aliases.get(metric)
        else:
            yname = get_nice_name(metric)
        ax.set(ylabel=yname, xlabel=get_nice_name(xsource))
        
        if not show_legend:
            ax.legend().remove()

        ax.set_xlim(left = 0.0, right = df[xsource].max())

        if metric == "CostAverage":
            draw_clims = [min(COST_LIMITS)]
            for cost_limit in draw_clims:
                if cost_limit < 500:
                    ax.axhline(cost_limit, linestyle=":", color="black")
            ax.set_ylim(bottom=0.0)
            pass
        elif metric == "CostSparseAverage":
            ax.axhline(2.0, linestyle=":", color="black")
            ax.set_ylim(bottom=0.0)
            pass
        elif metric == 'ep_cost_cvarAverage':
            cost_scale = min(COST_LIMITS) / min(COST_SCALES)
            ax.axhline(cost_scale, linestyle=":", color="black")
            ax.set_ylim(bottom=0.0, top=10)
        elif metric == 'ep_cost_varAverage':
            # ax.set_ylim(bottom=0.0, top=50)
            pass
        elif metric == "deltaAverage":
            ax.set_ylim(top=5, bottom=-1)
            ax.axhline(0, linestyle=":", color="black")
            # ax.axhline(0.5, linestyle=":", color="black")
            # ax.axhline(1.0, linestyle=":", color="black")
        elif metric == "costPenaltyAverage":
            b, t = ax.get_ylim()
            ax.set_ylim(top=min(t, 50), bottom=max(b, 0))
            # ax.set_ylim(top = .5, bottom=-.1)
            # ax.set_yscale('log')
            pass
        elif metric == "IsGoalReachedAverage" and not metric in y_limits:
            ax.set_ylim(top=1.0, bottom=0.0)
            ax.axhline(0.9, linestyle=":", color="black")
        elif "NViolationsRG" in metric:
            ax.set_ylim(bottom=0, top=10)

        if metric in y_limits:
            LOGGER.info(f"Y-Limits: {y_limits}")
            ax.set_ylim(**y_limits[metric])

        if use_min_x_limit:
            ax.set_xlim(left=0, right=min_x)
        
    
    fig.tight_layout()
    savepath = path.joinpath(f"plot.pdf" if output_path is None else output_path)
    LOGGER.info(f'✍️  {savepath.as_posix()}')
    fig.savefig(savepath, bbox_inches='tight', pad_inches=0.1)


    # save legend separatly
    
    legend_path = savepath.parent / f"{savepath.stem}_legend.pdf"
    palette = set_plotting_theme(n_variants, fontsize=legend_fontsize, palette_name=palette_name)

    fig, ax = plt.subplots(figsize=(5.84036, 0.4))
    patches = []
    for i, variant in enumerate(variants_ordered):
        l = mlines.Line2D([0],[0],color=palette[i], label=niceify_variant(variant), lw=1.5)
        # p = mpatches.Patch(color=palette[i], label=niceify_variant(variant))
        patches.append(l)
    l = ax.legend(handles=patches, ncol=n_variants if legend_cols == -1 else legend_cols, loc="center", frameon=False)
    plt.axis('off')
    plt.savefig(legend_path)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", nargs='+', help="Plot from this directory")
    parser.add_argument("-o", help="Save path", default="")
    parser.add_argument(
        "-m",
        "--metrics",
        help="Metrics to plot",
        nargs="+",
        default=["ReturnAverage", "CostAverage"],
    )
    parser.add_argument(
        "--xsource",
        "-x",
        help="Column to use for x axis",
        default="Diagnostics/CumSteps",
    )
    parser.add_argument("--smooth", default=25, type=int)
    parser.add_argument("--show_costrate", default="False", type=str)
    parser.add_argument("--legend_pos", default=(-1, -1), nargs="+", type=int)
    parser.add_argument("--plots_per_row", default=3, type=int)
    parser.add_argument("--y_limits", default="", type=str)
    parser.add_argument("--show_variance", default="True", type=str)
    parser.add_argument("--show_legend", default="False", type=bool)
    parser.add_argument("--palette_name", default="deep", type=str)
    parser.add_argument("--variant_name_depth", default=2, type=int, help="This is used to automatically name the runs based on the parent folders.")
    parser.add_argument("--extra_label_aliases", default="", type=str, help="Supply aliases for variants as a dictionary")
    parser.add_argument("--column_aliases", default="", type=str, help="Supply aliases for columns as a dictionary")
    parser.add_argument("--variant_sort_fun", default="", type=str, help="Can be used to improve the sorting of variants. See code for usage.")
    parser.add_argument("--line_styles", default="", type=str)
    parser.add_argument("--linewidth", '-lw', default=1.5, type=float)
    parser.add_argument("--linestyle", '-ls', default='-', type=str)
    parser.add_argument("--ewm", default=None, type=float, help="Set to [0,1] to smooth instead of using --smooth. It works like in tensorboard.")
    parser.add_argument("--legend_fontsize", default=10, type=int)
    parser.add_argument("--use_min_x_limit", default=False, type=bool)
    parser.add_argument("--legend_cols", default=-1, type=int)

    args = parser.parse_args()

    assert len(args.legend_pos) == 2, "Legend position needs to be exactly 2 values!"

    ylimits = dict() if args.y_limits == "" else ast.literal_eval(args.y_limits)
    line_styles = dict() if args.line_styles == '' else ast.literal_eval(args.line_styles)

    extra_label_aliases = dict() if args.extra_label_aliases == "" else ast.literal_eval(args.extra_label_aliases)
    VARIANT_NAME_MAP.update(extra_label_aliases)

    column_aliases = {
        "IsGoalReachedAverage": "Goal-Reaching Rate",
        "IsOffroadAverage": "Off-Road Rate",
        "IsCollisionAverage": "Collision Rate",
        "LengthAverage": "Episode Length",
        "deltaAverage": r"$(\mathrm{CVaR}_\mathrm{ep} - d)_+$",
        "costPenaltyAverage": r"$\lambda$",
        "CostSparseAverage": r"NViolations"
    }
    column_aliases_arg = dict() if args.column_aliases == "" else ast.literal_eval(args.column_aliases)
    assert isinstance(column_aliases_arg, dict)
    column_aliases.update(column_aliases_arg)

    variant_sort_fun = None if args.variant_sort_fun == "" else eval(args.variant_sort_fun)

    

    for d in args.d:
        source_path = pathlib.Path(d)
        if not source_path.is_dir():
            continue
        LOGGER.info(f'Plotting from {source_path.as_posix()}')
        plot_from_dir(
            pathlib.Path(d),
            output_path = pathlib.Path(args.o) if args.o != "" else None,
            metrics=args.metrics,
            xsource=args.xsource,
            smooth=args.smooth,
            show_costrate=eval(args.show_costrate),
            legend_pos=tuple(args.legend_pos),
            plots_per_row=args.plots_per_row,
            y_limits=ylimits,
            show_variance=eval(args.show_variance),
            show_legend=args.show_legend,
            palette_name=args.palette_name,
            variant_name_depth=args.variant_name_depth,
            variant_sort_fun=variant_sort_fun,
            line_styles=line_styles,
            linewidth = args.linewidth,
            linestyle = args.linestyle,
            ewm = args.ewm,
            column_aliases=column_aliases,
            legend_fontsize=args.legend_fontsize,
            use_min_x_limit=args.use_min_x_limit,
            legend_cols=args.legend_cols
        )
