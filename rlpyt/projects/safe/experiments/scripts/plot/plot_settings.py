"""
    Contains shared plotting logic such as:
    - Font
    - TUM CI Colors
"""

import matplotlib as mpl

import seaborn as sns
import matplotlib.pyplot as plt

TUM_BLUE = "#0065bd"
TUM_DARKBLUE = "#005293"
TUM_LIGHTBLUE = "#64a0c8"
TUM_LIGHTERBLUE = "#98c6ea"
TUM_ORANGE = "#e37222"
TUM_GREEN = "#a2ad00"

PALETTE_TUM_BLUES = [TUM_BLUE, TUM_LIGHTBLUE, TUM_DARKBLUE, TUM_LIGHTERBLUE]
PALETTE_TUM_DEEP = [TUM_BLUE, TUM_GREEN, TUM_ORANGE, TUM_LIGHTBLUE]
PALETTE_PAPER_SPECIAL = []

def set_plotting_theme(color_amount: int, fontsize=10, palette_name="deep", font="Palatino"):
    """Sets the plotting style and returns the color palette"""
    if palette_name == 'TUM_BLUES':
        palette = sns.color_palette(PALETTE_TUM_BLUES)
    elif palette_name == 'TUM':
        palette = sns.color_palette(PALETTE_TUM_DEEP)
    elif palette_name == "deep_r":
        palette = sns.color_palette("deep", color_amount)
        palette = list(reversed(palette))
    else:
        palette = sns.color_palette(palette_name, color_amount)
    sns.set_theme(style="ticks", palette=palette)

    # plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
    # Available fonts: https://matplotlib.org/stable/tutorials/text/usetex.html
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "Palatino",
        }
    )

    plt.rc('font', size=fontsize)          # controls default text sizes
    plt.rc('axes', titlesize=fontsize)     # fontsize of the axes title
    plt.rc('axes', labelsize=fontsize)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize)    # fontsize of the tick labels
    plt.rc('legend', fontsize=fontsize)    # legend fontsize
    plt.rc('figure', titlesize=fontsize)  # fontsize of the figure title
    mpl.rcParams['axes.linewidth'] = 0.8
    mpl.rcParams['xtick.major.width'] = 0.8
    mpl.rcParams['ytick.major.width'] = 0.8
    COLOR = 'black'
    mpl.rcParams['text.color'] = COLOR
    mpl.rcParams['axes.labelcolor'] = COLOR
    mpl.rcParams['axes.edgecolor'] = COLOR
    mpl.rcParams['xtick.color'] = COLOR
    mpl.rcParams['ytick.color'] = COLOR

    return palette

def get_nice_height(width_in=5.84036, fraction=1):
    
    fig_width_in = width_in * fraction

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    return (fig_width_in, fig_height_in)
