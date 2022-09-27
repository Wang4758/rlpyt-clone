""" Script used to evaluate the initial action distribution of our method. """

from io import StringIO
import pathlib
from matplotlib import gridspec
from rlpyt.projects.safe.dcppo_model_v2 import DCppoModelV2
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import gym
from collections import namedtuple
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import gym_monitor.monitor_env

from rlpyt.projects.safe.experiments.scripts.plot.plot_settings import TUM_BLUE, TUM_LIGHTBLUE, get_nice_height, set_plotting_theme
from rlpyt.projects.safe.dcppo_model import DCppoModel
from rlpyt.projects.safe.dcppo_agent import DCppoAgent


def sample_model():
    
    env = gym.make(
        'cr-monitor-v0', 
        max_problems = 300,
        preload_curvi_states = True, 
        active_rules=["R_G1", "R_G2", "R_G3"],
        # dont forget the filter! otherwise rule 3 results will be messed up!
        scenario_filter_file_path = '/home/pillmayerc/mth/rlpyt/rlpyt/projects/safe/experiments/scripts/cr_util/results_R_G1R_G2R_G3_new_a_obs_ALLokayids.txt',
    ) # observe_robustness=True,)
    obs_space = env.observation_space
    action_space = env.action_space

    obs_shape = obs_space.sample().shape
    agent = DCppoAgent(
        ModelCls=DCppoModelV2,
        model_kwargs = dict(
            # observation_shape = obs_shape,
            # action_size=action_space.sample().shape[0],
            hidden_sizes=[128, 128],
            normalize_observation=True,
            var_clip=1e-6,
            init_log_std=-1.386,
            a_long_shift=0.4
        )
    )
    env_space_cls = namedtuple("EnvSpaces", ["observation", "action"])
    c = env_space_cls(**{"observation": obs_space, "action": action_space})
    agent.initialize(c)
    
    # get sample actions
    n_actions = 32 * 256
    n_warmup = 500
    samples = np.zeros(shape=(n_actions, 2 + 3))
    o = env.reset()
    for i in tqdm(range(n_actions + n_warmup), desc="Sampling actions"):
        
        if i >= 0:
            act_pyt, agent_info = agent.step(torch.from_numpy(o).float(), None, None)
            dist_info = agent_info.dist_info
            mean, log_std = dist_info.mean, dist_info.log_std
            #print(mean, log_std)
            action = act_pyt.numpy()
            action = np.clip(action, a_min=action_space.low, a_max=action_space.high)
            absolute_acc = action[0] ** 2 + action[1] ** 2
            parameters_longitudinal_a_max = 11.5
            if absolute_acc > parameters_longitudinal_a_max ** 2:
                rescale_factor = (parameters_longitudinal_a_max - 1e-6) / np.sqrt(absolute_acc)
                action[0] *= rescale_factor
                action[1] *= rescale_factor
        else:
            action = action_space.sample()


        o, _, d, info = env.step(action)

        # if warmup update obs_rms,
        # otherwise save action and dont update obs_rms
        if i >= n_warmup:
            samples[i - n_warmup] = np.concatenate([action, [info.get("R_G1"), info.get("R_G2"), info.get("R_G3")]])
        else:
            agent.update_obs_rms(torch.from_numpy(o).float())

        if d:
            o = env.reset()

    # there is some rescaling for PM action but only multiplication no addition and we
    # dont really care about the magnitude for now, if we do perform the same rescaling here
    df = pd.DataFrame(samples, columns=["a_long", "a_lat", "R_G1", "R_G2", "R_G3"])
    print(df)
    df.to_csv("a_dist_new_model_with_shift.csv")
    

def plot(csv: pathlib.Path, paper=False):

    csv = pathlib.Path(csv)

    palette = set_plotting_theme(3, palette_name="deep" if not paper else "tab10", fontsize= 10 if not paper else 14)

    df: pd.DataFrame = pd.read_csv(csv)
    df.drop(columns=["Unnamed: 0"], inplace=True)
    # df.drop(df[df.abs().a_long >= 0.99].index, inplace=True)
    df.reset_index(drop=True)

    # to make table without robustness, only mtl
    io = StringIO()
    df_mtl = df.copy()
    df_mtl["a_long"] *= 11.5
    df_mtl['brake_abrupt'] = df_mtl["a_long"] <= -2.0
    df_mtl['R_G2_Violated'] = df_mtl["R_G2"] < 0.0
    df_mtl['R_G1_Violated'] = df_mtl["R_G1"] < 0.0
    df_mtl['R_G3_Violated'] = df_mtl["R_G3"] < 0.0
    df_mtl = df_mtl[["brake_abrupt", "R_G1_Violated", "R_G2_Violated", "R_G3_Violated"]]
    mtl_based_table = df_mtl.groupby("brake_abrupt").mean().round(2).reset_index()

    mtl_based_table.rename(columns={
        "brake_abrupt": r"a_long <= -2.0",
        "R_G1_Violated": "R_G1 Violated",
        "R_G2_Violated": "R_G2 Violated",
        "R_G3_Violated": "R_G3 Violated",
    }, inplace=True)

    mtl_based_table.to_latex(io, index=False)
    print(io.getvalue())

    df = pd.melt(df, id_vars=["a_long"], value_vars=["R_G1", "R_G2", "R_G3"], var_name="Rule", value_name=r"$\rho$")

    a_long_name = r"$a_\mathrm{long}$"
    df.rename(columns={"a_long": a_long_name}, inplace=True)

    df[a_long_name] *= 11.5
    # df["a_lat"] *= 11.5

    custom_grid_plot(df, a_long_name)

    # g: sns.JointGrid = sns.jointplot(
    #     data=df,
    #     x = r"$\rho$",
    #     hue = "Rule",
    #     y = a_long_name,
    #     height=5,
    #     # color="dodgerblue",
    #     palette=palette,
    #     ratio=5,
    #     # marginal_ticks=True,
    #     marginal_kws=dict(fill = True),
        
    #     kind="scatter",
    #     # fill=False,
    #     # alpha=0.8,
    #     # levels=5,
    # )

    # g.figure.set_figwidth(fw)
    # g.figure.set_figheight(fh)

    # sns.move_legend(g.ax_joint, "upper left", bbox_to_anchor=(0., 1.), ncol=1, title=None, frameon=False)

    # g.ax_joint.axhline(-2.0, linestyle="-.", color=".1", linewidth=1.0, xmin=0.0, xmax=0.95)
    # g.ax_joint.text(-1.2,-1.65,r"$-2$")

    # g = sns.scatterplot(data=df, x="R_G2", y="a_long", color="black", marker=".")
    # plt.axhline(y = -2.0, linestyle = "--", color="dodgerblue")
    plt.tight_layout()
    fname = csv.stem + ('_paper' if paper else '') + '.pdf'
    fparent = csv.parent
    plt.savefig(fparent / fname)

def custom_grid_plot(df, a_long_name):
    fw, fh = get_nice_height()

    df_all = df.copy()
    df = df.sample(frac = 0.5)
    df = df.sort_values("Rule", ascending=True)

    fig = plt.figure(figsize=(fw,fh))
    gs = GridSpec(2,3)
    axes = [
        fig.add_subplot(gs[0,0]),
        fig.add_subplot(gs[0,1]),
        fig.add_subplot(gs[0,2])
    ]

    for ax, rule in zip(axes, df["Rule"].unique()):
        query: pd.DataFrame = df[df["Rule"] == rule]
        query = query.sample(frac=0.25)
        ax.set_title(rule)
        print(query)
        sns.scatterplot(data=query, y = r"$\rho$", x=a_long_name, color=TUM_BLUE, ax = ax, marker="o")
        if rule != "R_G1":
            ax.set_ylabel("")
        if rule == "R_G2":
            ax.axvline(-2.0, color="black", linestyle="--", lw=1.0)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(-12, 12)

    a_dist_ax = fig.add_subplot(gs[1,:])
    sns.histplot(
        data=df_all,
        x=a_long_name,
        fill=True,
        ax=a_dist_ax,
        stat="percent",
        bins=25,
        color=TUM_BLUE
    )
    a_dist_ax.set_title("Longitudinal acceleration distribution")

def check_distribution(csv):
    df: pd.DataFrame = pd.read_csv(csv)
    df.drop(columns=["Unnamed: 0"], inplace=True)
    df["a_long"] *= 11.5
    df["a_lat"] *= 11.5
    a_abrupt_count = len(df[df["a_long"] <= -2.0])
    a_abrupt_percent = a_abrupt_count / len(df)
    print(a_abrupt_percent)
    # hist = sns.histplot(data = df, x="a_long", cumulative=True, bins=100, stat="probability")
    df = df.melt(value_vars=["a_long", "a_lat"], value_name="acceleration", var_name="variable")
    hist = sns.kdeplot(data = df, x = "acceleration", hue="variable")
    # plt.axvline(-2.0)
    # plt.axhline(0.5)
    plt.savefig("a_long_histplot.pdf")

def main():
    # sample_model()
    # check_distribution("a_dist_shift_0_4.csv")
    # plot("a_dist_shift_0_4.csv", paper=False)
    # plot("a_dist_new_model_with_shift.csv", paper=False)
    plot("a_dist.csv", paper=False)
    check_distribution("a_dist.csv")

if __name__ == "__main__":
    main()