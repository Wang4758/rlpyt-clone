"""
    Used to get the violation severity (aka. dense cost) for a rlpyt model.
    It basically just runs the model on all scenarios and records the cost
"""

import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import gzip
import pandas as pd
import numpy as np
from multiprocessing import Pool

import commonroad_rl.gym_commonroad
import gym
from pprint import pprint

def eval_scenario(row):

    idx, row = row

    env = gym.make(
        'commonroad-v1',
        test_env=True,
        load_problems_on_demand=True,
        logging_mode="ERROR",
        test_reset_config_path = "/home/pillmayerc/datasets/highD_new/pickles/problem",
        # this has to match the model you load...
        cost_configs = dict(use_sparse = False),
        surrounding_configs = dict(observe_relative_a = False, fast_distance_calculation=False),
        traffic_sign_configs = dict(observe_speed_limit = False),
    )

    benchmark_id = row["BenchmarkId"]
    actions = np.array(row["Actions"])
    cost_sparse_old = row["Cost"]
    env.reset(benchmark_id = benchmark_id)
    cost = 0
    cost_sparse = 0
    print(idx, benchmark_id)
    for action in actions:
        _, _, d, i = env.step(action)
        cost += i.get("cost")
        cost_sparse += int(i.get("cost") > 0)

    assert cost_sparse_old == cost_sparse, "costs not matching..."
    assert d, "not done at end of trajectory?"

    return benchmark_id, cost

def process_eval_file(gz: pathlib.Path):
    with gzip.open(gz, 'rb') as f:
        df = pd.read_json(f)

    print(df)

    with Pool(64) as pool:
        dense_cost = pool.map(eval_scenario, df.iterrows())
        dense_cost = pd.DataFrame(dense_cost, columns =['BenchmarkId', 'CostDense'])

    df = df.merge(dense_cost, left_on="BenchmarkId", right_on="BenchmarkId")

    drop_cols = set(df.columns) - {"BenchmarkId", "Cost", "CostDense", "CostSparse"}
    df.drop(columns=list(drop_cols), inplace=True)

    new_name = gz.name.replace(".json.gz", ".csv")
    save_path = gz.parent.joinpath(new_name)
    df.to_csv(save_path, index=False)


def main(d: pathlib.Path):
    process_eval_file(d)
    

if __name__ == '__main__':
    files = [
        "/home/pillmayerc/mth/data_to_keep/00_new/cr_wc_pid_v3_tuned1.0/commonroad-v1/1.0wc_alpha/eval_results0_new.json.gz",
        "/home/pillmayerc/mth/data_to_keep/00_new/cr_wc_pid_v3_tuned1.0/commonroad-v1/1.0wc_alpha/eval_results1_new.json.gz",
        "/home/pillmayerc/mth/data_to_keep/00_new/cr_wc_pid_v3_tuned1.0/commonroad-v1/1.0wc_alpha/eval_results2_new.json.gz"
    ]
    for p in files:
        main(pathlib.Path(p))