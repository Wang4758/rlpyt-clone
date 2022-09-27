"""
    Evaluate the traffic rule compliance rates from evaluation files.
"""

import argparse
import gzip
import json
import numpy as np
import pandas as pd
import pathlib
from itertools import chain

def load_stuff(d: pathlib.Path):
    results = []
    for gz in chain(d.glob('**/eval_results*_new.json.gz')):
        if 'itr' in gz.as_posix():
            continue
        print(f'Reading {gz}')
        with gzip.open(gz, 'rb') as f:
            results.append(pd.read_json(f))
    return pd.concat(results)

def do_stuff(df: pd.DataFrame, save_path: pathlib.Path, args):
    if "Actions" in df:
        df.drop(columns=['Actions'], inplace=True)
    print(f'Amount of scenarios = {len(df)}')
    means = df.mean()
    mean_len = means["Length"]
    columns = ['NViolationsRG1', 'NViolationsRG2', 'NViolationsRG3', 'NTrafficRuleViolations']
    counts = df[columns]

    total_steps = df["Length"].sum()
    print(counts.sum() / total_steps)
    print("===")
    # take into account that worst case sac has shorter trajectories
    mean_vios_per_length = means[columns]
    print(mean_vios_per_length / mean_len)

    violation_table = counts.apply(lambda x: x.astype('bool').value_counts()).T.stack()
    violation_table = (violation_table / len(df)) #.round(2)
    print(violation_table)
    print(means)
    mean_vios_per_length.to_string(save_path.joinpath(f'vios_per_length{("_"+args.suffix) if args.suffix else ""}.txt'))
    violation_table.to_string(save_path.joinpath(f'traffic_rule_violation_table{("_"+args.suffix) if args.suffix else ""}.txt'))
    means.to_string(save_path.joinpath(f'traffic_rule_vio_means{("_"+args.suffix) if args.suffix else ""}.txt'))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dir', type=pathlib.Path, required=True)
    p.add_argument('--suffix', default = "")
    args = p.parse_args()
    
    d = pathlib.Path(args.dir)
    df = load_stuff(d)

    # df = pd.read_csv('/home/pillmayerc/mth/rlpyt/rlpyt/projects/safe/experiments/scripts/cr_util/human_eval_test.csv')
    # d = pathlib.Path('/home/pillmayerc/mth/rlpyt/rlpyt/projects/safe/experiments/scripts/cr_util/')
    do_stuff(df, d, args)


    
