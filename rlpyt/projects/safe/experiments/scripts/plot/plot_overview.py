""" Creates the final eval table for commonroad in the thesis """

from os import access
from pathlib import Path
from typing import List
from black import json
import pandas as pd
import seaborn as sns
from rlpyt.projects.safe.experiments.scripts.plot.plot_settings import set_plotting_theme
from rlpyt.projects.safe.experiments.scripts.plot.plot_rlpyt import search_dict
from itertools import chain
import gzip

DATA = {
    'CPO': '/home/pillmayerc/mth/data_to_keep/00_thesis_data/cr_cpo',
    'PPO': '/home/pillmayerc/mth/data_to_keep/00_thesis_data/cr_unconstrained',
    'PPO-0.1': '/home/pillmayerc/mth/data_to_keep/00_thesis_data/cr_wc_pid_v3/commonroad-v1/0.1wc_alpha',
    'PPO-0.5': '/home/pillmayerc/mth/data_to_keep/00_thesis_data/cr_wc_pid_v3/commonroad-v1/0.5wc_alpha',
    'PPO-0.9': '/home/pillmayerc/mth/data_to_keep/00_thesis_data/cr_wc_pid_v3/commonroad-v1/0.9wc_alpha',
    'PPO-1.0': '/home/pillmayerc/mth/data_to_keep/00_thesis_data/cr_wc_pid_v3/commonroad-v1/1.0wc_alpha',
    'WCSAC-0.1': '/home/pillmayerc/mth/data_to_keep/00_thesis_data/cr_wcsac_muchlonger/0.1wc_alpha',
    'WCSAC-0.5': '/home/pillmayerc/mth/data_to_keep/00_thesis_data/cr_wcsac_muchlonger/0.5wc_alpha',
    'WCSAC-0.9': '/home/pillmayerc/mth/data_to_keep/00_thesis_data/cr_wcsac_muchlonger/0.9wc_alpha',
    'WCSAC-1.0': '/home/pillmayerc/mth/data_to_keep/00_thesis_data/cr_wcsac_muchlonger/1.0wc_alpha',
    'Primal': '/home/pillmayerc/mth/data_to_keep/00_thesis_data/cr_primal/0.5costema/Trueclipcosto',
    'PPO-OptLag': '/home/pillmayerc/mth/data_to_keep/00_thesis_data/cr_optlag_2/commonroad-v1/0.05penlr'
}

def load_data(fp: str, method_name: str):
    dfs = []
    for p in Path(fp).glob('**/eval_results*_new.json.gz'):
        with gzip.open(p.as_posix()) as f:
            df = df = pd.read_json(f)
        config_path = Path(p).parent.joinpath('run_0', 'params.json')
        with open(config_path.as_posix(), 'rt') as f:
            config = json.load(f)
        cost_limit = search_dict(config, 'cost_limit')
        cost_limit = 2.0
        print(f'cost limit = {cost_limit}')
        df['cost_limit'] = cost_limit
        df['Method'] = method_name
        dfs.append(df)

    return pd.concat(dfs)

def find_datasets(search_paths: List[Path]):
    globs = chain([p.glob('**/progress.csv') for p in search_paths])
    return list(globs)
        

def main():
    dfs = []
    for name, p in DATA.items():
        print(name)
        dfs.append(load_data(p, name))
    df = pd.concat(dfs)
    df['Cost'] = df['Cost'] - df['cost_limit']
    df = df[["Method","Cost", "IsGoalReached"]]
    absolute_eval = df.groupby('Method').mean().reset_index().sort_values('IsGoalReached', ascending=True).round(2)
    print(absolute_eval.sort_values('Cost', ascending=False))
    ppo = absolute_eval[absolute_eval['Method'] == 'PPO']
    gr, cost = ppo['IsGoalReached'].item(), ppo['Cost'].item()
    relative_eval = absolute_eval.copy()
    relative_eval['IsGoalReached'] /= gr
    relative_eval['Cost'] /= cost
    print(relative_eval.sort_values('Cost', ascending=False).round(2))

"""
Absolute:

        Method   Cost-Viola.    IsGoalReached
1          PPO  17.21           0.94
0          CPO   2.65           0.71
5      PPO-1.0   0.48           0.96
10   WCSAC-0.9   0.39           0.88
6   PPO-OptLag   0.26           0.95
7       Primal   0.23           0.94
4      PPO-0.9  -0.58           0.94
11   WCSAC-1.0  -0.74           0.89
9    WCSAC-0.5  -0.86           0.80
3      PPO-0.5  -1.32           0.94
8    WCSAC-0.1  -1.43           0.91
2      PPO-0.1  -1.75           0.90


Relative to PPO:

        Method  Cost-Viola.   IsGoalReached
1          PPO  1.00           1.00
0          CPO  0.15           0.76
5      PPO-1.0  0.03           1.02
10   WCSAC-0.9  0.02           0.94
6   PPO-OptLag  0.02           1.01
7       Primal  0.01           1.00
4      PPO-0.9 -0.03           1.00
11   WCSAC-1.0 -0.04           0.95
9    WCSAC-0.5 -0.05           0.85
3      PPO-0.5 -0.08           1.00
8    WCSAC-0.1 -0.08           0.97
2      PPO-0.1 -0.10           0.96
"""

main()