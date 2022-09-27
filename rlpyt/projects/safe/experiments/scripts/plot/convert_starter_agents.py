""" Converts the starter agents progress.txt files to csv and renames some columns such that is can be used by the plot_rlpyt.py script """

import pathlib
import json
from typing import List
import yaml
import argparse
import pandas as pd

def find_progess_files(dir: pathlib.Path) -> List[pathlib.Path]:
    return dir.glob('**/progress.txt')

def convert_one(progress_path: pathlib.Path):

    rename_map = {
        'Epoch': 'test',
        'AverageEpCost': 'CostAverage',
        'AverageEpCost': 'CostAverage',
        'CostRate': 'costrate',
        'TotalEnvInteracts': 'Diagnostics/CumSteps',
        'CumulativeCost': 'cumCostMax',
        'Beta': 'costPenaltyAverage',
        'EpLen': 'LengthAverage'
    }

    progress = pd.read_csv(progress_path, sep='\t')
    if 'EpGoals' in progress:
        rename_map.update({'EpGoals': 'IsGoalReachedAverage'})
    else:
        rename_map.update({'AverageEpGoals': 'IsGoalReachedAverage'})
    progress = progress.rename(rename_map, axis='columns', errors='raise')
    # drop_cols = [c for c in progress.columns if c not in rename_map]
    # progress.drop(columns=list(drop_cols), inplace=True)
    progress.to_csv(progress_path.parent.joinpath('progress.csv'), index=False)
    
    with open(progress_path.parent.joinpath('config.json'), 'r') as f:
        config = json.load(f)
        converted_config = dict(
            cost_limit = config['cost_lim'],
            cost_scale = 1,
            id = 'unknown-env'
        )
    with open(progress_path.parent.joinpath('params.json'), 'wt') as f:
        json.dump(converted_config, f)


def main(dir: pathlib.Path):
    files = find_progess_files(dir)
    for file in files:
        print(f'Converting {file.as_posix()}')
        convert_one(file)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dir', type=pathlib.Path)
    args = p.parse_args()
    main(**(args.__dict__))
