"""
    Bulk plotting script to process the whole data folder at once.
    It can run the plot_wc and plot_rlpyt scripts based on which config files it finds.
    It works by basically reading the yaml and converting all dict entries to strings and supplies them as args to the scripts.
    It overwrites the output directory such that all the plots end up in one directory.
"""

from os import path
import pathlib
import argparse
from typing import List
from itertools import repeat
import yaml
import sys
import subprocess
import multiprocessing

PLOT_SCRIPT = pathlib.Path(__file__).parent.joinpath('plot_rlpyt.py').as_posix()
WC_PLOT_SCRIPT = pathlib.Path(__file__).parent.joinpath('plot_wc.py').as_posix()

def find_directories(path: pathlib.Path) -> List[pathlib.Path]:
    if path.joinpath('plot_config.yml').exists():
        return [path]
    else:
        res = []
        for dirs in (find_directories(x) for x in path.iterdir() if x.is_dir()):
            res.extend(dirs)
        return res

def plot_from_dir(config_path: pathlib.Path, output_dir: pathlib.Path):
    
    with open(config_path, 'r') as f:
        print(f'📖 Reading config {config_path.as_posix()}')
        plot_config: dict = yaml.safe_load(f)
    
    if 'plot_wc' in config_path.name:
        output_subdir_name = plot_config.get('output_subdir_name', config_path.parent.stem)
        output_subdir = output_dir.joinpath(output_subdir_name)
        output_subdir.mkdir(exist_ok=True, parents=True)
        args = [
            sys.executable, WC_PLOT_SCRIPT,
            '-d', config_path.parent.as_posix(),
            '-o', output_subdir.as_posix()
        ]
        plot_config = [plot_config['configs']]
    else:
        if 'output_name' in plot_config:
            output_name = plot_config['output_name']
            del plot_config['output_name']
        else:
            output_name = config_path.parent.name
        args = [
            sys.executable, PLOT_SCRIPT, 
            '-d', config_path.parent.as_posix(),
            '-o', output_dir.joinpath(output_name + '.pdf')
        ]
        plot_config = [plot_config]

    for pcfg in plot_config:
        if pcfg is not None:
            for arg_name, arg_value in pcfg.items():
                args.append(f'--{arg_name}')
                
                if isinstance(arg_value, list) or isinstance(arg_value, tuple):
                    args.extend(arg_value)
                elif isinstance(arg_value, dict):
                    args.append(str(arg_value))
                else:
                    args.append(arg_value)
        
        args = list(map(str, args))
        # print(args)
        print('Launching plot process 🚀')
        proc = subprocess.Popen(args)
        proc.wait()


def main(source_dir: pathlib.Path, output_dir: pathlib.Path):

    output_dir.mkdir(exist_ok=True, parents=True)

    config_paths = list(source_dir.glob('**/plot_config.yml'))
    config_paths += list(source_dir.glob('**/plot_wc_config.yml'))
    print(config_paths)

    with multiprocessing.Pool(4) as pool:
        pool.starmap(plot_from_dir, zip(config_paths, repeat(output_dir.absolute())))

    print(f'🎉 Created {len(config_paths)} plots -> {output_dir.as_posix()} 🎉')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', help='Input directory', type=pathlib.Path, required=True)
    parser.add_argument('--output_dir', default='/home/pillmayerc/mth/plots', type=pathlib.Path)
    args = parser.parse_args()
    main(**(args.__dict__))