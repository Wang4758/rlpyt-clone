"""
    Main evaluation script.
    Searches for final model files in the provided folder and runs the evaluation.
    It also loads the env and model config from the log folder.
    Supply --train-scenarios to evaluate on training scenarios
"""

from copy import deepcopy
import json
import sys
import pprint
from typing import List
from multiprocessing import Pool
import multiprocessing as mp
import psutil
import pathlib
import re
import gzip
import torch
from rlpyt.samplers.serial.collectors import SerialEvalCollector

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config
from rlpyt.utils.launching.affinity import affinity_from_code, encode_affinity

from rlpyt.projects.safe.dcppo_agent import DCppoLstmAgent, DCppoAgent
from rlpyt.projects.safe.dcppo_pid import DCppoPID
from rlpyt.projects.safe.dcppo_model import DCppoModel
from rlpyt.projects.safe.dcppo_model_v2 import DCppoModelV2
from rlpyt.projects.safe.safety_gym_env import SafetyGymTrajInfo, safety_gym_make, CommonroadEvalTrajInfo
from rlpyt.projects.safe.commonroad_eval_collector import CommonroadCpuEvalCollector

import joblib

import commonroad_rl.gym_commonroad
import gym_monitor
import warnings

import torch
warnings.filterwarnings('ignore', category=FutureWarning)

def collect_traj_infos_commonroad(config: dict, model_path: str, rank = 0, world_size=1, snapshot_path: pathlib.Path = None, train_scenarios=False):

    test_with_train_scenarios = train_scenarios
    print(f"test_with_train_scenarios = {test_with_train_scenarios}")

    is_monitor = "monitor" in config['env']['id']

    affinity=affinity_from_code(f"0slt_1gpu_32cpu")
    
    config["sampler"]["batch_B"] = 32

    env_config = deepcopy(config["env"])
    test_env_config = deepcopy(config["env"])

    # load only one problem in the normal env to speed up everything
    env_config['load_problems_on_demand'] = True
    if is_monitor:
        env_config['preload_curvi_states'] = False

    test_env_config['test_env'] = True
    # test_env_config['max_problems'] = -1
    test_env_config['max_problems'] = 1000 if test_with_train_scenarios else -1
    test_env_config['test_with_train_scenarios'] = test_with_train_scenarios

    # the serial sampler will call reset() twice before actually collecting samples...
    test_env_config['play'] = True
    test_env_config['play_replay_first_n'] = 2
    if is_monitor:
        test_env_config['preload_curvi_states'] = False

    sampler = CpuSampler(
        EnvCls=safety_gym_make,
        env_kwargs=env_config,
        eval_n_envs = 32,
        eval_max_trajectories = 2800,
        eval_max_steps=1e7, #this gets divided per worker
        eval_env_kwargs=test_env_config,
        TrajInfoCls=CommonroadEvalTrajInfo,
        eval_CollectorCls=CommonroadCpuEvalCollector,
        **config["sampler"]
    )

    agent_cls = DCppoAgent if config['model']['lstm_size'] == None else DCppoLstmAgent
    agent = agent_cls(model_kwargs=config["model"], **config["agent"])

    _ = sampler.initialize(
        agent=agent,
        seed=rank,
        bootstrap_value=False,
        traj_info_kwargs=dict(discount=getattr(config['algo'], "discount", 1)),
        affinity=affinity
    )

    # load agent or model state dict
    if snapshot_path is None:
        agent.model.load_state_dict(
            torch.load(model_path)
        )
    else:
        agent.load_state_dict(
            torch.load(snapshot_path)['agent_state_dict']
        )

    agent.eval_mode(1)

    traj_infos: List[SafetyGymTrajInfo] = sampler.evaluate_agent(1)
    sampler.shutdown()


    return traj_infos

def collect_traj_infos_sg(config: dict, model_path: str, rank = 0, world_size=1):

    affinity=affinity_from_code("0slt_1gpu_32cpu")
    
    config["sampler"]["batch_T"] = 1000

    target_num_trajetories = 400

    sampler = CpuSampler(
        EnvCls=safety_gym_make,
        env_kwargs=config["env"],
        TrajInfoCls=SafetyGymTrajInfo,
        eval_n_envs = 32,
        eval_max_trajectories = target_num_trajetories,
        eval_max_steps=1e7,
        eval_env_kwargs=config['env'],
        **config["sampler"]
    )

    agent_cls = DCppoAgent if config['model']['lstm_size'] == None else DCppoLstmAgent 
    agent = agent_cls(model_kwargs=config["model"], **config["agent"])

    p = psutil.Process()
    try:
        if (affinity.get("master_cpus", None) is not None and
                affinity.get("set_affinity", True)):
            p.cpu_affinity(affinity["master_cpus"])
        cpu_affin = p.cpu_affinity()
    except AttributeError:
        cpu_affin = "UNAVAILABLE MacOS"

    if affinity.get("master_torch_threads", None) is not None:
        torch.set_num_threads(affinity["master_torch_threads"])

    _ = sampler.initialize(
        agent=agent,
        seed=rank,
        bootstrap_value=False,
        traj_info_kwargs=dict(discount=getattr(config['algo'], "discount", 1)),
        affinity=affinity
    )

    agent.eval_mode(1)
    agent.model.load_state_dict(
        torch.load(model_path)
    )

    traj_infos: List[SafetyGymTrajInfo] = sampler.evaluate_agent(1)
    sampler.shutdown()

    print(f'Got {len(traj_infos)} trajectories')

    return traj_infos[:target_num_trajetories]


def eval_model(model_path: pathlib.Path, task_id: int = 0, use_train=False):

    run_id = re.match(r'.*_run(\d+)\.pt', model_path.as_posix()).group(1)
    run_id = int(run_id)

    print(f"Evaling: {model_path} with run id {run_id}")

    variant_log_dir = model_path.parent.as_posix()
    with open(model_path.parent.joinpath(f'run_{run_id}/params.json')) as f:
        config = json.load(f)

    variant = load_variant(variant_log_dir)
    config = update_config(config, variant)

    env_config = config['env']
    is_commonroad = ('cr-monitor' in env_config['id']) or ('commonroad' in env_config['id'])

    if is_commonroad:
        traj_infos = collect_traj_infos_commonroad(config, model_path, rank=task_id, train_scenarios=use_train)
    else:
        traj_infos = collect_traj_infos_sg(config, model_path)

    # To dump results with commonroad State traces, use this code that can handle the State objects
    # jlresults_file = model_path.parent.joinpath(f'eval_results{run_id}_joblib{"_train" if use_train else ""}.gz')
    # joblib.dump(traj_infos, jlresults_file.as_posix(), compress=3)
    
    results_file = model_path.parent.joinpath(f'eval_results{run_id}_new{"_train" if use_train else ""}.json.gz')
    with gzip.open(results_file, 'wb') as f:
        print("Saving results!")
        data: str = json.dumps(traj_infos)
        f.write(data.encode('utf-8'))

def eval_snapshot_model(snapshot_path: pathlib.Path, task_id: int = 0):

    run_id = int(snapshot_path.parent.stem.replace('run_', ''))

    print(f"Evaling: {snapshot_path} with run id {run_id}")

    run_log_dir = snapshot_path.parent
    with open(run_log_dir.joinpath(f'params.json')) as f:
        config = json.load(f)

    variant_log_dir = run_log_dir.parent
    variant = load_variant(variant_log_dir)
    config = update_config(config, variant)

    env_config = config['env']
    is_commonroad = ('cr-monitor' in env_config['id']) or ('commonroad' in env_config['id'])

    if is_commonroad:
        traj_infos = collect_traj_infos_commonroad(config, None, rank=task_id, snapshot_path=snapshot_path)
    else:
        raise NotImplementedError('Pls update yourself :)')
        traj_infos = collect_traj_infos_sg(config, None)


    # results_file = model_path.parent.joinpath(f'eval_results{run_id}_new_trainset.json.gz')
    results_file = snapshot_path.parent.joinpath(f'eval_results{snapshot_path.stem}_new.json.gz')
    with gzip.open(results_file, 'wb') as f:
        print(f"Saving results! -> {results_file.as_posix()}")
        data: str = json.dumps(traj_infos)
        f.write(data.encode('utf-8'))

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def model_is_not_evaluated(model_path: pathlib.Path):
    runId = re.match(r'model_run(\d+)', model_path.stem).group(1)
    results_path = model_path.parent.joinpath(f'eval_results{runId}_new.json.gz')
    return not (results_path.is_file() and results_path.exists())

def run_eval_from_snapshot(snapshot_file: pathlib.Path):
    """Eval from model saved by rlpyt/rlpyt/utils/logging/logger.py > save_itr_params(...)"""
    print(f'Evaling snapshot {snapshot_file.absolute().as_posix()}')
    eval_snapshot_model(snapshot_file)

def run_eval(log_dir: str="/home/pillmayerc/mth/data_to_keep/01_paper_data/crm_dgae_7.5cl/0.5wc_alpha", cr_train_scenarios = False):

    model_paths = list(pathlib.Path(log_dir).glob('**/model_*.pt'))
    # model_paths = list(filter(lambda path: '1.0wc_alpha' not in path.as_posix(), model_paths))
    # model_paths = list(filter(model_is_not_evaluated, model_paths))
    # for path in model_paths:
    #     if '1.0wc_alpha' in path.as_posix():
    #         continue
    #     eval_model(path, 0)

    for paths in chunks(model_paths, 2):

        eval_processes: List[mp.Process] = []
        for i, model_path in enumerate(paths):
            p = mp.Process(target=eval_model, args=[model_path, i, cr_train_scenarios])
            # psp = psutil.Process(p.pid)
            # psp.cpu_affinity(list(range(4*i, 4*i+1)))
            p.start()
            eval_processes.append(p)
            # eval_model(model_path)

        for p in eval_processes:
            p.join()
        
        eval_processes.clear()

    print("EVAL DONE!")

def main(log_dir: str, cr_use_train: bool):
    if log_dir.endswith('.pkl'):
        run_eval_from_snapshot(pathlib.Path(log_dir))
    else:
        run_eval(log_dir, cr_use_train)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to training folder, models in this folder and subfolders will be evaluated")
    parser.add_argument("--train-scenarios", "-t", action="store_true")
    args = parser.parse_args()
    main(args.path, args.train_scenarios)