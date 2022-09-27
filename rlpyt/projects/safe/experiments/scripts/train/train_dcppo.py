"""
    Training entrypoint. Sets up model, agent, and algorithm and then trains the agent using minibatch-rl.
    Only start here directly for debugging. This script expects a certain folder structure with variant-config files as creates by the launch script.

    To debug, make sure to comment out the variant loading lines and probably reduce the number of samples
"""

import json
import os
import sys
import pprint
import numpy as np
from rlpyt.projects.safe.dcppo_model import DCppoModel
from rlpyt.projects.safe.dcppo_model_v2 import DCppoModelV2

from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config
from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.utils.logging import logger

from rlpyt.projects.safe.dcppo_agent import DCppoLstmAgent, DCppoAgent
from rlpyt.projects.safe.dcppo_pid import DCppoPID
from rlpyt.projects.safe.safety_gym_env import safety_gym_make, SafetyGymTrajInfo

from rlpyt.projects.safe.experiments.configs.cppo_pid import configs

import commonroad_rl.gym_commonroad
import gym_monitor

import warnings

import torch

warnings.filterwarnings("ignore", category=FutureWarning)


def build_and_train(
    slot_affinity_code="0slt_0gpu_1cpu_1cpr",
    log_dir="test",
    run_ID="0",
    config_key="CRMONITOR",
):

    # run_ID = str(2 * int(run_ID))

    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]

    seed = int(run_ID) * len(affinity["all_cpus"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    envid = config["env"]["id"]
    if ("commonroad" in envid) or ("cr-monitor" in envid):
        config["env"]["seed"] = seed

    # For debugging, change here:
    variant = load_variant(log_dir)
    config = update_config(config, variant)

    # For debugging, change here as well:
    # config["sampler"]['batch_T'] = 128
    # config["sampler"]['batch_B'] = 4
    # config["env"]['max_problems'] = 32

    pprint.pprint(config)

    assert not (
        config["env"].get("normalize_reward", False)
        and config["model"].get("normalize_reward", False)
    ), "Normalize reward with environment wrapper OR model, not both"

    sampler = CpuSampler(
        EnvCls=safety_gym_make,
        env_kwargs=config["env"],
        TrajInfoCls=SafetyGymTrajInfo,
        **config["sampler"],
    )
    algo = DCppoPID(**config["algo"])

    agent_cls = DCppoAgent if config["model"]["lstm_size"] == None else DCppoLstmAgent

    agent = agent_cls(model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRl(
        seed=seed,
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"],
    )
    name = "cppo_" + config["env"]["id"]
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="gap", use_summary_writer=False):
        variant_base_dir = logger.get_variant_base_dir()
        # n_steps / interval = number of snapshots
        # n_iters / number of snapshots = snapshot gap
        logger.set_snapshot_gap(400)

        # before starting training, save env config if we use commonroad
        if ("commonroad" in config["env"]["id"]) or (
            "cr-monitor" in config["env"]["id"]
        ):
            env = safety_gym_make(rlpyt_fast_init=True, **config["env"])
            with open(os.path.join(variant_base_dir, "env_config.json"), "w") as f:
                json.dump(env.configs, f, indent=1)
            del env

        # train
        runner.train()

        # save model
        model_path = os.path.join(variant_base_dir, f"model_run{run_ID}.pt")
        torch.save(agent.model.state_dict(), model_path)


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
