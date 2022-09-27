# Code for my masters thesis

Adapted from code for the paper: "Responsive Safety in RL by PID Lagrangian Methods" (ICML 2020).

## Setup
The folder structure after running the install script will be:
```
    <project_root> (called mth in most scripts)
    ├── rlpyt (this repository)
    ├── commonroad_rl
    ├── cr_monitor_gym
    ├── safety-gym
    ├── safety-starter-agents
```

The install script creates the conda environment, clones and install all the packages (including mujoco-py + MuJoCo). The apt libraries are required for safety-gym / mujoco-py. 

Wait while it runs, at 2 points you need to enter the sudo password or skip.
```sh
$ cd <project_root>

# install safety-gym libs
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

# 'install' repositories + create conda env
$ bash rlpyt/rlpyt/projects/safe/setup/install.sh <conda_env_name>

# setup pickles
$ cd <project_root>/commonroad_rl
$ rm -rf pickles
$ ln -s <path_to_highD_dataset>/pickles pickles
```

## Which file does what ? - rlpyt
All files are named relative to this readme.
| Filename | Purpose |
| -- | -- |
| [experiments/configs/cppo_pid.py](./experiments/configs/cppo_pid.py) | Base configs for SG, CR-RL, CR-Monitor |
| [experiments/scripts/cr_util](./experiments/scripts/cr_util) | Utility scripts to evaluate the initial state violations + results from that in csv. Those files are needed to filter the scenarios for the monitor env <br> **Additional info** is also present at the top of the scripts! |
| [experiments/scripts/eval](./experiments/scripts/eval) | Contains evaluation scripts. The model evaluation ones are `eval_dcppo.py` for our method, `eval_sg_pg.py` was only used for the CPO evaluation, and `eval_wcsac_cr.py` to evaluate the wcsac model on cr-rl or cr-monitor (see script for that) <br> **Additional info** is also present at the top of the scripts! |
| [experiments/scripts/train](./experiments/scripts/train) | Training entrypoints that the launch script calls (dcppo is new, cppo is the old one from PID-Lag) |
| [experiments/scripts/launch](./experiments/scripts/launch) | Contains the main experiment launcher for our method. |
| [experiments/scripts/plot](./experiments/scripts/plot) | Many different plotting scripts. `plot_rlpyt` is the main one for training graphs with a lot of options. `plot_wc` creates the KDE plots and CVAR cost tables. `plot_rlpyt_bulk` calls the first two based on config files (see my thesis data for examples). `plot_settings` has additional colormaps and is used from most other plot-scripts as it configures seaborn styles such that the figures work in my thesis (fontsize, font type,...). `traffic_rule_conformance_overview` create the barplots from the thesis/paper. `plot_traffic_rule_vios` evaluation the rule compliance that can be pasted into the previous script. |
| [setup](./setup) | just the `install.sh` script |
| [commonroad_eval_collector.py](./commonroad_eval_collector.py) | Eval collector that fills the actions, ego states, and benchmark_id values in the `CommonroadEvalTrajInfo` defined in `safety_gym_env.py` |
| [cppo_agent.py](./cppo_agent.py) | Agent file from PID-Lagrange |
| [cppo_model.py](./cppo_model.py) | Model file from PID-Lagrange |
| [cppo_pid.py](./cppo_pid.py) | Algorithm file from PID-Lagrange |
| [dcppo_agent.py](./dcppo_agent.py) | Agent implementation of our method. It encapsulated the model and is the way that the problem is setup in rlpyt |
| [dcppo_model.py](./dcppo_model.py) | Model used in the thesis experiments |
| [dcppo_model_v2.py](./dcppo_model_v2.py) | Model used in the paper experiments |
| [dcppo_pid.py](./dcppo_pid.py) | Implementation of our method. Has many switches to use primal and adam based weighting of cost and reward objective |
| [running_mean_std.py](./running_mean_std.py) | Has a running mean and std model that can be used to estimate the episodic mean and variance of the costs (config key: `use_ep_cost_rms`) |
| [safety_gym_env.py](./safety_gym_env.py) | Contains the previously implemented wrapper for the safety-gym environments. Also has definitions for so called "trajectory-info" classes that the collector classes need to know which attributes to log. There are special versions of that for commonroad that log the robustness values and the termination criteria. |
| [util.py](./util.py) | contains utility functions like the distributional gae implementation |

## Performing Trainings
### rlpyt
The setup here is adapted from what was used originally and "how you are supposed to do it in rlpyt" (https://rlpyt.readthedocs.io/en/latest/pages/launch.html)
- **DCPPO Training Entrypoint**: `experiments/scripts/train/train_dcppo.py`. You are not really supposed to use this for starting a training though as it already expects some folders and files to be present (Variant configuration). But when debugging, use this file as the start (and comment out the variant loading at least).
- **Experiment Launch File**: `experiments/scripts/launch/launch_cppo_main_point.py`. This file is where you run the experiments from. If only the base config should be used, just dont add any variants. The function `run_experiments(...)` has settings for the amount of runs per settings which will perform the same training multiple times but with different seeds. It also creates the directories and variant config files that the training entrypoint script wants.
- **Base Configs**: In the launch file, you are expected to give a base config key. These correspond to dictionary entries in `experiments/configs/cppo_pid.py`
    - `SAFETYGYM`: For safety-gym environments (actually activates the LSTM part)
    - `COMMONROAD`: For CommonRoad-RL (commonroad-v1)
    - `CRMONITOR`: For CommonRoad-Monitor (cr-monitor-v0)

#### Enabling the different methods to weight/select J_C and J_R
| Optimization Method | `use_primal` | `learn_penalty` |
|--|--|--|
| PID | `False` | `False` |
| Adam (-Optimizer) | `True` | `False` |
| Primal | `False` | `True` |

### safety-starter-agents (CPO, WCSAC):
Clone the starter-agents repo next to the safety-gym repo:
```sh
$ git clone git@gitlab.lrz.de:ga84moc/safety-starter-agents-clone.git safety-starter-agents
```
I only added more logging for the most part here. (And copied the wcsac from the authors repo)

- WCSAC: use `safe_rl/sac/wcsac.py` directly 
- CPO: use `safe_rl/pg/run_agent.py`: You need to check the settings and arguments there (`--algo cpo` makes it use CPO for example).

### General CommonroadEnv notes:

⚠️ The scenario loading part is a little different from the commonroad-rl examples. Whereas there, you should create different folders with subsets of scenarios for each training process, here the environment is adapted to only load a subset of scenarios in its path based on `mp_world_size` and `mp_rank` arguments. Check the description of `__init__` args below for details.

All scripts assume that the correct scenario pickles are in `commonroad_rl/pickles`. Simply remove the default pickle folder and link the correct (highD) pickles using 
```sh
$ cd <commonroad_rl_dir>
$ rm -rf pickles
$ ln -s <path_to_pickles> pickles
```

The constructor for `CommonroadEnv` has more arguments in general now. These were required for various things throughout this project:
- `rlpyt_fast_init`: rlpyt quickly instantiates the env when setting up the multiprocessing sample buffers --> Dont want to wait forever until everything is loaded. It only sets up the observation and actions spaces and returns random values for `step` and `reset`.
- `mp_world_size`: #parallel envs (for scenario loading)
- `mp_rank`: env index from #parallel envs (for scenario loading)
- `max_problems`: Only load this many problems at most (summed up over all parallel envs) -> Good for debugging as it loads much faster this way.
- `play_replay_first_n`: When collecting evals with rlpyt, it resets the env 2 times for whatever reason in the beginning. So if `play=True` then it wastes two scenarios. Set this to 2 to play the first two scenarios again at the end.
- `load_problems_on_demand`: This does not preload any pickle files, useful for fast instantiation
- `test_with_train_scenarios`: If true, loads train scenarios when `test_env=True`. Used to collect the evaluation on training scenarios.
- `scenario_filter_file_path`: Should be None or point to a .txt file with one benchmark id per line. Only ids listed in the file will be used for training. This is used to filter the monitor scenarios where the initial state is unwanted.

### Monitor env notes
To speed up the training the `cr-monitor-v0` environment had the option to load obstacle curvilinear states from a file. In order to remove the need to always load the whole file, the script that generates this file has been adapted to ouput single files for each benchmark id instead of one large file.

The path to the folder with these files needs to be put into the contructor of the env. For example:
```py
obstacle_vehicle_dict_dir: str = "/home/pillmayerc/datasets/highD_new/obstacle_curvi_states"
```
To generate the files, use the `store_curvilinear_states.py` just like before. Supply `--output_folder` argument to output individual files to the provided folder.
