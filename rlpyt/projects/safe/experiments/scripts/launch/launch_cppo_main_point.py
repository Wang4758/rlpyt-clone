"""
    Main experiment launch file. Please refer to https://rlpyt.readthedocs.io/en/latest/pages/launch.html for details.
    Use the wait_for_pid function to make sure you can use the server 24/7
"""

from pathlib import Path
from rlpyt.utils.launching.variant import make_variants, VariantLevel
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.affinity import quick_affinity_code, encode_affinity
from rlpyt.projects.safe.experiments.scripts.eval.eval_dcppo import run_eval

def wait_for_pid(pid: int, check_interval_s: int = 60):
    import psutil
    import time
    from datetime import datetime as dt
    while psutil.pid_exists(pid):
        time.time()
        now = dt.now().strftime('%Hh:%Mm')
        print(f'{now}: {pid} still running. Check again in {check_interval_s} seconds...')
        time.sleep(check_interval_s)
    print('Process done! Launching 🚀🚀🚀')


script = Path(__file__).parent.joinpath("../train/train_dcppo.py").resolve().as_posix()

affinity_code = encode_affinity(n_cpu_core=64, cpu_per_run=32, n_gpu=0, hyperthread_offset=None, set_affinity=True)

variant_levels = list()

# values = [1.0, 0.9, 0.5, 0.1]
# dir_names = [f'{v}wc_alpha' for v in values]
# keys = [('algo', 'wc_risk_lvl')]
# variant_levels.append(VariantLevel(keys, list(zip(values)), dir_names))

# values = [7.5]
# dir_names = [f'{v}cl' for v in values]
# keys = [('algo', 'cost_limit')]
# variant_levels.append(VariantLevel(keys, list(zip(values)), dir_names))

variants, log_dirs = make_variants(*variant_levels)

default_config_key = "CRMONITOR"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title='test',
    runs_per_setting=1,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
    run_id_offset=0
)

exit()

#
#   MONITOR SEPARATE RULES
#

default_config_key = "CRMONITOR"
runs_per_setting = 2
experiment_title = "crm_separate"

variant_levels = list()

# Monitor Rules
active_rules = [["R_G2"]]
values = active_rules
dir_names = ['rules'+''.join(rs).replace('_','') for rs in values]
keys = [("env", "active_rules")]
variant_levels.append(VariantLevel(keys, values, dir_names))

# steps
n_steps = [25e6]
dir_names = ['10e6steps']
keys = [("runner", 'n_steps')]
variant_levels.append(VariantLevel(keys, list(zip(n_steps)), dir_names))

variants, log_dirs = make_variants(*variant_levels)

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)

#
#   MONITOR WITH ALL RULES AND NO CONSTRAINT
#

variant_levels = list()

# cost_limit (+ default config has to have penalty_init == 0!!)
cost_limits = [1000] # max steps = 1000 --> can not have more than 1000 cost per trajectory...
dir_names = ['1000clim']
keys = [('algo', 'cost_limit')]
variant_levels.append(VariantLevel(keys, list(zip(cost_limits)), dir_names))

penalty_inits = [0.0]
dir_names = ['0pi']
keys = [('algo', 'penalty_init')]
variant_levels.append(VariantLevel(keys, list(zip(penalty_inits)), dir_names))

pid_Ki = [0.0]
dir_names = ['0pid_Ki']
keys = [('algo', 'pid_Ki')]
variant_levels.append(VariantLevel(keys, list(zip(pid_Ki)), dir_names))

# steps
n_steps = [10e6]
dir_names = ['10e6steps']
keys = [("runner", 'n_steps')]
variant_levels.append(VariantLevel(keys, list(zip(n_steps)), dir_names))

variants, log_dirs = make_variants(*variant_levels)


run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title='crm_unconstrained',
    runs_per_setting=2,
    variants=variants,
    log_dirs=log_dirs,
    common_args=('CRMONITOR',),
)
