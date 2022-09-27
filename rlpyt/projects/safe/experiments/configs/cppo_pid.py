""" Contains base configs for safety-gym, commonroad-rl, and commonroad-monitor
    The config keys are defined at the very end of the file 
"""

import copy
from pathlib import Path

configs = dict()

config = dict(
    env=dict(
        id="commonroad-v1",
        obs_prev_cost=True,
        obs_version="default",
        normalize_reward=True,
    ),
    sampler=dict(
        batch_T=1024, #128
        batch_B=32,  #104 # Might bust memory limits.
        max_decorrelation_steps=1000,
    ),
    algo=dict(
        discount=0.99,
        learning_rate=1e-4,
        value_loss_coeff=1.,
        entropy_loss_coeff=0,
        clip_grad_norm=1e4,
        gae_lambda=0.97,
        minibatches=4,
        epochs=8,
        ratio_clip=0.1, #.1?
        linear_lr_schedule=False,
        normalize_advantage=True, #True?
        cost_discount=None,
        cost_gae_lambda=None,
        cost_value_loss_coeff=0.5, # originally 0.5
        ep_cost_ema_alpha=0.5,  # 0 for hard update.
        use_ep_cost_rms=True,
        ep_cost_rms_max_count=1000.,
        objective_penalized=True,
        learn_c_value=True,
        penalty_init=0.,
        cost_limit=2,
        cost_scale=1,  # yes 10. (1 is better for commonroad)
        normalize_cost_advantage=True, # this ONLY centers, but does not rescale (like in safety starter agents)
        pid_Kp=2.0,#2.0,
        pid_Ki=0.05,#0.01,
        pid_Kd=0,
        pid_d_delay=1,
        pid_delta_p_ema_alpha=0.5,  # 0 for hard update
        pid_delta_d_ema_alpha=0.95,
        pid_i_decay=1.0,
        sum_norm=True,
        diff_norm=False,
        penalty_max=100,  # only if sum_norm=diff_norm=False
        step_cost_limit_steps=None,
        step_cost_limit_value=None,
        use_beta_kl=False,
        use_beta_grad=False,
        record_beta_kl=False,
        record_beta_grad=False,
        beta_max=10,
        beta_ema_alpha=0.9,
        beta_kl_epochs=1,
        reward_scale=1,
        lagrange_quadratic_penalty=False,
        quadratic_penalty_coeff=1,
        wc_risk_lvl=1.,
        wc_update_advantage = True,
        use_primal=False,

        use_clipped_cost_obj=True,

        use_kl_limit = True,
        target_kl=0.01, # from safety starter agents ppo-lag
        kl_margin=1.2, # from safety starter agents ppo-lag
        learn_penalty=False, # this disables PID and instead learns the cost_penalty
        penalty_lr = 5e-2, # used if learn_penalty
    ),
    agent=dict(
        ModelCls="DCppoModelV2", #DCppoModel
    ),
    model=dict(
        hidden_sizes=[128, 128],
        lstm_size=None,
        lstm_skip=False,
        constraint=True,  # must match algo.learn_c_value
        normalize_observation=True,
        var_clip=1e-6,
        mu_input_mode="concat"
    ),
    runner=dict(
        n_steps=10e6,
        log_interval_steps=30000,
    ),
)

config_monitor = dict(
    env=dict(
        id="cr-monitor-v0",
        obs_prev_cost=True,
        obs_version="default",
        normalize_reward=True,
        # for commonroad:
        max_problems = -1,
        # for monitor:
        sparse_rule_reward=-1.0,
        active_rules = ["R_G1","R_G2", "R_G3"],
        rob_as_reward=False,
        scenario_filter_file_path = Path(__file__).parent.joinpath('../scripts/cr_util/results_R_G1R_G2R_G3_max_acc_ALLokayids.txt').resolve().as_posix(),
        # scenario_filter_file_path = '/home/pillmayerc/mth/rlpyt/rlpyt/projects/safe/experiments/scripts/cr_util/results_R_G1R_G2R_G3_new_a_obs_ALLokayids.txt',
        observe_robustness=False,
        preload_curvi_states=True,
    ),
    sampler=dict(
        batch_T=256, #128
        batch_B=32,  #104 # Might bust memory limits.
        max_decorrelation_steps=1000,
    ),
    algo=dict(
        discount=0.99,
        learning_rate=1e-4,
        value_loss_coeff=1.,
        entropy_loss_coeff=0,
        clip_grad_norm=1e4,
        gae_lambda=0.97,
        minibatches=4,
        epochs=8,
        ratio_clip=0.1, #.1?
        linear_lr_schedule=False,
        normalize_advantage=True, #True?
        cost_discount=None,
        cost_gae_lambda=None,
        cost_value_loss_coeff=0.5, # originally 0.5
        ep_cost_ema_alpha=0.975,  # 0 for hard update.
        use_ep_cost_rms=False,
        ep_cost_rms_max_count=25_000.,
        objective_penalized=True,
        learn_c_value=True,
        penalty_init=0.0,
        cost_limit=7.5,
        cost_scale=5.0,  # yes 10. (1 is better for commonroad)
        normalize_cost_advantage=True, # this ONLY centers, but does not rescale (like in safety starter agents)
        pid_Kp=0.5,#2.0,
        pid_Ki=0.001,#0.01,
        pid_Kd=0,
        pid_d_delay=1,
        pid_delta_p_ema_alpha=0.95,  # 0 for hard update
        pid_delta_d_ema_alpha=0.95,
        pid_i_decay=1.0,
        pid_i_max=25.0,
        sum_norm=True,
        diff_norm=False,
        penalty_max=100,  # only if sum_norm=diff_norm=False
        step_cost_limit_steps=None,
        step_cost_limit_value=None,
        use_beta_kl=False,
        use_beta_grad=False,
        record_beta_kl=False,
        record_beta_grad=False,
        beta_max=10,
        beta_ema_alpha=0.9,
        beta_kl_epochs=1,
        reward_scale=1,
        lagrange_quadratic_penalty=False,
        quadratic_penalty_coeff=1,
        wc_risk_lvl=0.5,
        wc_update_advantage = True,
        use_primal=False,

        use_clipped_cost_obj=True,

        use_kl_limit = True,
        target_kl=0.01, # from safety starter agents ppo-lag
        kl_margin=1.2, # from safety starter agents ppo-lag
        learn_penalty=False, # this disables PID and instead learns the cost_penalty
        penalty_lr = 0.05, # used if learn_penalty
    ),
    agent=dict(
        ModelCls="DCppoModelV2", #DCppoModel
    ),
    model=dict(
        hidden_sizes=[128, 128],
        lstm_size=None,
        lstm_skip=False,
        constraint=True,  # must match algo.learn_c_value
        normalize_observation=True,
        var_clip=1e-6,
        # shift distribution maybe better for rule 2
        init_log_std=-1.9, # -1.9, # -1.386,
        a_long_shift=0.15, # 0.15 # 0.4
        mu_input_mode="concat"
    ),
    runner=dict(
        n_steps=40e6,
        log_interval_steps=30000,
    ),
)

config_orig = dict(
    env=dict(
        id="Safexp-PointGoal1-v0",
        obs_prev_cost=True,
        obs_version="default",
        normalize_reward=False,
    ),
    sampler=dict(
        batch_T=208, # 128
        batch_B=64,  # 104 Might bust memory limits.
        max_decorrelation_steps=1000,
    ),
    algo=dict(
        discount=0.99,
        learning_rate=1e-4,
        value_loss_coeff=1.,
        entropy_loss_coeff=0,
        clip_grad_norm=1e4,
        gae_lambda=0.97,
        minibatches=2,
        epochs=8,
        ratio_clip=0.1,
        linear_lr_schedule=False,
        normalize_advantage=False,
        cost_discount=None,
        cost_gae_lambda=None,
        cost_value_loss_coeff=0.5,
        ep_cost_ema_alpha=0.5,  # 0 for hard update.
        objective_penalized=True,
        learn_c_value=True,
        penalty_init=0.,
        cost_limit=25,
        cost_scale=10,  # yes 10.
        normalize_cost_advantage=False,
        pid_Kp=1,
        pid_Ki=1e-2,
        pid_Kd=0,
        pid_d_delay=1,
        pid_delta_p_ema_alpha=0.95,  # 0 for hard update
        pid_delta_d_ema_alpha=0.95,
        pid_i_decay=1.0,
        sum_norm=True,
        diff_norm=False,
        penalty_max=100,  # only if sum_norm=diff_norm=False
        step_cost_limit_steps=None,
        step_cost_limit_value=None,
        use_beta_kl=False,
        use_beta_grad=False,
        record_beta_kl=False,
        record_beta_grad=False,
        beta_max=10,
        beta_ema_alpha=0.9,
        beta_kl_epochs=1,
        reward_scale=1,
        lagrange_quadratic_penalty=False,
        quadratic_penalty_coeff=1,
        wc_risk_lvl=1.,
        wc_update_advantage = True,
        use_primal=False,

        use_clipped_cost_obj=True,
        
        use_kl_limit = False,
        target_kl=0.01, # from safety starter agents ppo-lag
        kl_margin=1.2, # from safety starter agents ppo-lag
        learn_penalty=False, # this disables PID and instead learns the cost_penalty
        penalty_lr = 5e-2, # used if learn_penalty
    ),
    agent=dict(),
    model=dict(
        hidden_sizes=[512, 512],
        lstm_size=512,
        lstm_skip=True,
        constraint=True,  # must match algo.learn_c_value
        normalize_observation=True,
        var_clip=1e-6,
    ),
    runner=dict(
        n_steps=100e6,
        log_interval_steps=30000*4,
    ),
)


configs["COMMONROAD"] = config # previously called LSTM
configs["SAFETYGYM"] = config_orig # previously called LSTM_ORIG
configs["CRMONITOR"] = config_monitor # previously called LSTM_MONITOR
