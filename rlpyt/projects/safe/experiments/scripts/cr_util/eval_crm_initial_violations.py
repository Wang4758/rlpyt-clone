""" 
    Evaluate traffic rule robustness for the initial states of the commenroad scenarios
"""

import json
from multiprocessing import Pool
from pathlib import Path
from typing import List
from crmonitor.predicates.predicate import BasePredicateEvaluator, PredAbruptBreaking
import numpy as np
import pandas as pd
import gym
import gym_monitor
from commonroad_rl.gym_commonroad.commonroad_env import CommonroadPlayNoScenarioException
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from ruamel.yaml.comments import CommentedMap
from pprint import pprint

def brakes_abruptly(a, a_abrupt = -2.) -> float:
    return a_abrupt - a

def brakes_abruptly_rel(a_k, a_p, a_abrupt = -2.) -> float:
    return - a_k + a_p + a_abrupt

def eval_rules(rules, rank=0, ws=1, scenarios = 'train', max_problems = -1):

    assert scenarios == 'train' or scenarios == 'test'

    print(f'Evaluating rule {rules} on {scenarios} scenarios')

    # dict of scenario id -> initial state robustness
    results = dict()

    env = gym.make(
        'cr-monitor-v0',
        active_rules = rules,
        test_env=True,
        test_with_train_scenarios = (scenarios == 'train'),
        play=True,
        max_problems=max_problems,
        mp_rank=rank,
        mp_world_size=ws,
        load_problems_on_demand=True,
        flatten_observation=False,
    )

    while True:
        try:
            zero_action = np.array([1.0, 0.0])
            env.reset()

            o, _, d, info = env.step(zero_action)
            if d:
                continue
            o, _, d, info = env.step(zero_action)
            if d:
                continue
            rule_robustness = {
                rule: float(info.get(rule))
                for rule in rules
            }
            scenario_id = env.benchmark_id
            results[scenario_id] = rule_robustness
            # check_robustness(rank, results, env, zero_action, o, scenario_id)
            
                
        except CommonroadPlayNoScenarioException:
            # done with all scenarios
            break

        except AssertionError as e:
            with open(f'error_dump_r{rank}_{scenario_id}.txt', 'wt') as f:
                f.write(str(e))
                pprint(locals(), f)

    return results

def check_robustness(rank, results, env, zero_action, o, scenario_id):
    """ Used for debugging the robustness values when the acceleration values were a problem """
    assert env.current_step in env.debug_predicate_robustness, f'Predicate robs does not have key {env.current_step}, it only has keys {list(env.debug_predicate_robustness.keys())}, {scenario_id}'
    predicate_robustness: dict = env.debug_predicate_robustness[env.current_step]
    ego_a_monitor = env.debug_ego_a
    ego_a_cr = o['a_ego'][0] # pm model has 2 dim ego a obs
    a_rel_left_lead = o['lane_based_a_rel'][3]
    a_rel_same_lead = o['lane_based_a_rel'][4]
    a_rel_right_lead = o['lane_based_a_rel'][5]

    leading_exists = (o['lane_based_p_rel'][3] < 100.0 or
                            o['lane_based_p_rel'][4] < 100.0 or
                            o['lane_based_p_rel'][5] < 100.0)

    assert np.allclose(ego_a_monitor, ego_a_cr), f'Acceleration discrepancy: {ego_a_monitor} and {ego_a_cr}, action {zero_action}, scenario {scenario_id}'
            
            # this should always be the same as the monitor one as long as the assertion above always passes
            # which it seems to be doing
    cr_brakes_abruptly = brakes_abruptly(ego_a_cr)
            # a_rel_lead = a_obstacle - ego_state.acceleration (in surrouding_obs..py)
    lead_left_a = a_rel_left_lead + ego_a_cr
    lead_same_a = a_rel_same_lead + ego_a_cr
    lead_right_a = a_rel_right_lead + ego_a_cr
    cr_brakes_abruptly_rel = [
                brakes_abruptly_rel(ego_a_cr, lead_left_a),
                brakes_abruptly_rel(ego_a_cr, lead_same_a), # only use same lane, preceding pred also does that
                brakes_abruptly_rel(ego_a_cr, lead_right_a)
            ]

    G2_robs = predicate_robustness['R_G2']
            # this is only -1 or 1 because 
            # predicate_node.io_type == IOType.INPUT and self.output_type == OutputType.OUTPUT_ROBUSTNESS
            # is true --> only compare the signs and not the actual values
    monitor_brake_abruptly = G2_robs["brakes_abruptly__a0_i"]
    monitor_brake_abruptly_rel = G2_robs["rel_brakes_abruptly__a0_a1_i"]
    monitor_has_preceding = G2_robs["precedes__a0_a1"] >= 0.0 # a1 DIRECTLY precedes a0 IN THE SAME LANE! (see paper)

    acc_scaler = PredAbruptBreaking(config=CommentedMap())
    cr_brakes_abruptly = acc_scaler._scale_acc(cr_brakes_abruptly)
    cr_brakes_abruptly_rel = [acc_scaler._scale_acc(e) for e in cr_brakes_abruptly_rel]

            # assert np.sign(cr_brakes_abruptly) == np.sign(monitor_brake_abruptly), f"brakes_abruptly not equal! {scenario_id}, {zero_action}"
    if monitor_has_preceding:
                # only makes sense to check the preceding brake stuff if there is a preceding vehicle...
        rel_okay = (not leading_exists) or np.any(np.sign(cr_brakes_abruptly_rel) == np.sign(monitor_brake_abruptly_rel))
        if not rel_okay:
            with open(f'error_dump_r{rank}_{scenario_id}.txt', 'wt') as f:
                pprint(locals(), f)
                f.write('\n')
                f.write(f'Leading exists: {leading_exists}, {cr_brakes_abruptly_rel}, {monitor_brake_abruptly_rel}')

    results[scenario_id].update(G2_robs)
    results[scenario_id].update({'a_ego': ego_a_cr})
    results[scenario_id].update({'a_lead_rel': a_rel_same_lead})

def eval_rulesx(rules = ['R_G1', 'R_G2' ,'R_G3'], scenarios='train'):

    n_cpus = 32
    with Pool(n_cpus) as pool:
        args = [
            (rules, i, n_cpus, scenarios)
            for i in range(n_cpus)
        ]
        results = pool.starmap(eval_rules, args)

    result = {}
    for r in results:
        result.update(r)

    df = pd.DataFrame.from_dict(result, orient='index').reset_index().rename(columns={'index': 'Id'})
    df.to_csv(f"results_{''.join(rules)}_max_acc_{scenarios}.csv", index=False)

def analyze_results(fp):
    fp = Path(fp)
    print(f'============== Checking {fp.stem} ==============')
    if 'json' in fp.suffix:
        df = pd.read_json(fp, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Id'}, inplace=True)
    else:
        df = pd.read_csv(fp)
        df.sort_values('Id').to_csv(fp, index=False)
    # selected = df[df['R_G3'] < 0]
    # print(selected.sort_values('R_G3').tail(10))
    # print(df.sort_values(by=['R_G3']).head(200))
    violated = df.applymap(lambda v: v < 0 if not isinstance(v, str) else v)
    all_okay = violated.query('not (R_G1 or R_G2 or R_G3)')
    all_okay_ids = all_okay['Id']
    with open(fp.parent.joinpath(fp.stem + 'okayids.txt'), 'wt') as f:
        lines = [l + '\n' for l in list(all_okay_ids)]
        f.writelines(lines)
    all_okay_ratio = len(all_okay) / len(violated)
    print(f'All Okay Ratio = {round(all_okay_ratio, 2)}')
    print("====== Rule is violated ======")
    print(violated[['R_G1', 'R_G2' ,'R_G3']].apply(lambda x: round(x.value_counts(normalize=True), 5)).T.stack())

def compare_g3_violations(fps, types: List[str]):
    dfs = []
    for fp, type_ in zip(fps, types):
        tmp = pd.read_csv(fp)
        tmp['Type'] = type_.capitalize()
        dfs.append(tmp)

    df = pd.concat(dfs)
    df['Violated'] = df['R_G3'] < 0
    df['Not Violated'] = df['R_G3'] >= 0

    sorted = df.sort_values(by='R_G3')
    print(sorted.head(50))

    violations_by_type = df[['Violated', 'Not Violated', 'Type']].groupby('Type').mean()
    """
        For training scenarios with default monitor config (expect the g3 rule changed to only one predicate...)

               Violated  Not Violated
        Type                         
        Brake  0.002404      0.997596
        Fov    0.002404      0.997596
        Lane   0.261058      0.738942           ---> we can see that almost all G3 violations are from the lane limit
        Type   0.000000      1.000000
    """
    print(violations_by_type)

"""
    Train:
    All Okay Ratio = 0.67
    ====== Rule is violated ======
    R_G1    False    0.970
            True     0.030
    R_G2    False    0.940
            True     0.060
    R_G3    False    0.722
            True     0.278
    
    Test:
    All Okay Ratio = 0.69
    ====== Rule is violated ======
    R_G1    False    0.958
            True     0.042
    R_G2    False    0.939
            True     0.061
    R_G3    False    0.749
            True     0.251


    WITH NEW OBSERVATION FOR ACCELERATION:
    All Okay Ratio = 0.7
====== Rule is violated ======
R_G1  False    0.96970
      True     0.03030
R_G2  False    1.00000
R_G3  False    0.72198
      True     0.27802
dtype: float64
============== Checking results_R_G1R_G2R_G3_new_a_obs_test ==============
All Okay Ratio = 0.72
====== Rule is violated ======
R_G1  False    0.95847
      True     0.04153
R_G2  False    1.00000
R_G3  False    0.74860
      True     0.25140
"""

if __name__ == '__main__':

    # eval_rules(rules=["R_G2"], max_problems=-1)
    # eval_rulesx(rules=["R_G1", "R_G2", "R_G3"], scenarios='train')
    # eval_rulesx(rules=["R_G1", "R_G2", "R_G3"], scenarios='test')

    """
        ============== Checking results_R_G1R_G2R_G3_max_acc_test ==============
        All Okay Ratio = 0.62
        ====== Rule is violated ======
        R_G1  False    0.93490
              True     0.06510
        R_G2  False    1.00000
        R_G3  False    0.65657
              True     0.34343
        ============== Checking results_R_G1R_G2R_G3_max_acc_train ==============
        All Okay Ratio = 0.62
        ====== Rule is violated ======
        R_G1  False    0.93843
              True     0.06157
        R_G2  False    1.00000
        R_G3  False    0.65705
              True     0.34295
    """
    analyze_results('results_R_G1R_G2R_G3_max_acc_test.csv')
    analyze_results('results_R_G1R_G2R_G3_max_acc_train.csv')

    # analyze_results('results_R_G2_new_a_obs_train.csv')
    # analyze_results('results_R_G2_new_a_obs_train.csv')
    # analyze_results('results_R_G1R_G2R_G3_65.csv')

    # compare_g3_violations([
    #     'results_R_G1R_G2R_G3_lane.csv',
    #     'results_R_G1R_G2R_G3_type.csv',
    #     'results_R_G1R_G2R_G3_brake.csv',
    #     'results_R_G1R_G2R_G3_fov.csv',
    #     ],
    #     ['lane', 'type', 'brake', 'fov']
    # )

    # analyze_results('results_R_G1R_G2R_G3_with_tolerance.csv')
    # analyze_results('results_R_G1R_G2R_G3_lane.csv')
    # analyze_results('results_R_G1R_G2R_G3_type.csv')
    # analyze_results('results_R_G1R_G2R_G3_brake.csv')
    # analyze_results('results_R_G1R_G2R_G3_fov.csv')