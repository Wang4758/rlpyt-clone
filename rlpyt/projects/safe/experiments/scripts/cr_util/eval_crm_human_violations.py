""" Generates the compliance evaluation data for the humans. 
    The result .csv of this script is meant to be used by
    ../plot/plot_traffic_rule_vios.py. There should be commented out lines
    at the end of that script where it loads the csv instead of the normal
    evaluation results.
"""

import os
import pathlib
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.prediction.prediction import Prediction, TrajectoryPrediction
import crmonitor
from crmonitor.common.helper import create_ego_vehicle_param, create_other_vehicles_param, create_scenario_vehicles, create_simulation_param, load_yaml
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.world_state import WorldState
from crmonitor.evaluation.evaluation import RuleSetEvaluator
from crmonitor.monitor.rtamt_monitor_stl import OutputType
import pandas as pd
import numpy as np
from argparse import ArgumentParser
import logging

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger('main')

def analyze_scenario(p: pathlib.Path):
    LOGGER.info(f'Processing {p.as_posix()}')
    scenario, pps = CommonRoadFileReader(p.as_posix()).open()
    ego_obstacle = None
    pp_id = list(pps.planning_problem_dict.values())[0].planning_problem_id

    # find ego in dynamic obstacles
    for do in scenario.dynamic_obstacles:
        if do.obstacle_id == pp_id:
            ego_obstacle = do
            break

    assert ego_obstacle is not None, "Did you modify the dataset-converter? Read the note at the top of main()."

    # update dynamic obstacles with lanelet assignments
    obs_id_map = dict()
    old_obstacles = list()
    for dynamic_obstacle in list(scenario.dynamic_obstacles):
        old_obstacles.append(dynamic_obstacle)
        dynamic_obstacle_shape = dynamic_obstacle.obstacle_shape
        dynamic_obstacle_initial_state = dynamic_obstacle.prediction.trajectory.state_list[0]
        rotated_shape = dynamic_obstacle_shape.rotate_translate_local(
            dynamic_obstacle_initial_state.position, 
            dynamic_obstacle_initial_state.orientation
        )
        initial_shape_lanelet_ids = set(scenario.lanelet_network.find_lanelet_by_shape(rotated_shape))
        initial_center_lanelet_ids = set(scenario.lanelet_network.find_lanelet_by_position([dynamic_obstacle_initial_state.position])[0])
        assert initial_center_lanelet_ids is not None
        assert initial_shape_lanelet_ids is not None

        shape_lanlet_ids = dict()
        center_lanlet_ids = dict()
        for state in dynamic_obstacle.prediction.trajectory.state_list:
            rotated_shape = dynamic_obstacle_shape.rotate_translate_local(
                state.position, 
                state.orientation
            )
            shape_lanlet_ids[state.time_step] = set(
                scenario.lanelet_network.find_lanelet_by_shape(rotated_shape)
            )
            center_lanlet_ids[state.time_step] = set(
                scenario.lanelet_network.find_lanelet_by_position([state.position])[0]
            )

        prediction = TrajectoryPrediction(
            shape=dynamic_obstacle_shape,
            trajectory=dynamic_obstacle.prediction.trajectory,
            center_lanelet_assignment=center_lanlet_ids,
            shape_lanelet_assignment=shape_lanlet_ids
        )

        new_id = scenario.generate_object_id()
        obs_id_map[dynamic_obstacle.obstacle_id] = new_id
        new_obs = DynamicObstacle(
            obstacle_id=new_id,
            obstacle_shape=dynamic_obstacle.obstacle_shape,
            obstacle_type=dynamic_obstacle.obstacle_type,
            initial_state=dynamic_obstacle_initial_state,
            prediction=prediction,
            initial_center_lanelet_ids=initial_center_lanelet_ids,
            initial_shape_lanelet_ids=initial_shape_lanelet_ids,
            signal_series=None,
            initial_signal_state=None,
        )
        scenario.add_objects(new_obs)

    # delete old obstacles without lanelet assignments
    for old_obs in old_obstacles:
        scenario.remove_obstacle(old_obs)

    # make sure no buggy highD acceleration gets used
    for o in scenario.dynamic_obstacles:
        for s in o.prediction.trajectory.state_list:
            if hasattr(s, "acceleration"):
                del s.acceleration

    # create evaluator and evaluate
    ego_id = obs_id_map[ego_obstacle.obstacle_id]
    world_state = WorldState.create_from_scenario(scenario, ego_obs_id=ego_id)
    rule_set_evaluator = RuleSetEvaluator.create_from_config(['R_G1','R_G2','R_G3'], dt=scenario.dt, output_type=OutputType.OUTPUT_ROBUSTNESS)
    rule_robustness, _ = rule_set_evaluator.evaluate_incremental(world_state, to_pandas=False)

    # create output data in same format as the model eval script
    eval_res = dict(
        NViolationsRG1 = 0,
        NViolationsRG2 = 0,
        NViolationsRG3 = 0,
        NTrafficRuleViolations = 0,
    )
    for _, robs in rule_robustness.items():
        eval_res['NViolationsRG1'] += int(robs['R_G1'] < 0)
        eval_res['NViolationsRG2'] += int(robs['R_G2'] < 0)
        eval_res['NViolationsRG3'] += int(robs['R_G3'] < 0)
        eval_res['NTrafficRuleViolations'] += min(1, int(robs['R_G1'] < 0) + int(robs['R_G2'] < 0) + int(robs['R_G3'] < 0))

    eval_res['Length'] = len(rule_robustness.items())
    eval_res['Actions'] = 0
    eval_res['ScenarioId'] = scenario.benchmark_id

    return {scenario.benchmark_id: eval_res}


def main(p: pathlib.Path):

    """
        IMPORTANT NOTE: In the dataset-converter repo, need to change the creation of planning_problem_id if keep_ego == True:

        In method: def generate_planning_problem(...):

        if not keep_ego:
            planning_problem_id = dynamic_obstacle_selected.obstacle_id
            scenario.remove_obstacle(dynamic_obstacle_selected)
        else:
            planning_problem_id = dynamic_obstacle_selected.obstacle_id   <<< CHANGE TO THIS, so we can find the obs from the pp_id
    """

    from multiprocessing import Pool

    paths = list(p.glob('*.xml'))
    # results = [analyze_scenario(p) for p in paths]
    with Pool(128) as pool:
        results = pool.map(analyze_scenario, paths)

    result = dict()
    for r in results:
        result.update(r)

    df = pd.DataFrame.from_dict(result, orient='index')
    df.to_csv('human_eval_test.csv', index=False)

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--source-dir', type=pathlib.Path, default='/home/pillmayerc/datasets/highD_new_keepego/xmls')
    args = p.parse_args()
    main(p = args.source_dir)