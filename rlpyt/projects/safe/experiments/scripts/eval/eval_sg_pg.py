"""
    Exclusively created to evalutate CPO on commonroad. 
    It is mostly a copy of the relevant parts from the training script in the starter-agents repo + the model loading part.
"""

import gzip
import itertools
import json
from multiprocessing import Pool
import pathlib
import joblib
import numpy as np
from safe_rl.utils.running_mean_std import RunningMeanStd
import tensorflow as tf
import gym
from safe_rl.utils.logx import restore_tf_graph
from commonroad_rl.gym_commonroad.commonroad_env import CommonroadPlayNoScenarioException


def test_agent_pool(model_path: pathlib.Path, pool_size = 32):
    with Pool(pool_size) as pool:
        args = [
            (model_path, pool_size, i)
            for i in range(pool_size)
        ]
        results = pool.starmap(test_agent, args)
    return list(itertools.chain.from_iterable(results))


def test_agent(model_path: pathlib.Path, world_size = 1, rank = 0):

    test_env = gym.make(
        "commonroad-v1", 
        test_env=True, 
        play=True, 
        logging_mode='INFO', 
        max_problems=-1, 
        mp_world_size=world_size, 
        mp_rank = rank
    )

    state_dict = joblib.load(model_path.parent.joinpath('vars.pkl'))
    obs_rms = state_dict['obs_rms']

    # obs_rms = RunningMeanStd(shape = test_env.observation_space.sample().shape)

    sess = tf.Session()

    tf_graph_io: dict = restore_tf_graph(sess, model_path.as_posix())
    pi = tf_graph_io["pi"]
    x_ph = tf_graph_io["x"]

    max_ep_len = 1000    
    results = []

    while True:
        try:
            o, r, d, ep_ret, ep_cost, ep_len, ep_goals, = test_env.reset(), 0, False, 0, 0, 0, 0
            while not(d or (ep_len == max_ep_len)):
                
                o = (o - obs_rms.mean) / obs_rms.var
                a = sess.run(pi, feed_dict={x_ph: o[np.newaxis]})

                o, r, d, info = test_env.step(a)

                ep_ret += r
                ep_cost += info.get('cost', 0)
                ep_len += 1
                ep_goals += info.get('is_goal_reached', False) # commonroad

                if ep_goals > 0 or d:
                    if ep_goals > 0:
                        print('Goal!')
                    results.append({
                        "Length": ep_len,
                        "Return": ep_ret,
                        "Cost": ep_cost,
                        "IsGoalReached": info.get("is_goal_reached", 0),
                        "IsCollision": info.get("is_collision", 0),
                        "IsOffroad": info.get("is_off_road", 0),
                        "IsTimeout": info.get("is_time_out", 0),
                    })
        except CommonroadPlayNoScenarioException:
            # all done!
            break
    
    return results


if __name__ == '__main__':
    results = test_agent_pool(pathlib.Path('/home/pillmayerc/mth/data_to_keep/00_new/cr_cpo/run_2/simple_save'))
    results_file = '/home/pillmayerc/mth/data_to_keep/00_new/cr_cpo/eval_results2_new.json.gz'
    with gzip.open(results_file, 'wb') as f:
        print("Saving results!")
        data: str = json.dumps(results)
        f.write(data.encode('utf-8'))