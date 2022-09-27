"""Only used for paper related things"""

from typing import List
import joblib
import numpy as np
import pandas as pd
import gzip
from pathlib import Path

def gather_additional_info(benchmarkId: str, actions: List[np.ndarray]):
    """Rerun the actions in the environment -> Can log additional env infos here without redoing the whole evaluation"""
    import gym
    import gym_monitor

    env = gym.make(
        "cr-monitor-v0",
        test_env=True,
        play=True,
        logging_mode='INFO',
        max_problems=-1,
        mp_world_size=1, 
        mp_rank = 0,
        active_rules = ["R_G1", "R_G2", "R_G3"],
        scenario_filter_file_path='/home/pillmayerc/mth/rlpyt/rlpyt/projects/safe/experiments/scripts/cr_util/results_R_G1R_G2R_G3_max_acc_ALLokayids.txt',
        load_problems_on_demand=True,
    )

    r_g1s = []
    r_g2s = []
    r_g3s = []
    ego_states = []
    o = env.reset(benchmark_id=benchmarkId)
    for a in actions:
        o, r, d, i = env.step(np.array(a))
        ego_states.append(env.ego_state_list[-1])
        r_g1s.append(float(i.get('R_G1')))
        r_g2s.append(float(i.get('R_G2')))
        r_g3s.append(float(i.get('R_G3')))


    assert d, "not done at the end?"

    return dict(
        ego_states = ego_states,
        r_g1s = r_g1s,
        r_g2s = r_g2s,
        r_g3s = r_g3s,
    )

def keep_cols(df: pd.DataFrame, cols):
    keep_set = set(cols)
    drop_set = set(df.columns) - keep_set
    return df.drop(columns=list(drop_set))

def load(p:Path) -> pd.DataFrame:
    dfs = list()
    for compressed in p.glob('eval_results*_new.json.gz'):
        with gzip.open(compressed.as_posix(), "rb") as f:
            df = pd.read_json(f)
            df = df[["BenchmarkId", "NTrafficRuleViolations", "IsGoalReached", "NViolationsRG1", "NViolationsRG2", "NViolationsRG3", "Actions"]]
        dfs.append(df)
    return pd.concat(dfs)

def load_joblib(p:Path) -> pd.DataFrame:
    dfs = list()
    for i, compressed in enumerate(p.glob('eval_results*_joblib.gz')):
        print(f"Loading {i}")
        traj_infos = joblib.load(compressed.as_posix())
        df = pd.DataFrame.from_dict(traj_infos)
        # temporary verification code
        # first_row: pd.Series = next(df.iterrows())[1]
        # print(type(first_row))
        # for idx, value in first_row.iteritems():
        #     if type(value) == list:
        #         print(f'{idx} is List[{type(value[0])}]')
        #     else:
        #         print(f'{idx} is {type(value)}')
        # for id, row in df.iterrows():
        #     actions = row["Actions"]
        #     r_g1s = row["R_G1s"]
        #     r_g1 = row["R_G1"]
        #     assert r_g1 == sum(r_g1s)
        # one_state = df.iloc[0]["EgoStates"][0]
        # print(type(one_state), one_state)
        dfs.append(df)
    return pd.concat(dfs)

def extract_benchmark(df: pd.DataFrame, id: str) -> pd.DataFrame:
    return df[df["BenchmarkId"] == id]

def compute_termination_table(ours: pd.DataFrame, wcsac: pd.DataFrame):
    ours = ours[["IsGoalReached", "IsCollision", "IsOffroad", "IsTimeout", "Method"]]
    wcsac = wcsac[["IsGoalReached", "IsCollision", "IsOffroad", "IsTimeout", "Method"]]

    df = pd.concat([ours, wcsac]).reset_index()
    termination_reasons = df.groupby(["Method"]).mean().reset_index()
    print(termination_reasons.round(3))

def main():
    our_path = Path('/home/pillmayerc/mth/data_to_keep/01_paper_data/crm_dgae_7.5cl')
    wcsac_path = Path('/home/pillmayerc/mth/data_to_keep/01_paper_data/wcsac_crm_0.5and0.9_2seeds')

    ours0_5 = load_joblib(our_path.joinpath('0.5wc_alpha'))
    ours0_5["Method"] = "Ours-0.5"
    ours0_9 = load_joblib(our_path.joinpath('0.9wc_alpha'))
    ours0_9["Method"] = "Ours-0.9"
    ours = pd.concat([ours0_5, ours0_9])
    wcsac0_5 = load_joblib(wcsac_path.joinpath('0.5wc_alpha'))
    wcsac0_5["Method"] = "WCSAC-0.5"
    wcsac0_9 = load_joblib(wcsac_path.joinpath('0.9wc_alpha'))
    wcsac0_9["Method"] = "WCSAC-0.9"
    wcsac = pd.concat([wcsac0_5, wcsac0_9])

    compute_termination_table(ours, wcsac)
    
    # print(ours)
    # print(wcsac)

    # wcsac_ext = extract_benchmark(wcsac, "DEU_LocationALower-35_20_T-1")
    # wcsac_ext.to_json("traj_comparison/wcsac0.5_DEU_LocationALower-35_20_T-1.json")
    # ours_ext = extract_benchmark(ours, "DEU_LocationALower-35_20_T-1")
    # ours_ext.to_json("traj_comparison/ours0.5_DEU_LocationALower-35_20_T-1.json")
    exit()

    ours = ours.set_index("BenchmarkId")
    wcsac = wcsac.set_index("BenchmarkId")
    joined = ours.drop(columns=["Actions"]).join(wcsac.drop(columns=["Actions"]), lsuffix="_ours", rsuffix="_wcsac")
    joined["Difference"] = joined["NTrafficRuleViolations_wcsac"] - joined["NTrafficRuleViolations_ours"]
    joined = joined.sort_values("Difference", ascending=False)
    joined = joined.query("`IsGoalReached_ours` == True and `IsGoalReached_wcsac` == True")
    joined = joined.reset_index()
    joined.to_csv("wcsac_vs_ours_nviolations.csv", index=False)
    # joined = joined.query("`NTrafficRuleViolations_ours` > 0")
    print(joined.head(50))
    averages = joined.mean()
    print(averages)


if __name__ == "__main__":
    main()