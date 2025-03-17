import os
import sys
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

# ray 관련 라이브러리 임포트
import ray
from ray import tune
from ray.tune.registry import register_env

# 프로젝트 루트 디렉토리 설정
project_root = "/home/youngjins/project/belief_trading"
os.chdir(project_root)

# lib 하위 경로 추가
sys.path.append("/home/youngjins/project/belief_trading/lib/")
sys.path.append("/home/youngjins/project/belief_trading/lib/abides_jpmc_public")

# 환경 임포트
import gym
from abides_gym.envs.markets_execution_environment_v0 import (
    SubGymMarketsExecutionEnv_v0,
)

# 경로를 직접 지정하여 모듈 로드
import importlib.util
spec = importlib.util.spec_from_file_location("utils", os.path.join(project_root, "lib/utils/__init__.py"))
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

# utils 모듈에서 필요한 함수 가져오기
gym_types = utils.gym_types
flatten_dict = utils.flatten_dict

# 매개 변수 선언
seed = 0
gym_type_str = "markets-execution-v0"

np.random.seed(seed)

gym_type_abb = gym_types[gym_type_str]

register_env(
    gym_type_str,
    lambda config: SubGymMarketsExecutionEnv_v0(**config),
)

# policy 정의
class policyPassive:
    """
    패시브 정책은 항상 1을 반환 (시장 가격에 영향을 최소화, 소극적으로 주문을 실행하는 전략)
    """

    def __init__(self):
        self.name = "passive"

    def get_action(self, state):
        return 1


class policyAggressive:
    """
    공격적 정책은 항상 0을 반환 (적극적으로 주문을 처리하는 전략, 시장 충격이 클 수 있음)
    """

    def __init__(self):
        self.name = "aggressive"

    def get_action(self, state):
        return 0


class policyRandom:
    """
    무작위 정책은 0과 1 중에서 무작위로 선택 (패시브와 공격적 전략을 무작위로 혼합하는 방식)
    """

    def __init__(self):
        self.name = "random"

    def get_action(self, state):
        return np.random.choice([0, 1])


class policyRandomWithNoAction:
    """
    완전 무작위 정책은 -2 ~ 2 중에서 무작위로 선택 (관망하며 시장 상황을 지켜보는 옵션을 포함)
    """

    def __init__(self):
        self.name = "random_no_action"

    def get_action(self, state):
        action = np.random.choice([-2, -1, 0, 1, 2])
        if state[0] < 0 and action < 0:
            action = 0
        return action
    
# 강화학습 관련 변수 선언
env = gym.make(
    gym_type_str,
    background_config="rmsc04",
    timestep_duration="10S",
    execution_window="04:00:00",
    parent_order_size=2000,
    order_fixed_size=50,
    not_enough_reward_update=-100,  # penalty
    debug_mode=True
)

env.seed(seed)

target_agent = policyRandomWithNoAction()

# 시뮬레이션 시작
state = env.reset()
done = False
episode_reward = 0

# 데이터 추적을 위한 히스토리 딕셔너리 초기화
history_dict = {
    "mid_price": [],
    "best_bid": [],
    "best_ask": [],
    "reward": [],
    "cumulative_reward": [],
    "pnl": [],
    "holdings": [],
    "time": [],
    "action": [],
    "order_size": [],  # debuging
    "direction": [], # debuging
    "executed": [], # debuging
    "executed_size": [], # debuging
}

# 1개 에피소드 실행 (소요 시간 측정)
start_time = time.time()    
while not done:
    action_ = target_agent.get_action(state)
    # env.direction action_이 양수면, "BUY", 음수면, "SELL", "HOLD"는 고려 안함
    if action_ > 0:
        env.direction = "BUY"
    elif action_ < 0:
        env.direction = "SELL"
    action = abs(action_)
    state, reward, done, info = env.step(action)
    episode_reward += reward
    
    # 현재 상태에서 필요한 정보 추출
    mid_price = (info["best_bid"] + info["best_ask"]) / 2  # 중간 가격 계산
    history_dict["mid_price"].append(mid_price)
    history_dict["best_bid"].append(info["best_bid"])
    history_dict["best_ask"].append(info["best_ask"])
    history_dict["reward"].append(reward)
    history_dict["cumulative_reward"].append(episode_reward)
    history_dict["pnl"].append(info["pnl"])
    history_dict["holdings"].append(info["holdings"])
    history_dict["time"].append(info["current_time"])
    history_dict["action"].append(action_)
    history_dict["order_size"].append(env.parent_order_size)  # debuging
    history_dict["direction"].append(env.direction)  # debuging
    # history_dict["executed"].append(info["orders_executed"])  # debuging
    # history_dict["executed_size"].append(env.metrics_tracker.executed_quantity)  # debuging
                
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")

# could add a few more...
output = flatten_dict(info)
output["episode_reward"] = episode_reward
output["name"] = target_agent.name
