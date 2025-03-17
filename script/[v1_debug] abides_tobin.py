import os
import sys
import time
import numpy as np

# ray 관련 라이브러리 임포트
import ray
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

# utils, policy 모듈 로드
spec = importlib.util.spec_from_file_location(
    "utils", os.path.join(project_root, "lib/utils/__init__.py")
)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

spec = importlib.util.spec_from_file_location(
    "policy", os.path.join(project_root, "lib/policy/__init__.py")
)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)

gym_types = utils.gym_types
flatten_dict = utils.flatten_dict
history_dict = utils.history_dict
PolicyTobinToCAPM = policy.PolicyTobinToCAPM

# 매개 변수 선언
seed = 0
np.random.seed(seed)

# window size 선언
window_size = 20

# 환경 선언
gym_type_str = "markets-execution-v0"
gym_type_abb = gym_types[gym_type_str]
register_env(
    gym_type_str,
    lambda config: SubGymMarketsExecutionEnv_v0(**config),
)

# ray init
ray.shutdown()
ray.init()


# 강화학습 관련 변수 선언
env = gym.make(
    gym_type_str,
    background_config="rmsc04",
    timestep_duration="10S",
    execution_window="04:00:00",
    parent_order_size=2000,
    order_fixed_size=50,
    not_enough_reward_update=-100,  # penalty
    state_history_length=window_size,
    debug_mode=True,
)
env.seed(seed)

target_agent = PolicyTobinToCAPM(window_size=window_size)

# 시뮬레이션 시작
state = env.reset()
done = False
episode_reward = 0

# 1개 에피소드 실행 (소요 시간 측정)
start_time = time.time()
while not done:
    action_ = target_agent.get_action(state)
    env.direction = "BUY" if action_ > 0 else "SELL" if action_ < 0 else env.direction
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

end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")

# could add a few more...
output = flatten_dict(info)
output["episode_reward"] = episode_reward
output["name"] = target_agent.name
