import os
import sys
import time
import numpy as np

# lib 하위 경로 추가
sys.path.append("/home/youngjins/project/belief_trading/lib/")
sys.path.append("/home/youngjins/project/belief_trading/lib/pymarketsim")

# 프로젝트 루트 디렉토리 설정
project_root = "/home/youngjins/project/belief_trading"
os.chdir(project_root)

# 경로를 직접 지정하여 모듈 로드
import importlib.util

# utils, policy 모듈 로드
spec = importlib.util.spec_from_file_location(
    "mm_wrapper",
    os.path.join(project_root, "lib/pymarketsim/marketsim/wrappers/MM_wrapper.py"),
)
mm_wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm_wrapper)

spec = importlib.util.spec_from_file_location(
    "policy", os.path.join(project_root, "lib/policy/__init__.py")
)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)

MMEnv = mm_wrapper.MMEnv
policyRandom1 = policy.policyRandom1

# 매개 변수 선언 -> 노이즈 트레이더
seed = 0
np.random.seed(seed)

num_assets = 1
normalizers = {"fundamental": 1.2e5, "invt": 10, "cash": 5e5}
beta_params = {"a_buy": 0.5, "b_buy": 0.5, "a_sell": 0.5, "b_sell": 0.5}

# 강화학습 관련 변수 선언
env = MMEnv(
    num_background_agents=25,
    sim_time=5000,
    num_assets=num_assets,
    lam=0.1,
    mean=1e5,
    r=0.05,
    shock_var=5e6,
    q_max=10,
    pv_var=5e6,
    shade=[250, 500],
    normalizers=normalizers,
    beta_params=beta_params,
)

target_agent = policyRandom1()

# 시뮬레이션 시작
state, info = env.reset()
done = False
episode_reward = 0
history_dict = dict()

# 1개 에피소드 실행 (소요 시간 측정)
start_time = time.time()
while not done:

    action = target_agent.get_action(state)

    state, reward, terminated, truncated, info = env.step(action)

    episode_reward += reward

    # # 현재 상태에서 필요한 정보 추출
    # history_dict["mid_price"].append((info["best_bid"] + info["best_ask"]) / 2)
    # history_dict["best_bid"].append(info["best_bid"])
    # history_dict["best_ask"].append(info["best_ask"])
    # history_dict["reward"].append(reward)
    # history_dict["cumulative_reward"].append(episode_reward)
    # history_dict["pnl"].append(info["pnl"])
    # history_dict["holdings"].append(info["holdings"])
    # history_dict["time"].append(info["current_time"])
    # history_dict["action"].append(action_)
    # history_dict["order_size"].append(env.parent_order_size)  # debuging
    # history_dict["direction"].append(env.direction)  # debuging

    print(state)

end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")
