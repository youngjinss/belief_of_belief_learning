# ABIDES-Gym을 사용한 멀티 봇 시뮬레이션

# 필요한 라이브러리 임포트
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing import Pool, cpu_count # 병렬 처리를 위한 설정
from collections.abc import MutableMapping
from p_tqdm import p_map
from ray.util.multiprocessing import Pool

import abides_gym

# 전략 클래스
class policyPassive:
    def __init__(self):
        self.name = 'passive'
        
    def get_action(self, state):
        return 1
        
class policyAggressive:
    def __init__(self):
        self.name = 'aggressive'
        
    def get_action(self, state):
        return 0
    
class policyRandom:
    def __init__(self):
        self.name = 'random'
        
    def get_action(self, state):
        return np.random.choice([0, 1])
    
class policyRandomWithNoAction:
    def __init__(self):
        self.name = 'random_no_action'
        
    def get_action(self, state):
        return np.random.choice([0, 1, 2])


def generate_env(seed):
    """
    특정 파라미터로 환경을 생성하고 시드를 설정합니다.
    
    매개변수:
        seed (int): 환경의 랜덤 시드 값
        
    반환값:
        gym.Env: 설정된 시장 실행 환경
    """
    env = gym.make(
        "markets-execution-v0",
        background_config="rmsc04",
        timestep_duration="10S",
        execution_window= "04:00:00",
        parent_order_size= 20000,
        order_fixed_size= 50,
        not_enough_reward_update=-100)

    env.seed(seed)
    
    return env

def flatten_dict(d: MutableMapping, sep: str= '.') -> MutableMapping:
    """
    중첩된 딕셔너리를 평탄화합니다.
    
    매개변수:
        d (MutableMapping): 평탄화할 중첩 딕셔너리
        sep (str): 중첩 키를 구분할 구분자, 기본값은 '.'
        
    반환값:
        MutableMapping: 평탄화된 딕셔너리
    """
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict

def run_episode(seed = None, policy=None):
    """
    주어진 시드와 정책으로 하나의 에피소드를 완전히 실행합니다.
    
    매개변수:
        seed (int, optional): 환경의 랜덤 시드 값
        policy (object): 행동을 결정하는 정책 객체, get_action 메서드가 있어야 함
        
    반환값:
        dict: 에피소드 실행 결과 정보가 담긴 딕셔너리 (보상, 정책 이름 등 포함)
    """
    
    env = generate_env(seed)
    state = env.reset()
    done = False
    episode_reward = 0 
    
    while not done:
        action = policy.get_action(state)
        state, reward, done, info = env.step(action)
        episode_reward += reward
    
    #could add a few more... 
    output = flatten_dict(info) 
    output['episode_reward'] = episode_reward
    output['name'] = policy.name
    return output

def run_N_episode(N):
    """
    여러 정책에 대해 N개의 에피소드를 병렬로 실행합니다.
    
    참고: rllib 정책에는 아직 작동하지 않음 - pickle 오류 발생
    #https://stackoverflow.com/questions/28821910/how-to-get-around-the-pickling-error-of-python-multiprocessing-without-being-in
    
    rllib 정책은 다음 셀에서 병렬 처리 없이 실행해야 함
    
    매개변수:
        N (int): 각 정책별로 실행할 에피소드 수
        
    반환값:
        list: 모든 에피소드 실행 결과가 담긴 리스트
    """
    #define policies 
    policies = [policyAggressive(), policyRandom(), policyPassive(), policyRandomWithNoAction()]
    seeds = [i for i in range(N)]
    
    tests = [{"policy": policy, 'seed': seed} for policy in policies for seed in seeds]
    
    def wrap_run_episode(param):
        return run_episode(**param)
    
    
    # 키보드 인터럽트 처리를 위한 설정
    try:
        # 프로세스 수를 CPU 코어 수의 70%로 제한하여 시스템 부하 방지
        num_processes = max(1, int(cpu_count() * 0.70))
        
        with Pool(processes=num_processes) as pool:
            # chunksize 파라미터를 추가하여 작업 분배 최적화
            outputs = pool.map(wrap_run_episode, tests, chunksize=max(1, len(tests) // num_processes))
    except KeyboardInterrupt:
        print("병렬 처리가 사용자에 의해 중단되었습니다.")
        return []
    
    return outputs


if __name__ == "__main__":
    N = 50
    outputs = run_N_episode(N) 
