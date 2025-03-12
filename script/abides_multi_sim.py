"""
ABIDES-Gym을 사용한 멀티 봇 시뮬레이션
다양한 전략을 가진 봇들의 시뮬레이션을 병렬로 실행하는 코드
"""

# 필요한 라이브러리 임포트
import gym
import numpy as np
import pandas as pd
import time
import psutil
from multiprocessing import Pool, cpu_count
from collections.abc import MutableMapping
import abides_gym


# 전략 클래스들 정의
class PolicyPassive:
    """항상 passive 액션(1)을 반환하는 전략"""

    def __init__(self):
        self.name = "passive"

    def get_action(self, state):
        return 1


class PolicyAggressive:
    """항상 aggressive 액션(0)을 반환하는 전략"""

    def __init__(self):
        self.name = "aggressive"

    def get_action(self, state):
        return 0


class PolicyRandom:
    """0과 1 중 무작위로 액션을 선택하는 전략"""

    def __init__(self):
        self.name = "random"

    def get_action(self, state):
        return np.random.choice([0, 1])


class PolicyRandomWithNoAction:
    """0, 1, 2 중 무작위로 액션을 선택하는 전략 (2는 no action)"""

    def __init__(self):
        self.name = "random_no_action"

    def get_action(self, state):
        return np.random.choice([0, 1, 2])


# 기존 정책 클래스들
class PolicyPassive:
    """항상 passive 액션(1)을 반환하는 전략"""

    def __init__(self):
        self.name = "passive"

    def get_action(self, state):
        return 1


class PolicyAggressive:
    """항상 aggressive 액션(0)을 반환하는 전략"""

    def __init__(self):
        self.name = "aggressive"

    def get_action(self, state):
        return 0


class PolicyRandom:
    """0과 1 중 무작위로 액션을 선택하는 전략"""

    def __init__(self):
        self.name = "random"

    def get_action(self, state):
        return np.random.choice([0, 1])


class PolicyRandomWithNoAction:
    """0, 1, 2 중 무작위로 액션을 선택하는 전략 (2는 no action)"""

    def __init__(self):
        self.name = "random_no_action"

    def get_action(self, state):
        return np.random.choice([0, 1, 2])


# Evan et al (2024) 논문 참고
class PolicyRiskAverse:
    """
    위험 회피 투자자 전략 - 논문 3.2 섹션에서 영감
    높은 리스크 계수를 가지며, 위험을 피하는 경향이 있음
    시장이 불안정할 때 passive 전략을 선호
    """

    def __init__(self, risk_coefficient=5.0):
        self.name = "risk_averse"
        self.risk_coefficient = risk_coefficient  # 높은 값은 더 위험 회피적
        self.risk_threshold = 0.6  # 위험 임계값

    def get_action(self, state):
        # 시장 변동성이 높을 때 passive 행동 선호
        # state에서 시장 변동성을 추정
        market_volatility = self._estimate_volatility(state)

        if market_volatility > self.risk_threshold:
            return 1  # passive
        else:
            # 변동성이 낮을 때는 특정 조건에 따라 결정
            return 0 if np.random.random() < 0.3 else 1

    def _estimate_volatility(self, state):
        # 실제 구현에서는 state에서 시장 변동성을 추정하는 로직 구현
        # 지금은 간단히 random 값으로 대체
        return np.random.random()


class PolicyRiskTolerant:
    """
    위험 감수 투자자 전략 - 논문 3.2 섹션에서 영감
    낮은 리스크 계수를 가지며, 높은 수익을 위해 위험을 감수하는 경향이 있음
    공격적인 전략을 선호하지만 시장 상황에 따라 조정
    """

    def __init__(self, risk_coefficient=1.5):
        self.name = "risk_tolerant"
        self.risk_coefficient = risk_coefficient  # 낮은 값은 위험 감수 성향
        self.expected_return_threshold = 0.4  # 기대 수익 임계값

    def get_action(self, state):
        # 기대 수익이 높을 때 aggressive 행동 선호
        expected_return = self._estimate_expected_return(state)

        if expected_return > self.expected_return_threshold:
            return 0  # aggressive
        else:
            # 기대 수익이 낮을 때는 상황에 따라 조정
            return 0 if np.random.random() < 0.7 else 1

    def _estimate_expected_return(self, state):
        # 실제 구현에서는 state에서 기대 수익을 추정하는 로직 구현
        # 지금은 간단히 random 값으로 대체
        return np.random.random()


class PolicyMarketFollower:
    """
    시장 추종 전략 - 논문의 "dealer" 역할에서 영감
    시장 포트폴리오에 따라 행동을 조정하는 전략
    시장 추세를 따르는 경향이 있음
    """

    def __init__(self):
        self.name = "market_follower"
        self.market_trend = 0  # 초기 시장 추세 (0: 중립, 1: 상승, -1: 하락)
        self.trend_memory = []  # 최근 시장 추세 기억
        self.memory_size = 5

    def get_action(self, state):
        # 시장 추세 업데이트
        self._update_market_trend(state)

        # 시장 추세에 따라 행동 결정
        if self.market_trend > 0:
            # 상승 추세일 때 공격적
            return 0
        elif self.market_trend < 0:
            # 하락 추세일 때 방어적
            return 1
        else:
            # 중립 추세일 때 균형
            return np.random.choice([0, 1])

    def _update_market_trend(self, state):
        # 실제 구현에서는 state에서 시장 추세를 추정
        # 지금은 간단한 모의 로직으로 대체

        # 가상의 시장 방향 (현실에서는 state에서 추출)
        current_direction = np.random.choice([-1, 0, 1])

        # 트렌드 메모리 업데이트
        self.trend_memory.append(current_direction)
        if len(self.trend_memory) > self.memory_size:
            self.trend_memory.pop(0)

        # 트렌드 계산 (평균)
        if len(self.trend_memory) > 0:
            self.market_trend = sum(self.trend_memory) / len(self.trend_memory)
        else:
            self.market_trend = 0


def generate_env(seed):
    """
    특정 파라미터로 환경을 생성하고 시드를 설정합니다.

    Args:
        seed (int): 환경의 랜덤 시드 값

    Returns:
        gym.Env: 설정된 시장 실행 환경
    """
    env = gym.make(
        "markets-execution-v0",
        background_config="rmsc04",
        timestep_duration="10S",
        execution_window="04:00:00",
        parent_order_size=20000,
        order_fixed_size=50,
        not_enough_reward_update=-100,
    )
    env.seed(seed)
    return env


def flatten_dict(d: MutableMapping, sep: str = ".") -> MutableMapping:
    """
    중첩된 딕셔너리를 평탄화합니다.

    Args:
        d (MutableMapping): 평탄화할 중첩 딕셔너리
        sep (str): 중첩 키를 구분할 구분자, 기본값은 '.'

    Returns:
        MutableMapping: 평탄화된 딕셔너리
    """
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient="records")
    return flat_dict


def run_episode(seed=None, policy=None):
    """
    주어진 시드와 정책으로 하나의 에피소드를 완전히 실행합니다.

    Args:
        seed (int, optional): 환경의 랜덤 시드 값
        policy (object): 행동을 결정하는 정책 객체, get_action 메서드가 있어야 함

    Returns:
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

    output = flatten_dict(info)
    output["episode_reward"] = episode_reward
    output["name"] = policy.name
    return output


def monitor_resources(duration=1.0):
    """
    CPU 및 메모리 사용량을 모니터링합니다.

    Args:
        duration (float): CPU 사용률을 측정할 간격(초)

    Returns:
        dict: CPU 및 메모리 사용 정보가 담긴 딕셔너리
    """
    cpu_percent = psutil.cpu_percent(interval=duration)
    memory_info = psutil.virtual_memory()
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory_info.percent,
        "memory_used_gb": memory_info.used / (1024**3),
    }


def wrap_run_episode(param):
    """
    run_episode 함수를 래핑하여 실행 시간을 측정하고 결과에 포함합니다.

    Args:
        param (dict): run_episode에 전달할 파라미터 딕셔너리

    Returns:
        dict: 실행 시간이 포함된 run_episode 함수의 결과
    """
    start_time = time.time()
    result = run_episode(**param)
    elapsed_time = time.time() - start_time
    result["processing_time"] = elapsed_time
    return result


def run_N_episode(N, use_parallel=True):
    """
    여러 정책에 대해 N개의 에피소드를 병렬 또는 순차로 실행합니다.

    Args:
        N (int): 각 정책별로 실행할 에피소드 수
        use_parallel (bool): 병렬 처리 사용 여부

    Returns:
        tuple: (결과 리스트, 실행 시간)
    """
    # 정책 정의
    policies = [
        PolicyAggressive(),
        PolicyRandom(),
        PolicyPassive(),
        PolicyRandomWithNoAction(),
    ]
    seeds = list(range(N))

    # 모든 정책과 시드 조합 생성
    tests = [{"policy": policy, "seed": seed} for policy in policies for seed in seeds]

    total_tasks = len(tests)
    print(f"총 {total_tasks}개의 에피소드를 실행합니다...")

    # 리소스 모니터링 - 시작 상태
    print("시작 상태의 시스템 리소스:")
    start_resources = monitor_resources(0.1)
    print(
        f"CPU 사용률: {start_resources['cpu_percent']}%, "
        f"메모리 사용률: {start_resources['memory_percent']}%, "
        f"사용된 메모리: {start_resources['memory_used_gb']:.2f}GB"
    )

    start_time = time.time()  # 시작 시간 기록

    if use_parallel:
        # 병렬 처리 코드
        try:
            num_processes = max(1, int(cpu_count() * 0.70))
            print(f"{num_processes}개의 프로세스를 사용합니다.")

            outputs = []
            resource_samples = []

            with Pool(processes=num_processes) as pool:
                for i, result in enumerate(
                    pool.imap_unordered(
                        wrap_run_episode,
                        tests,
                        chunksize=max(1, len(tests) // num_processes),
                    )
                ):
                    outputs.append(result)

                    # 중간 리소스 샘플링 (20%마다)
                    if (i + 1) % max(1, total_tasks // 5) == 0:
                        resource_samples.append(monitor_resources(0.1))

                    # 진행 상황 표시
                    if (i + 1) % max(1, total_tasks // 20) == 0 or (
                        i + 1
                    ) == total_tasks:
                        print(
                            f"진행률: {(i + 1) / total_tasks * 100:.1f}% ({i + 1}/{total_tasks})"
                        )
                        if resource_samples:
                            last_sample = resource_samples[-1]
                            print(
                                f"현재 CPU: {last_sample['cpu_percent']}%, "
                                f"메모리: {last_sample['memory_percent']}%"
                            )
        except KeyboardInterrupt:
            print("병렬 처리가 사용자에 의해 중단되었습니다.")
            elapsed_time = time.time() - start_time
            print(f"실행 시간: {elapsed_time:.2f}초")
            return outputs, elapsed_time
    else:
        # 순차 처리 코드
        outputs = []
        resource_samples = []
        try:
            for i, test in enumerate(tests):
                result = wrap_run_episode(test)
                outputs.append(result)

                # 중간 리소스 샘플링 (20%마다)
                if (i + 1) % max(1, total_tasks // 5) == 0:
                    resource_samples.append(monitor_resources(0.1))

                # 진행 상황 표시
                if (i + 1) % max(1, total_tasks // 20) == 0 or (i + 1) == total_tasks:
                    print(
                        f"진행률: {(i + 1) / total_tasks * 100:.1f}% ({i + 1}/{total_tasks})"
                    )
                    if resource_samples:
                        last_sample = resource_samples[-1]
                        print(
                            f"현재 CPU: {last_sample['cpu_percent']}%, "
                            f"메모리: {last_sample['memory_percent']}%"
                        )
        except KeyboardInterrupt:
            print("순차 처리가 사용자에 의해 중단되었습니다.")
            elapsed_time = time.time() - start_time
            print(f"실행 시간: {elapsed_time:.2f}초")
            return outputs, elapsed_time

    elapsed_time = time.time() - start_time  # 종료 시간 기록

    # 종료 상태 리소스 모니터링
    end_resources = monitor_resources(0.1)
    print(f"종료 상태의 시스템 리소스:")
    print(
        f"CPU 사용률: {end_resources['cpu_percent']}%, "
        f"메모리 사용률: {end_resources['memory_percent']}%, "
        f"사용된 메모리: {end_resources['memory_used_gb']:.2f}GB"
    )

    # 처리 시간 분석
    processing_times = [result["processing_time"] for result in outputs]
    avg_processing_time = sum(processing_times) / len(processing_times)
    max_processing_time = max(processing_times)
    min_processing_time = min(processing_times)

    print(f"총 실행 시간: {elapsed_time:.2f}초")
    print(f"작업당 평균 처리 시간: {avg_processing_time:.2f}초")
    print(f"작업당 최대 처리 시간: {max_processing_time:.2f}초")
    print(f"작업당 최소 처리 시간: {min_processing_time:.2f}초")

    return outputs, elapsed_time


if __name__ == "__main__":
    N = 2  # 각 정책별로 실행할 에피소드 수

    # 병렬 처리로 실행
    print("======== 병렬 처리 실행 ========")
    parallel_outputs, parallel_time = run_N_episode(N, use_parallel=True)

    print("\n======== 순차 처리 실행 ========")
    sequential_outputs, sequential_time = run_N_episode(N, use_parallel=False)

    # 속도 향상(Speedup) 계산
    speedup = sequential_time / parallel_time

    # 효율성(Efficiency) 계산
    num_processes = max(1, int(cpu_count() * 0.70))
    efficiency = speedup / num_processes

    print("\n===== 병렬 처리 효율성 분석 =====")
    print(f"순차 처리 시간: {sequential_time:.2f}초")
    print(f"병렬 처리 시간: {parallel_time:.2f}초")
    print(f"속도 향상(Speedup): {speedup:.2f}배")
    print(f"효율성(Efficiency): {efficiency:.2f} (1.0이 이상적)")
    print(f"사용된 프로세스 수: {num_processes}")

    # 결과 데이터를 데이터프레임으로 변환하여 분석
    parallel_df = pd.DataFrame(parallel_outputs)
    print("\n정책별 성능 분석:")
    print(parallel_df.groupby("name")["episode_reward"].mean())

    # 정책별 처리 시간 분석
    print("\n정책별 평균 처리 시간:")
    print(parallel_df.groupby("name")["processing_time"].mean())
