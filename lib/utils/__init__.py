import pandas as pd

from collections.abc import MutableMapping

# Daily Investor Environment
# holdings_t: 실험 에이전트가 t 시점에 보유한 주식의 수
# imbalance_t = bids_volume / (bids_volume + asks_volume)로, 주문장의 첫 3단계 수준을 사용함. 호가 없음, 매도 호가 없음, 주문장 비어있음에 따라 각각 0, 1, 0.5로 설정됨
# spread_t = bestAsk_t - bestBid_t
# directionFeature_t = midPrice_t - lastTransactionPrice_t
# R^k_t = (r_t, ..., r_t-k+1): 중간 가격 차이의 시계열, r_t-i = mid_t-i - mid_t-i-1. 정의되지 않은 경우 0으로 설정됨. 기본값 k = 3
die = ["holdings_t", "imbalance_t", "spread_t", "directionFeature_t", "R^k_t"]

# Algorithmic Execution Environment
# holdings_pct: 현재 보유 수량 / 총 주문량 (parent_order_size)
# time_pct: 경과 시간 / 총 실행 시간 (execution_window)
# diff_pct: 보유량 진행률과 시간 진행률의 차이 (holdings_pct - time_pct)
# imbalance_all: 전체 호가창에서의 매수-매도 불균형 지표
# imbalance_5: 5단계 깊이까지의 호가 불균형
# price_impact: 시장 진입 가격 대비 중간가격 변화 (매수/매도 방향에 따라 다름)
# spread: 최우선 매도호가와 매수호가의 차이
# direction_feature: 중간가격과 마지막 거래가격의 차이
# returns: 중간가격의 변화율(과거 데이터 포함)
aee = [
    "holdingsPct_t",
    "timePct_t",
    "differencePct_t",
    "imbalance5_t",
    "imbalanceAll_t",
    "priceImpact_t",
    "spread_t",
    "directionFeature_t",
    "R^k_t",
]

# gym의 종류 dictionary
gym_types = {
    "daily-investor-v0": "die",
    "markets-execution-v0": "aee",
}

# info 메모 (debug mode)
# "last_transaction": 마지막 거래 가격,
# "best_bid": 최우선 매수호가,
# "best_ask": 최우선 매도호가,
# "current_time": 현재 시간,
# "holdings": 현재 보유량,
# "parent_size": 총 주문량,
# "pnl": 손익,
# "reward": 정규화된 손익(pnl / parent_order_size)


def print_state(state: list, env_type: str) -> None:
    """
    - 주어진 상태(state) 리스트의 각 요소와 해당 환경 유형에 맞는 이름을 출력.
    - 환경 유형에 따라 die(Daily Investor Environment) 또는 aee(Algorithmic Execution Environment)의 상태 이름을 사용하여 상태 값을 출력
    - R^k_t의 경우 k가 1보다 클 때 여러 값이 있을 수 있으며, 이 경우 모든 값을 함께 출력합니다.

    Args:
        state (list): 출력할 상태 값들의 리스트
        env_type (str): 환경 유형 ("die" 또는 "aee")

    Raises:
        ValueError: 유효하지 않은 환경 유형이 제공될 경우 발생
    """
    if env_type == "die":
        for i in range(len(state)):
            if i < len(die) - 1:
                print(f"{die[i]}: {state[i]}")
            else:
                print(f"R^k_t: {state[i]}")
    elif env_type == "aee":
        for i in range(len(state)):
            if i < len(aee) - 1:
                print(f"{aee[i]}: {state[i]}")
            else:
                print(f"R^k_t: {state[i]}")
    else:
        raise ValueError(f"Invalid environment type: {env_type}")


def flatten_dict(d: MutableMapping, sep: str = ".") -> MutableMapping:
    """
    - 중첩된 딕셔너리를 평탄화(flatten)
    - 중첩된 딕셔너리의 키를 구분자(separator)로 연결하여 단일 레벨의 딕셔너리로 변환

    Args:
        d (MutableMapping): 평탄화할 중첩 딕셔너리
        sep (str, optional): 중첩 키를 연결할 때 사용할 구분자. 기본값은 "."

    Returns:
        MutableMapping: 평탄화된 딕셔너리
    """
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient="records")
    return flat_dict
