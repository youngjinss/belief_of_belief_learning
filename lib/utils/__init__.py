# Daily Investor Environment
# holdings_t, imbalance_t, spread_t, directionFeature_t, R^k_t
# holdings_t: 실험 에이전트가 t 시점에 보유한 주식의 수
# imbalance_t = bids_volume / (bids_volume + asks_volume)로, 주문장의 첫 3단계 수준을 사용함. 호가 없음, 매도 호가 없음, 주문장 비어있음에 따라 각각 0, 1, 0.5로 설정됨
# spread_t = bestAsk_t - bestBid_t
# directionFeature_t = midPrice_t - lastTransactionPrice_t
# R^k_t = (r_t, ..., r_t-k+1): 중간 가격 차이의 시계열, r_t-i = mid_t-i - mid_t-i-1. 정의되지 않은 경우 0으로 설정됨. 기본값 k = 3
die = ["holdings_t", "imbalance_t", "spread_t", "directionFeature_t", "R^k_t"]

# Algorithmic Execution Environment
# holdingsPct_t, timePct_t, differencePct_t, imbalance5_t, imbalanceAll_t, priceImpact_t, spread_t, directionFeature_t, R^k_t
# holdingsPct_t = holdings_t / parentOrderSize: 실행 진행도
# timePct_t = (t - startingTime) / timeWindow: 시간 진행도
# differencePct_t = holdingsPct_t - timePct_t
# priceImpact_t = midPrice_t - entryPrice
# imbalance5_t와 imbalanceAll_t는 4.1.2에 정의된 것과 유사하지만 주문장의 첫 5단계와 모든 단계를 각각 사용
# spread_t, directionFeature_t, R^k_t는 4.1.2에 정의된 것과 동일하며, k = 3
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


# print all state name and value from state 2-d list using for loop (state와 die, aee의 길이가 안맞다면, R^k_t의 k > 1이므로, 이 부분에 대해서 k 숫자에 따른 프린터를 하도록 구현 추가)
def print_state(state: list, env_type: str) -> None:
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
