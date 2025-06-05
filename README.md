## exp2 버전 기록
[v0] synthetic data 연구

## exp1 버전 기록 (archive) 
[v0] 시뮬레이션 기반 추론 연구
[v1] 데이터 기반 예측 연구
[v2] 데이터 탐색 (e.g., WRDS, COT, Binance, 등등)
[v3] Binance TAQ 데이터 기준 데이터 정제 및 확인
[v4] concatenate_ohlcv_w_pd.ipynb -> 개인, 기관별 klines(ohlcv) + position distribuiton concat
[v5] train /inference model

[v6] 수식 결합 피드백 -> construction.py는 너무 복잡함 (돌리기엔 부적합))

## 학습 기록
1. [data 1, v5] 30m, i_df, -i_df 수정 후 기록
2025-04-13 16:22:36: benchmark
2025-04-13 16:24:02: proposed 

2. [data 2, v5] 15m, top 20%
2025-04-14 00:36:31 proposed
2025-04-14 00:37:21 benchmark

## 데이터 가공 기록
data 1: 30m, top 25%
data 2: 15m, top 20%
2025-04-13 23:37:00: 15분(900000) 가공
