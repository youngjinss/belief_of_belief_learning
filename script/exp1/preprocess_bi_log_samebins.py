# todo: telegram으로 오류, 결과 메세지 전송 하는 기능 추가


import os
import re
import numpy as np
import pandas as pd
import glob
from datetime import datetime
import logging
import argparse
import json
import csv

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# 명령줄 인자 파싱
parser = argparse.ArgumentParser(description="바이낸스 데이터 전처리 스크립트")
parser.add_argument(
    "--window_size",
    type=int,
    default=1800000,
    help="윈도우 크기(밀리초), 기본값: 1시간",
)
parser.add_argument(
    "--type_threshold",
    type=float,
    default=0.25,
    help="기관/개인 투자자 분류 임계값, 기본값: 0.25",
)
parser.add_argument(
    "--quantile", type=int, default=20, help="분위수 구간 수, 기본값: 20"
)
parser.add_argument(
    "--project_dir",
    type=str,
    default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    help="프로젝트 디렉토리 경로",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data/binance/futures/um/monthly/aggTrades/BTCUSDT",
    ),
    help="입력 데이터 디렉토리 경로",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data/binance/futures/um/monthly/position_distribution",
    ),
    help="출력 데이터 디렉토리 경로",
)
parser.add_argument(
    "--log_dir",
    type=str,
    default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "log",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ),
    help="로그 파일 저장 디렉토리 경로",
)
args = parser.parse_args()

# 중요 변수들 선언
window_size = args.window_size  # 1시간(1h)을 밀리초로 변환
type_threshold = args.type_threshold
quantile = args.quantile
quantile_bins = np.linspace(-1, 1, 2 * quantile + 1)
usecols = [
    "price",
    "quantity",
    "transact_time",
    "is_buyer_maker",
]

# 경로 관련 변수 선언
project_dir = args.project_dir
data_dir = args.data_dir
output_dir = args.output_dir
log_dir = args.log_dir

# 로거 설정
logger = logging.getLogger("preprocess_bi_log")
logger.setLevel(logging.DEBUG)

# 콘솔 핸들러 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 포맷 설정
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# 핸들러 추가
logger.addHandler(console_handler)


# 디렉토리 생성 함수
def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"디렉토리 생성됨: {directory_path}")
    else:
        logger.info(f"디렉토리가 이미 존재함: {directory_path}")


# 파일 처리 함수
def process_file(file_path):
    """
    데이터 전처리 함수:
    - arg.window_size 단위로 거래 데이터 그룹화
    - 거래량×가격 기준으로 상위 20%는 기관, 나머지는 개인 투자자로 분류
    - 거래 체결자를 기준으로 주문자(is_buyer_maker)가 false면 매수(long), true면 매도(short)로 분류
    - 각 시간 윈도우 내 투자자 유형별 매수/매도 분포 계산 (log1p 적용)
    """
    try:
        # 파일명에서 연도와 월 추출
        filename = os.path.basename(file_path)
        match = re.search(r"BTCUSDT-aggTrades-(\d{4})-(\d{2})\.csv", filename)

        if not match:
            logger.error(f"파일명 형식이 맞지 않습니다: {filename}")
            return None

        year, month = match.groups()

        logger.info(f"[PROCESS] {file_path} 처리 중...")

        # 파일 읽기
        df = pd.read_csv(
            file_path,
            index_col=0,
            usecols=usecols,
        ).reset_index()

        # transact_time을 datetime으로 변환
        df.loc[:, "transact_time"] = pd.to_datetime(
            df.loc[:, "transact_time"], unit="ms"
        )

        # df를 윈도우 크기 단위로 그룹화 - 윈도우 내에 발생한 것 중에서 거래량 * 가격으로 개인/기관으로 분류 후, long/short 행동 분포를 list로 분류
        df.loc[:, "quantity_price"] = df.loc[:, "quantity"] * df.loc[:, "price"]

        # 윈도우 크기를 밀리초에서 시간 단위로 변환
        window_size_timedelta = pd.Timedelta(milliseconds=window_size)

        # 결과 데이터 프레임 생성
        # - index: 윈도우 크기에 맞게 조정된 시간 간격
        # - columns: retail_long, retail_short, institutional_long, institutional_short
        result_start = df.iloc[0]["transact_time"].floor(f"{window_size}ms")
        result_end = df.iloc[-1]["transact_time"].floor(f"{window_size}ms")

        result_df = pd.DataFrame(
            index=pd.date_range(
                start=result_start,
                end=result_end,
                freq=f"{window_size}ms",
            ),
            columns=[
                "retail_long",
                "retail_short",
                "institutional_long",
                "institutional_short",
            ],
        )

        # df의 윈도우 단위로 for loop 돌리기
        for pivot_time in result_df.index:
            df_subset = df.loc[
                (df.loc[:, "transact_time"] < pivot_time)
                & (df.loc[:, "transact_time"] >= pivot_time - window_size_timedelta)
            ]

            if df_subset.empty:
                continue

            # df_subset을 is_buyer_maker 기준으로 long, short 값으로 분류
            # long: is_buyer_maker == False
            # short: is_buyer_maker == True
            # long_df에 로그 함수 적용
            long_df = df_subset[df_subset["is_buyer_maker"] == False].loc[
                :, "quantity_price"
            ]
            long_df = np.log1p(long_df)  # 로그 함수 적용 (1을 더해 0 값 처리)

            # short_df는 로그 함수 적용 후 -1 곱하기
            short_df = df_subset[df_subset["is_buyer_maker"] == True].loc[
                :, "quantity_price"
            ]
            short_df = np.log1p(short_df)  # 로그 함수 적용 (1을 더해 0 값 처리)
            short_df = (-1) * short_df  # 로그 적용 후 -1 곱하기

            # long_df, short_df를 합친 후, 거래량 * 가격 기준으로 정렬
            quantity_price_df = pd.concat([long_df, short_df]).sort_values(
                ascending=False
            )

            # -1 ~ 1 사이의 값으로 정규화
            quantity_price_df = quantity_price_df / quantity_price_df.abs().max()

            # long에 해당하는 +값 중에서 상위 20%는 기관, 나머지 80%는 개인으로 분류
            qp_long_df = quantity_price_df.loc[quantity_price_df >= 0]
            long_type_threshold = int(len(qp_long_df) * type_threshold)
            institutional_long_df = qp_long_df.iloc[:long_type_threshold]
            retail_long_df = qp_long_df.iloc[long_type_threshold:]

            # short에 해당하는 -값 중에서 하위 20%는 기관, 나머지 80%는 개인으로 분류
            qp_short_df = quantity_price_df.loc[quantity_price_df < 0]
            short_type_shreshold = int(len(qp_short_df) * type_threshold)
            institutional_short_df = qp_short_df.iloc[short_type_shreshold:]
            retail_short_df = qp_short_df.iloc[:short_type_shreshold]

            # quantile_bins의 각 구간에 해당하는 값들을 카운트
            # 카운트 결과를 result_df에 저장
            retail_long_count = np.zeros(quantile)
            retail_short_count = np.zeros(quantile)
            institutional_long_count = np.zeros(quantile)
            institutional_short_count = np.zeros(quantile)
            for i in range(len(quantile_bins) - 1):
                if quantile_bins[i] < 0:
                    retail_short_count[i] += len(
                        retail_short_df.loc[
                            (quantile_bins[i] <= retail_short_df)
                            & (quantile_bins[i + 1] >= retail_short_df)
                        ]
                    )
                    institutional_short_count[i] += len(
                        institutional_short_df.loc[
                            (quantile_bins[i] <= institutional_short_df)
                            & (quantile_bins[i + 1] >= institutional_short_df)
                        ]
                    )
                else:
                    retail_long_count[i - quantile] += len(
                        retail_long_df.loc[
                            (quantile_bins[i] <= retail_long_df)
                            & (quantile_bins[i + 1] >= retail_long_df)
                        ]
                    )

                    institutional_long_count[i - quantile] += len(
                        institutional_long_df.loc[
                            (quantile_bins[i] <= institutional_long_df)
                            & (quantile_bins[i + 1] >= institutional_long_df)
                        ]
                    )

            # 개인 / 기관별 최대값으로 정규화
            retail_sum = retail_long_count.sum() + retail_short_count.sum()
            institutional_sum = (
                institutional_long_count.sum() + institutional_short_count.sum()
            )

            # 배열을 DataFrame에 저장 후 나중에 한 번에 JSON으로 변환
            result_df.at[pivot_time, "retail_long"] = json.dumps(
                (retail_long_count / retail_sum).tolist(), ensure_ascii=False
            )
            result_df.at[pivot_time, "retail_short"] = json.dumps(
                (retail_short_count / retail_sum).tolist(), ensure_ascii=False
            )
            result_df.at[pivot_time, "institutional_long"] = json.dumps(
                (institutional_long_count / institutional_sum).tolist(),
                ensure_ascii=False,
            )
            result_df.at[pivot_time, "institutional_short"] = json.dumps(
                (institutional_short_count / institutional_sum).tolist(),
                ensure_ascii=False,
            )

        # result_df의 row 중에서 nan이 있다면, 해당 row 삭제
        result_df = result_df.dropna()

        # 결과 저장 경로
        output_filename = f"BTCUSDT-pd-{year}-{month}.csv"
        output_path = os.path.join(output_dir, output_filename)

        # 결과를 CSV 파일로 저장 (JSON 형식의 문자열이 포함됨)
        result_df.to_csv(output_path, index=True, quoting=csv.QUOTE_NONNUMERIC)
        logger.info(f"[RESULT] 처리 완료: {output_path}")

        return output_path

    except Exception as e:
        import traceback

        error_info = traceback.extract_tb(e.__traceback__)
        logger.error(
            f"\n\n[ERROR] 파일 처리 중 오류 발생: {file_path}\n오류: {str(e)}\n코드: {error_info}"
        )
        return None


# ProcessPoolExecutor 종료 시 리소스 정리를 위한 함수
def cleanup_pool():
    for p in multiprocessing.active_children():
        p.terminate()
    logger.info("모든 프로세스 정리 완료")


def main():
    # 출력 디렉토리 생성
    create_directory(output_dir)

    # 파일 경로 확인
    file_pattern = os.path.join(data_dir, "BTCUSDT-aggTrades-*.csv")
    files_to_process = glob.glob(file_pattern)

    if not files_to_process:
        logger.error(f"[ERROR] 처리할 파일을 찾을 수 없습니다: {file_pattern}")
        return

    logger.info(f"[PROCESS] {len(files_to_process)}개 파일을 처리합니다...")

    # CPU 코어 수 확인
    num_cores = multiprocessing.cpu_count()
    logger.info(f"[PROCESS] CPU 코어 수: {num_cores}")

    # 병렬 처리 실행 (tqdm으로 진행 상황 표시)
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(
            tqdm(
                executor.map(process_file, files_to_process),
                total=len(files_to_process),
                desc="파일 처리 진행률",
            )
        )

    # 성공적으로 처리된 파일 수 확인
    successful_results = [r for r in results if r is not None]
    logger.info(
        f"[RESULT] 처리 완료: {len(successful_results)}/{len(files_to_process)} 파일"
    )


if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"[PROCESS] 처리 시작: {start_time}")
    main()
    end_time = datetime.now()
    logger.info(f"[RESULT] 총 처리 시간: {end_time - start_time}")
