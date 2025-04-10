import os
import pandas as pd
import multiprocessing
from datetime import datetime
import glob
from pathlib import Path
import numpy as np


def ensure_directory(directory_path):
    """지정된 디렉토리가 존재하는지 확인하고, 존재하지 않으면 생성합니다."""
    os.makedirs(directory_path, exist_ok=True)
    return directory_path


def process_file(file_path):
    """
    단일 파일을 처리하고 결과 DataFrame을 반환합니다.
    이 함수는 원본 노트북의 데이터 처리 로직을 포함해야 합니다.
    """
    try:
        print(f"파일 처리 중: {file_path}")

        # 여기서 노트북에 있던 데이터 처리 로직을 구현해야 합니다.
        # 예시로 간단한 로직을 보여드립니다. 실제 로직으로 대체하세요.
        df = pd.read_csv(file_path)

        # 데이터 처리 로직...
        result_df = df.copy()  # 실제 처리 로직으로 대체하세요

        # 파일 이름에서 년도와 월 추출 (파일명 형식에 따라 수정 필요)
        file_name = os.path.basename(file_path)
        date_parts = extract_date_from_filename(file_name)

        if date_parts:
            year, month = date_parts
            output_path = f"/home/youngjins/project/belief_trading/data/binance/futures/um/monthly/position_distribution/BTCUSDT-pd-{year}-{month}.csv"

            # 결과 저장
            result_df.to_csv(output_path, index=False)
            print(f"결과 저장 완료: {output_path}")

            return True, output_path
        else:
            print(f"파일명에서 날짜 정보를 추출할 수 없습니다: {file_name}")
            return False, None

    except Exception as e:
        print(f"파일 처리 중 오류 발생: {file_path}, 에러: {str(e)}")
        return False, None


def extract_date_from_filename(filename):
    """
    파일명에서 연도와 월 정보를 추출합니다.
    파일명 형식에 따라 이 함수를 수정해야 할 수 있습니다.
    """
    try:
        # 예시: 파일명 형식이 'something_YYYY_MM.csv' 또는 'something_YYYYMM.csv'라고 가정
        # 실제 파일명 형식에 맞게 수정하세요
        parts = filename.split("_")
        for part in parts:
            if part.isdigit() and len(part) == 6:  # YYYYMM 형식
                year = part[:4]
                month = part[4:6]
                return year, month
            elif (
                part.isdigit() and len(part) == 4
            ):  # YYYY 형식 (월 정보가 다른 부분에 있는 경우)
                year = part
                # 다른 부분에서 월 정보 찾기...
                for other_part in parts:
                    if other_part.isdigit() and len(other_part) == 2:
                        month = other_part
                        return year, month

        # 다른 형식의 파일명 처리...

        return None
    except Exception:
        return None


def main():
    # 출력 디렉토리 생성
    output_dir = "/home/youngjins/project/belief_trading/data/binance/futures/um/monthly/position_distribution/"
    ensure_directory(output_dir)

    # 처리할 파일 목록 (노트북에서 사용한 데이터 파일 경로로 수정 필요)
    # 예: /path/to/data/*.csv 또는 특정 디렉토리 내 모든 CSV 파일
    input_files = glob.glob("/path/to/your/data/files/*.csv")  # 실제 경로로 수정하세요

    if not input_files:
        print("처리할 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return

    print(f"총 {len(input_files)}개 파일을 처리합니다.")

    # CPU 코어 수 확인
    num_cores = multiprocessing.cpu_count()
    print(f"사용 가능한 CPU 코어 수: {num_cores}")

    # 병렬 처리 풀 생성 및 실행
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(process_file, input_files)

    # 결과 요약
    successful = sum(1 for result, _ in results if result)
    print(f"처리 완료: 총 {len(input_files)}개 중 {successful}개 성공")


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"작업 시작 시간: {start_time}")

    main()

    end_time = datetime.now()
    print(f"작업 종료 시간: {end_time}")
    print(f"총 소요 시간: {end_time - start_time}")
