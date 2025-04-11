#!/bin/bash

# 스크립트 실행 시간 기록
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SCRIPT_NAME="[v3] preprocess_bi_log_samebins.py"

# 기본값 설정
WINDOW_SIZE="3600000"
TYPE_THRESHOLD="0.25"
QUANTILE="20"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${PROJECT_DIR}/data/binance/futures/um/monthly/aggTrades/BTCUSDT"
OUTPUT_DIR="${PROJECT_DIR}/data/binance/futures/um/monthly/position_distribution/${WINDOW_SIZE}"
LOG_DIR="${PROJECT_DIR}/log/${START_TIME}"
LOG_FILE="${LOG_DIR}/result.log"

# 도움말 함수
show_help() {
    echo "사용법: $0 [옵션]"
    echo "옵션:"
    echo "  -h, --help                  도움말 표시"
    echo "  -w, --window-size VALUE     윈도우 크기(밀리초), 기본값: 3600000"
    echo "  -t, --type-threshold VALUE  기관/개인 투자자 분류 임계값, 기본값: 0.25"
    echo "  -q, --quantile VALUE        분위수 구간 수, 기본값: 20"
    echo "  -p, --project-dir PATH      프로젝트 디렉토리 경로"
    echo "  -d, --data-dir PATH         입력 데이터 디렉토리 경로"
    echo "  -o, --output-dir PATH       출력 데이터 디렉토리 경로"
}

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -w|--window-size)
            WINDOW_SIZE="$2"
            shift 2
            ;;
        -t|--type-threshold)
            TYPE_THRESHOLD="$2"
            shift 2
            ;;
        -q|--quantile)
            QUANTILE="$2"
            shift 2
            ;;
        -p|--project-dir)
            PROJECT_DIR="$2"
            shift 2
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            show_help
            exit 1
            ;;
    esac
done

# 로그 디렉토리 생성
mkdir -p "$LOG_DIR"

# 시작 메시지 출력
echo "백그라운드에서 스크립트 실행을 시작합니다."
echo "시작 시간: $START_TIME"
echo "파이썬 스크립트: $SCRIPT_NAME"
echo "사용 인자: --window_size $WINDOW_SIZE --type_threshold $TYPE_THRESHOLD --quantile $QUANTILE --project_dir $PROJECT_DIR --data_dir $DATA_DIR --output_dir $OUTPUT_DIR"

# 파이썬 스크립트 백그라운드에서 실행
nohup python "script/${SCRIPT_NAME}" \
    --window_size "$WINDOW_SIZE" \
    --type_threshold "$TYPE_THRESHOLD" \
    --quantile "$QUANTILE" \
    --project_dir "$PROJECT_DIR" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" > "$LOG_FILE" 2>&1 &

# 프로세스 ID 저장
PID=$!
echo "프로세스 ID: $PID"
echo "$PID" > "${LOG_DIR}/last_pid.txt"

echo "백그라운드에서 실행 중입니다. 다음 명령어로 로그를 확인할 수 있습니다:"
echo "tail -f $LOG_FILE"
echo "다음 명령어로 프로세스 상태를 확인할 수 있습니다:"
echo "ps -p $PID"
