#!/bin/bash

# 스크립트 실행 시간 기록
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SCRIPT_NAME="train_hbt.py"

# 기본값 설정
MODEL_TYPE="proposed"
CONFIG_PATH="config/train.yaml"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PROJECT_DIR}/log/${START_TIME}"
LOG_FILE="${LOG_DIR}/result.log"

# 도움말 함수
show_help() {
    echo "사용법: $0 [옵션]"
    echo "옵션:"
    echo "  -h, --help                 도움말 표시"
    echo "  -m, --model-type VALUE     학습할 모델 유형 (proposed, benchmark, both), 기본값: proposed"
    echo "  -c, --config-path PATH     설정 파일 경로, 기본값: config/train.yaml"
    echo "  -p, --project-dir PATH     프로젝트 디렉토리 경로"
    echo "  -l, --log-dir PATH         로그 디렉토리 경로"
    echo "  -t, --tensorboard-port PORT 텐서보드 포트 번호, 기본값: 6006"
    echo "  -d, --tensorboard-logdir DIR 텐서보드 로그 디렉토리, 기본값: runs"
}

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -m|--model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        -c|--config-path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -p|--project-dir)
            PROJECT_DIR="$2"
            shift 2
            ;;
        -l|--log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -t|--tensorboard-port)
            TENSORBOARD_PORT="$2"
            shift 2
            ;;
        -d|--tensorboard-logdir)
            TENSORBOARD_LOGDIR="$2"
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
echo "사용 인자: --model_type $MODEL_TYPE --config_path $CONFIG_PATH --project_dir $PROJECT_DIR --log_dir $LOG_DIR"

# 파이썬 스크립트 백그라운드에서 실행
nohup python "script/${SCRIPT_NAME}" \
    --model_type "$MODEL_TYPE" \
    --config_path "$CONFIG_PATH" \
    --project_dir "$PROJECT_DIR" \
    --telegrambot true \
    --log_dir "$LOG_DIR" > "$LOG_FILE" 2>&1 &

# 프로세스 ID 저장
PID=$!
echo "프로세스 ID: $PID"
echo "$PID" > "${LOG_DIR}/last_pid.txt"

echo "백그라운드에서 실행 중입니다. 다음 명령어로 로그를 확인할 수 있습니다:"
echo "tail -f $LOG_FILE"
echo "다음 명령어로 프로세스 상태를 확인할 수 있습니다:"
echo "ps -p $PID"

echo "다음 명령어로 프로세스를 종료할 수 있습니다:"
echo "pgrep -f \"${SCRIPT_NAME}\" | xargs kill -9"
