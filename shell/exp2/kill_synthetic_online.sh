'''
chmod +x shell/exp2/kill_synthetic_online.sh
./shell/exp2/kill_synthetic_online.sh
'''

# 가장 최근 실험의 모든 프로세스 종료
LATEST_DIR=$(find ./results/exp2/ -maxdepth 1 -type d -name "20*" | sort | tail -1)

if [ -z "$LATEST_DIR" ]; then
    echo "실행 중인 실험을 찾을 수 없습니다."
    exit 1
fi

PID_FILE="${LATEST_DIR}/experiment_pids.txt"

if [ ! -f "$PID_FILE" ]; then
    echo "PID 파일을 찾을 수 없습니다: $PID_FILE"
    exit 1
fi

echo "실험 프로세스들을 종료합니다..."
echo "대상 디렉토리: $LATEST_DIR"

# 먼저 정상 종료 시도
echo "정상 종료 시도 중..."
cat "$PID_FILE" | xargs -I {} sh -c 'kill {} 2>/dev/null && echo "PID {} 종료 신호 전송" || echo "PID {} 이미 종료됨"'

sleep 3

# 아직 실행 중인 프로세스 강제 종료
echo "남은 프로세스 강제 종료 중..."
cat "$PID_FILE" | xargs -I {} sh -c 'kill -0 {} 2>/dev/null && kill -9 {} && echo "PID {} 강제 종료" || true'

echo "모든 실험 프로세스 종료 완료"
