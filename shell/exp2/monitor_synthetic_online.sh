'''
chmod +x shell/exp2/monitor_synthetic_online.sh
./shell/exp2/monitor_synthetic_online.sh
'''

# 가장 최근 실험 디렉토리 찾기
LATEST_DIR=$(find ./results/exp2/ -maxdepth 1 -type d -name "20*" | sort | tail -1)

if [ -z "$LATEST_DIR" ]; then
    echo "실행 중인 실험을 찾을 수 없습니다."
    exit 1
fi

echo "모니터링 대상: $LATEST_DIR"

PID_FILE="${LATEST_DIR}/experiment_pids.txt"
LOG_DIR="${LATEST_DIR}/logs"

if [ ! -f "$PID_FILE" ]; then
    echo "PID 파일을 찾을 수 없습니다: $PID_FILE"
    exit 1
fi

# 실시간 상태 확인
echo "=== 실험 상태 ==="
while IFS= read -r pid; do
    if kill -0 "$pid" 2>/dev/null; then
        echo "PID $pid: 실행중"
    else
        echo "PID $pid: 완료/종료됨"
    fi
done < "$PID_FILE"

echo ""
echo "=== 로그 파일들 ==="
ls -la "$LOG_DIR"/*.log 2>/dev/null || echo "로그 파일 없음"

echo ""
echo "=== 결과 파일들 ==="
ls -la "$LATEST_DIR"/*.json 2>/dev/null || echo "결과 파일 없음"

echo ""
echo "실시간 로그 확인: tail -f ${LOG_DIR}/*.log"
echo "모든 프로세스 종료: cat ${PID_FILE} | xargs kill"
