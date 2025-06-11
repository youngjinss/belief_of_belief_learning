# chmod +x shell/exp2/execute_synthetic_online.sh
# ./shell/exp2/execute_synthetic_online.sh

# 실험 설정
TIMESTAMP=$(date +"%Y%m%d_%H%M")
RESULT_DIR="./results/exp2/${TIMESTAMP}"
LOG_DIR="${RESULT_DIR}/logs"
SCRIPT_PATH="./script/exp2/simulate_synthetic_online.py"

# 실험 파라미터
EXPLORATION_NOISE_STD=1  # 탐험 노이즈 표준편차
MAX_ITERATIONS=1000        # 최대 반복 횟수
WINDOW_SIZE=50            # 슬라이딩 윈도우 크기
CONVERGENCE_THRESHOLD=0.001 # 수렴 임계값

# 레벨 조합 정의 (buyer_level,seller_level)
LEVEL_COMBINATIONS=(
    "0,1"
    "1,1" 
    "2,2"
    "2,3"
    "2,4"
    "3,3"
    "3,4"
)

# 디렉토리 생성
mkdir -p "${LOG_DIR}"

# 실험 정보 저장
EXPERIMENT_INFO_FILE="${RESULT_DIR}/experiment_info.txt"
PID_FILE="${RESULT_DIR}/experiment_pids.txt"

echo "실험 시작: ${TIMESTAMP}"
echo "결과 디렉토리: ${RESULT_DIR}"

# PID 배열
PIDS=()

# 각 조합에 대해 실험 실행
for combo in "${LEVEL_COMBINATIONS[@]}"; do
    # 레벨 분리
    buyer_level=$(echo $combo | cut -d',' -f1)
    seller_level=$(echo $combo | cut -d',' -f2)
    
    # 로그 파일명
    log_file="${LOG_DIR}/L${buyer_level}B_vs_L${seller_level}S.log"
    
    echo "시작: 구매자 L${buyer_level} vs 판매자 L${seller_level}"
    
    # 백그라운드에서 실험 실행
    nohup python "${SCRIPT_PATH}" \
        --level_combinations "${combo}" \
        --experiment_time "${TIMESTAMP}" \
        --result_dir "${RESULT_DIR}/" \
        --exploration_noise_std "${EXPLORATION_NOISE_STD}" \
        --max_iterations "${MAX_ITERATIONS}" \
        --window_size "${WINDOW_SIZE}" \
        --convergence_threshold "${CONVERGENCE_THRESHOLD}" \
        > "${log_file}" 2>&1 &
    
    # PID 저장
    pid=$!
    PIDS+=($pid)
    echo "  PID: ${pid}, 로그: ${log_file}"
    
    # PID 파일에 저장
    echo $pid >> "${PID_FILE}"
done

echo ""
echo "모든 실험이 백그라운드에서 실행 중입니다."
echo "실행 중인 프로세스 수: ${#PIDS[@]}"
echo "PID들: ${PIDS[*]}"

# 실험 정보 파일 생성
cat > "${EXPERIMENT_INFO_FILE}" << EOF
실험 시작 시각: $(date)
타임스탬프: ${TIMESTAMP}
결과 디렉토리: ${RESULT_DIR}
로그 디렉토리: ${LOG_DIR}
스크립트 경로: ${SCRIPT_PATH}

실험 파라미터:
  탐험 노이즈 표준편차: ${EXPLORATION_NOISE_STD}
  최대 반복 횟수: ${MAX_ITERATIONS}
  윈도우 크기: ${WINDOW_SIZE}
  수렴 임계값: ${CONVERGENCE_THRESHOLD}

레벨 조합:
EOF

for i in "${!LEVEL_COMBINATIONS[@]}"; do
    combo="${LEVEL_COMBINATIONS[$i]}"
    pid="${PIDS[$i]}"
    buyer_level=$(echo $combo | cut -d',' -f1)
    seller_level=$(echo $combo | cut -d',' -f2)
    echo "  L${buyer_level}B vs L${seller_level}S: PID ${pid}" >> "${EXPERIMENT_INFO_FILE}"
done

echo "" >> "${EXPERIMENT_INFO_FILE}"
echo "PID 목록:" >> "${EXPERIMENT_INFO_FILE}"
for pid in "${PIDS[@]}"; do
    echo "  ${pid}" >> "${EXPERIMENT_INFO_FILE}"
done

# 모니터링 안내
echo ""
echo "=== 실험 진행 상황 모니터링 ==="
echo "로그 파일들:"
for combo in "${LEVEL_COMBINATIONS[@]}"; do
    buyer_level=$(echo $combo | cut -d',' -f1)
    seller_level=$(echo $combo | cut -d',' -f2)
    echo "  ${LOG_DIR}/L${buyer_level}B_vs_L${seller_level}S.log"
done

echo ""
echo "실시간 로그 확인 명령어 예시:"
buyer_level=$(echo ${LEVEL_COMBINATIONS[0]} | cut -d',' -f1)
seller_level=$(echo ${LEVEL_COMBINATIONS[0]} | cut -d',' -f2)
echo "  tail -f ${LOG_DIR}/L${buyer_level}B_vs_L${seller_level}S.log"

echo ""
echo "모든 로그 동시 확인:"
echo "  tail -f ${LOG_DIR}/*.log"

# 종료 명령어 안내
echo ""
echo "=== 실험 중단 명령어 ==="
echo "1. 개별 프로세스 종료:"
for i in "${!LEVEL_COMBINATIONS[@]}"; do
    combo="${LEVEL_COMBINATIONS[$i]}"
    pid="${PIDS[$i]}"
    buyer_level=$(echo $combo | cut -d',' -f1)
    seller_level=$(echo $combo | cut -d',' -f2)
    echo "   kill ${pid}  # ${buyer_level},${seller_level}"
done

echo ""
echo "2. 모든 실험 프로세스 한번에 종료:"
echo "   kill ${PIDS[*]}"

echo ""
echo "3. 저장된 PID 파일로 종료:"
echo "   cat ${PID_FILE} | xargs kill"

echo ""
echo "4. 강제 종료 (필요시):"
echo "   kill -9 ${PIDS[*]}"

echo ""
echo "실험 정보가 저장되었습니다: ${EXPERIMENT_INFO_FILE}"

# 모니터링 명령어 안내
echo ""
echo "10초마다 완료 상태를 확인하려면 다음 명령어를 실행하세요:"
echo "  watch -n 10 'echo \"=== \$(date) ===\"; cat ${PID_FILE} | xargs -I {} sh -c \"kill -0 {} 2>/dev/null && echo PID {} 실행중 || echo PID {} 완료\"'"

echo ""
echo "모든 실험 완료까지 대기하려면 다음 명령어를 실행하세요:"
echo "  wait ${PIDS[*]} && echo \"모든 실험 완료!\" || echo \"일부 실험 실패\""
