'''
chmod +x shell/exp2/execute_synthetic_online_multi_beta_pref.sh
./shell/exp2/execute_synthetic_online_multi_beta_pref.sh
'''

# 실험 설정
TIMESTAMP=$(date +"%Y%m%d_%H%M")
RESULT_DIR="./results/exp2/${TIMESTAMP}"
LOG_DIR="${RESULT_DIR}/logs"
SCRIPT_PATH="./script/exp2/simulate_synthetic_online.py"

# 실험 파라미터
EXPLORATION_BETAS=(0.1 0.3 0.5 0.7 1.3 1.5 3.0)  # 여러 탐험 노이즈 표준편차 값
DEFAULT_BETA=1.0          # 원래 노이즈 표준편차
N_PREFERENCE_POINTS=51    # 선호도 포인트 수
N_PRICE_POINTS=31         # 가격 포인트 수
MAX_ITERATIONS=1000       # 최대 반복 횟수
WINDOW_SIZE=100           # 슬라이딩 윈도우 크기
CONVERGENCE_THRESHOLD=0.05 # 수렴 임계값

# 레벨 조합 정의 (buyer_level,seller_level)
LEVEL_COMBINATIONS=(
    "0,1"
    "1,1" 
    "1,2"
    "2,3"
    "3,3"
    "3,4"
)
# LEVEL_COMBINATIONS=("0,1")  # fixme: for debug

# 실제 선호도 설정들 (preference_1,preference_2 쌍들)
TRUE_PREFERENCE_PAIRS=(
    "7,3"
    "4,7"
    "2,8"
)

# 디렉토리 생성
mkdir -p "${LOG_DIR}"

# 실험 정보 저장
EXPERIMENT_INFO_FILE="${RESULT_DIR}/experiment_info.txt"
PID_FILE="${RESULT_DIR}/experiment_pids.txt"

echo "실험 시작: ${TIMESTAMP}"
echo "결과 디렉토리: ${RESULT_DIR}"
echo "탐험 베타 값들: ${EXPLORATION_BETAS[*]}"
echo "선호도 쌍들: ${TRUE_PREFERENCE_PAIRS[*]}"
echo "레벨 조합들: ${LEVEL_COMBINATIONS[*]}"

# 총 실험 수 계산
TOTAL_EXPERIMENTS=$((${#EXPLORATION_BETAS[@]} * ${#TRUE_PREFERENCE_PAIRS[@]} * ${#LEVEL_COMBINATIONS[@]}))
echo "총 실험 수: ${TOTAL_EXPERIMENTS}"

# PID 배열
PIDS=()

# 각 EXPLORATION_BETA, TRUE_PREFERENCE_PAIR, LEVEL_COMBINATION 조합에 대해 실험 실행
for exploration_beta in "${EXPLORATION_BETAS[@]}"; do
    for pref_pair in "${TRUE_PREFERENCE_PAIRS[@]}"; do
        # 선호도 분리
        true_preference_1=$(echo $pref_pair | cut -d',' -f1)
        true_preference_2=$(echo $pref_pair | cut -d',' -f2)
        # 파일명용 선호도 문자열 (소수점 제거)
        pref_str="${true_preference_1//.}${true_preference_2//.}"
        
        for combo in "${LEVEL_COMBINATIONS[@]}"; do
            # 레벨 분리
            buyer_level=$(echo $combo | cut -d',' -f1)
            seller_level=$(echo $combo | cut -d',' -f2)
            
            # 로그 파일명 (beta, prefer, level combos 순서)
            log_file="${LOG_DIR}/beta${exploration_beta}_pref${pref_str}_L${buyer_level}B_vs_L${seller_level}S.log"
            
            echo "시작: 탐험베타=${exploration_beta}, 선호도=(${true_preference_1},${true_preference_2}), 구매자 L${buyer_level} vs 판매자 L${seller_level}"
            
            # 백그라운드에서 실험 실행
            nohup python "${SCRIPT_PATH}" \
                --true_preference "${true_preference_1}" "${true_preference_2}" \
                --level_combinations "${combo}" \
                --experiment_time "${TIMESTAMP}" \
                --result_dir "${RESULT_DIR}/" \
                --exploration_beta "${exploration_beta}" \
                --default_beta "${DEFAULT_BETA}" \
                --n_preference_points "${N_PREFERENCE_POINTS}" \
                --n_price_points "${N_PRICE_POINTS}" \
                --max_iterations "${MAX_ITERATIONS}" \
                --window_size "${WINDOW_SIZE}" \
                --convergence_threshold "${CONVERGENCE_THRESHOLD}" \
                > "${log_file}" 2>&1 &
            
            # PID 저장
            pid=$!
            PIDS+=($pid)
            echo "  PID: ${pid}, 로그: ${log_file}"
            
            # PID 파일에 저장 (beta, prefer, level combos 정보 포함)
            echo "${pid} # beta${exploration_beta}_pref${pref_str}_L${buyer_level}B_vs_L${seller_level}S" >> "${PID_FILE}"
            
            # 시스템 부하 방지를 위한 짧은 대기
            sleep 0.1
        done
    done
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
  탐험 노이즈 표준편차들: ${EXPLORATION_BETAS[*]}
  기본 노이즈 표준편차: ${DEFAULT_BETA}
  최대 반복 횟수: ${MAX_ITERATIONS}
  윈도우 크기: ${WINDOW_SIZE}
  수렴 임계값: ${CONVERGENCE_THRESHOLD}
  실제 선호도 쌍들: ${TRUE_PREFERENCE_PAIRS[*]}
  
총 실험 수: ${TOTAL_EXPERIMENTS}

레벨 조합, 선호도, 탐험 베타 조합:
EOF

# 각 조합 정보를 파일에 저장
experiment_idx=0
for exploration_beta in "${EXPLORATION_BETAS[@]}"; do
    for pref_pair in "${TRUE_PREFERENCE_PAIRS[@]}"; do
        true_preference_1=$(echo $pref_pair | cut -d',' -f1)
        true_preference_2=$(echo $pref_pair | cut -d',' -f2)
        pref_str="${true_preference_1//.}${true_preference_2//.}"
        
        for combo in "${LEVEL_COMBINATIONS[@]}"; do
            buyer_level=$(echo $combo | cut -d',' -f1)
            seller_level=$(echo $combo | cut -d',' -f2)
            pid="${PIDS[$experiment_idx]}"
            echo "  beta${exploration_beta}_pref${pref_str}_L${buyer_level}B_vs_L${seller_level}S: PID ${pid}" >> "${EXPERIMENT_INFO_FILE}"
            experiment_idx=$((experiment_idx + 1))
        done
    done
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
for exploration_beta in "${EXPLORATION_BETAS[@]}"; do
    for pref_pair in "${TRUE_PREFERENCE_PAIRS[@]}"; do
        true_preference_1=$(echo $pref_pair | cut -d',' -f1)
        true_preference_2=$(echo $pref_pair | cut -d',' -f2)
        pref_str="${true_preference_1//.}${true_preference_2//.}"
        
        for combo in "${LEVEL_COMBINATIONS[@]}"; do
            buyer_level=$(echo $combo | cut -d',' -f1)
            seller_level=$(echo $combo | cut -d',' -f2)
            echo "  ${LOG_DIR}/beta${exploration_beta}_pref${pref_str}_L${buyer_level}B_vs_L${seller_level}S.log"
        done
    done
done

echo ""
echo "실시간 로그 확인 명령어 예시:"
first_beta="${EXPLORATION_BETAS[0]}"
first_pref_pair="${TRUE_PREFERENCE_PAIRS[0]}"
first_pref_1=$(echo $first_pref_pair | cut -d',' -f1)
first_pref_2=$(echo $first_pref_pair | cut -d',' -f2)
first_pref_str="${first_pref_1//.}${first_pref_2//.}"
buyer_level=$(echo ${LEVEL_COMBINATIONS[0]} | cut -d',' -f1)
seller_level=$(echo ${LEVEL_COMBINATIONS[0]} | cut -d',' -f2)
echo "  tail -f ${LOG_DIR}/beta${first_beta}_pref${first_pref_str}_L${buyer_level}B_vs_L${seller_level}S.log"

echo ""
echo "모든 로그 동시 확인:"
echo "  tail -f ${LOG_DIR}/*.log"

echo ""
echo "특정 탐험 베타의 모든 로그 확인 (예: beta${first_beta}):"
echo "  tail -f ${LOG_DIR}/beta${first_beta}_*.log"

echo ""
echo "특정 선호도의 모든 로그 확인 (예: pref${first_pref_str}):"
echo "  tail -f ${LOG_DIR}/beta*_pref${first_pref_str}_*.log"

# 종료 명령어 안내
echo ""
echo "=== 실험 중단 명령어 ==="
echo "1. 개별 프로세스 종료:"
experiment_idx=0
for exploration_beta in "${EXPLORATION_BETAS[@]}"; do
    for pref_pair in "${TRUE_PREFERENCE_PAIRS[@]}"; do
        true_preference_1=$(echo $pref_pair | cut -d',' -f1)
        true_preference_2=$(echo $pref_pair | cut -d',' -f2)
        pref_str="${true_preference_1//.}${true_preference_2//.}"
        
        for combo in "${LEVEL_COMBINATIONS[@]}"; do
            buyer_level=$(echo $combo | cut -d',' -f1)
            seller_level=$(echo $combo | cut -d',' -f2)
            pid="${PIDS[$experiment_idx]}"
            echo "   kill ${pid}  # beta${exploration_beta}_pref${pref_str}_${buyer_level},${seller_level}"
            experiment_idx=$((experiment_idx + 1))
        done
    done
done

echo ""
echo "2. 모든 실험 프로세스 한번에 종료:"
echo "   kill ${PIDS[*]}"

echo ""
echo "3. 저장된 PID 파일로 종료:"
echo "   cat ${PID_FILE} | cut -d' ' -f1 | xargs kill"

echo ""
echo "4. 강제 종료 (필요시):"
echo "   kill -9 ${PIDS[*]}"

echo ""
echo "실험 정보가 저장되었습니다: ${EXPERIMENT_INFO_FILE}"

# 모니터링 명령어 안내
echo ""
echo "10초마다 완료 상태를 확인하려면 다음 명령어를 실행하세요:"
echo "  watch -n 10 'echo \"=== \$(date) ===\"; cat ${PID_FILE} | cut -d\" \" -f1 | xargs -I {} sh -c \"kill -0 {} 2>/dev/null && echo PID {} 실행중 || echo PID {} 완료\"'"

echo ""
echo "모든 실험 완료까지 대기하려면 다음 명령어를 실행하세요:"
echo "  wait ${PIDS[*]} && echo \"모든 실험 완료!\" || echo \"일부 실험 실패\""

# 결과 정리를 위한 명령어 안내
echo ""
echo "=== 결과 정리 명령어 ==="
echo "각 탐험 베타별로 결과 파일 확인:"
for exploration_beta in "${EXPLORATION_BETAS[@]}"; do
    echo "  ls ${RESULT_DIR}/*beta${exploration_beta}_*.json"
done

echo ""
echo "각 선호도별로 결과 파일 확인:"
for pref_pair in "${TRUE_PREFERENCE_PAIRS[@]}"; do
    true_preference_1=$(echo $pref_pair | cut -d',' -f1)
    true_preference_2=$(echo $pref_pair | cut -d',' -f2)
    pref_str="${true_preference_1//.}${true_preference_2//.}"
    echo "  ls ${RESULT_DIR}/*pref${pref_str}_*.json"
done

echo ""
echo "수렴된 실험 확인:"
echo "  grep -l '수렴됨' ${LOG_DIR}/*.log"

echo ""
echo "실험 완료 후 결과 분석을 위한 Python 스크립트 실행 예시:"
echo "  python analyze_multi_beta_pref_results.py --result_dir ${RESULT_DIR}"

# EXPERIMENT_INFO_FILE에 모니터링 및 중단 명령어 추가
cat >> "${EXPERIMENT_INFO_FILE}" << EOF

=== 실험 진행 상황 모니터링 ===
로그 파일들:
EOF

for exploration_beta in "${EXPLORATION_BETAS[@]}"; do
    for pref_pair in "${TRUE_PREFERENCE_PAIRS[@]}"; do
        true_preference_1=$(echo $pref_pair | cut -d',' -f1)
        true_preference_2=$(echo $pref_pair | cut -d',' -f2)
        pref_str="${true_preference_1//.}${true_preference_2//.}"
        
        for combo in "${LEVEL_COMBINATIONS[@]}"; do
            buyer_level=$(echo $combo | cut -d',' -f1)
            seller_level=$(echo $combo | cut -d',' -f2)
            echo "  ${LOG_DIR}/beta${exploration_beta}_pref${pref_str}_L${buyer_level}B_vs_L${seller_level}S.log" >> "${EXPERIMENT_INFO_FILE}"
        done
    done
done

cat >> "${EXPERIMENT_INFO_FILE}" << EOF

실시간 로그 확인 명령어 예시:
  tail -f ${LOG_DIR}/beta${first_beta}_pref${first_pref_str}_L${buyer_level}B_vs_L${seller_level}S.log

모든 로그 동시 확인:
  tail -f ${LOG_DIR}/*.log

특정 탐험 베타의 모든 로그 확인 (예: beta${first_beta}):
  tail -f ${LOG_DIR}/beta${first_beta}_*.log

특정 선호도의 모든 로그 확인 (예: pref${first_pref_str}):
  tail -f ${LOG_DIR}/beta*_pref${first_pref_str}_*.log

=== 실험 중단 명령어 ===
1. 개별 프로세스 종료:
EOF

experiment_idx=0
for exploration_beta in "${EXPLORATION_BETAS[@]}"; do
    for pref_pair in "${TRUE_PREFERENCE_PAIRS[@]}"; do
        true_preference_1=$(echo $pref_pair | cut -d',' -f1)
        true_preference_2=$(echo $pref_pair | cut -d',' -f2)
        pref_str="${true_preference_1//.}${true_preference_2//.}"
        
        for combo in "${LEVEL_COMBINATIONS[@]}"; do
            buyer_level=$(echo $combo | cut -d',' -f1)
            seller_level=$(echo $combo | cut -d',' -f2)
            pid="${PIDS[$experiment_idx]}"
            echo "   kill ${pid}  # beta${exploration_beta}_pref${pref_str}_${buyer_level},${seller_level}" >> "${EXPERIMENT_INFO_FILE}"
            experiment_idx=$((experiment_idx + 1))
        done
    done
done

cat >> "${EXPERIMENT_INFO_FILE}" << EOF

2. 모든 실험 프로세스 한번에 종료:
   kill ${PIDS[*]}

3. 저장된 PID 파일로 종료:
   cat ${PID_FILE} | cut -d' ' -f1 | xargs kill

4. 강제 종료 (필요시):
   kill -9 ${PIDS[*]}

모니터링 명령어:
10초마다 완료 상태를 확인하려면 다음 명령어를 실행하세요:
  watch -n 10 'echo "=== \$(date) ==="; cat ${PID_FILE} | cut -d" " -f1 | xargs -I {} sh -c "kill -0 {} 2>/dev/null && echo PID {} 실행중 || echo PID {} 완료"'

모든 실험 완료까지 대기하려면 다음 명령어를 실행하세요:
  wait ${PIDS[*]} && echo "모든 실험 완료!" || echo "일부 실험 실패"

=== 결과 정리 ===
각 탐험 베타별로 결과 파일 확인:
EOF

for exploration_beta in "${EXPLORATION_BETAS[@]}"; do
    echo "  ls ${RESULT_DIR}/*beta${exploration_beta}_*.json" >> "${EXPERIMENT_INFO_FILE}"
done

echo "" >> "${EXPERIMENT_INFO_FILE}"
echo "각 선호도별로 결과 파일 확인:" >> "${EXPERIMENT_INFO_FILE}"
for pref_pair in "${TRUE_PREFERENCE_PAIRS[@]}"; do
    true_preference_1=$(echo $pref_pair | cut -d',' -f1)
    true_preference_2=$(echo $pref_pair | cut -d',' -f2)
    pref_str="${true_preference_1//.}${true_preference_2//.}"
    echo "  ls ${RESULT_DIR}/*pref${pref_str}_*.json" >> "${EXPERIMENT_INFO_FILE}"
done

cat >> "${EXPERIMENT_INFO_FILE}" << EOF

수렴된 실험 확인:
  grep -l '수렴됨' ${LOG_DIR}/*.log

실험 완료 후 결과 분석을 위한 Python 스크립트 실행 예시:
  python analyze_multi_beta_pref_results.py --result_dir ${RESULT_DIR}
EOF