import os
import json
import argparse
from datetime import datetime
from typing import Dict, Tuple, Protocol, Optional
import logging

import numpy as np
from scipy.special import softmax

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Simulate synthetic online learning experiment"
    )

    # Default values from original constants
    parser.add_argument(
        "--true_preference",
        nargs=2,
        type=float,
        default=[7.0, 3.0],
        help="True preference vector [apple, orange] (default: [7.0, 3.0])",
    )
    parser.add_argument(
        "--level_combinations",
        nargs="*",
        type=str,
        default=["0,1"],
        help="Level combinations as 'buyer,seller' pairs (default: ['0,1'])",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=1000,
        help="Maximum number of iterations (default: 1000)",
    )
    parser.add_argument(
        "--convergence_threshold",
        type=float,
        default=0.01,
        help="Convergence threshold (default: 0.01)",
    )
    parser.add_argument(
        "--default_beta",
        type=float,
        default=2.0,
        help="Default beta parameter for softmax (default: 2.0)",
    )
    parser.add_argument(
        "--default_n_items",
        type=int,
        default=2,
        help="Default number of items (default: 2)",
    )
    parser.add_argument(
        "--default_max_value",
        type=float,
        default=10.0,
        help="Default maximum value (default: 10.0)",
    )
    parser.add_argument(
        "--default_n_preference_points",
        type=int,
        default=101,
        help="Default number of preference points (default: 101)",
    )
    parser.add_argument(
        "--default_n_price_points",
        type=int,
        default=81,
        help="Default number of price points (default: 81)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-10,
        help="Small epsilon value (default: 1e-10)",
    )
    parser.add_argument(
        "--experiment_time",
        type=str,
        default=None,
        help="Experiment timestamp (default: current time)",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="Result directory (default: ./results/exp2/{timestamp}/)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=50,
        help="Window size for sliding window memory (default: 50)",
    )

    args = parser.parse_args()

    # Process level combinations
    level_combinations = []
    for combo in args.level_combinations:
        buyer_level, seller_level = map(int, combo.split(","))
        level_combinations.append((buyer_level, seller_level))
    args.level_combinations = level_combinations

    # Set experiment time if not provided
    if args.experiment_time is None:
        args.experiment_time = datetime.now().strftime("%Y%m%d_%H%M")

    # Set result directory if not provided
    if args.result_dir is None:
        args.result_dir = f"./results/exp2/{args.experiment_time}/"

    return args


# Parse arguments at module level
args = parse_args()

# Constants from parsed arguments
TRUE_PREFERENCE = np.array(args.true_preference)
LEVEL_COMBINATIONS = args.level_combinations
MAX_ITERATIONS = args.max_iterations
CONVERGENCE_THRESHOLD = args.convergence_threshold
DEFAULT_BETA = args.default_beta
DEFAULT_N_ITEMS = args.default_n_items
DEFAULT_MAX_VALUE = args.default_max_value
DEFAULT_N_PREFERENCE_POINTS = args.default_n_preference_points
DEFAULT_N_PRICE_POINTS = args.default_n_price_points
WINDOW_SIZE = args.window_size
EPSILON = args.epsilon
EXPERIMENT_TIME = args.experiment_time
RESULT_DIR = args.result_dir

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


class BuyerAgentProtocol(Protocol):
    """Buyer agent protocol for stage 1 policy P^{b,t=1}_{k=l}(a_1 | r, d)"""

    def get_P_t1(self, vector_d: np.ndarray, vector_r: np.ndarray) -> np.ndarray:
        """Return stage 1 policy P^{b,t=1}(a_1|r,d) for given preferences r and distances d"""
        ...


class SellerAgentProtocol(Protocol):
    """Seller agent protocol for Bayesian inference and price optimization"""

    def p_r_given_a(
        self, observed_action_a1: int, vector_d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bayesian inference: p^s(r | a_1) ∝ P^b(a_1 | r, d) * p(r)"""
        ...

    def compute_m_star(
        self, preference_grid_r: np.ndarray, posterior_p_r_given_a: np.ndarray
    ) -> np.ndarray:
        """Compute optimal prices m*_{k=l} by maximizing E[U^{t=3}_s(m)]"""
        ...


class NotationUtils:
    """Utility functions following mathematical notation"""

    @staticmethod
    def create_preference_grid_r(
        n_points: int = DEFAULT_N_PREFERENCE_POINTS,
    ) -> np.ndarray:
        """Create preference grid for Bayesian inference"""
        return np.linspace(0, DEFAULT_MAX_VALUE, n_points)

    @staticmethod
    def create_uniform_prior_p_r(size: int) -> np.ndarray:
        """Create uniform prior distribution p(r)"""
        return np.ones(size) / size

    @staticmethod
    def compute_softmax_policy_P(
        q_values: np.ndarray, beta: float = DEFAULT_BETA
    ) -> np.ndarray:
        """Compute softmax policy: P^{agent,t}(a | Q)"""
        q_values = np.array(q_values)
        if len(q_values) == 2:
            diff = beta * (q_values[0] - q_values[1])
            prob_0 = 1 / (1 + np.exp(-diff))
            return np.array([prob_0, 1 - prob_0])
        else:
            return softmax(beta * q_values)


class InformationTheoryMetrics:
    """Information theory metrics calculation"""

    @staticmethod
    def D_KL(p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence D_KL(p || q)"""
        p = p + EPSILON
        q = q + EPSILON
        p = p / np.sum(p)
        q = q / np.sum(q)
        return np.sum(p * np.log(p / q))


class BaseAgent:
    """Base agent class with common functionality"""

    def __init__(self, beta: float = DEFAULT_BETA):
        self.beta = beta

    def compute_softmax_policy_P(self, q_values: np.ndarray) -> np.ndarray:
        """Convert Q-values to softmax policy"""
        return NotationUtils.compute_softmax_policy_P(q_values, self.beta)

    def select_action_a(self, q_values: np.ndarray) -> int:
        """Select action according to softmax policy"""
        policy_probs = self.compute_softmax_policy_P(q_values)
        return np.random.choice(len(q_values), p=policy_probs)


class LevelLBuyer(BaseAgent):
    """Level-l buyer agent"""

    def __init__(
        self,
        level_l: int,
        seller_model: Optional[SellerAgentProtocol] = None,
        beta: float = DEFAULT_BETA,
    ):
        super().__init__(beta)
        self.level_l = level_l
        self.seller_model = seller_model

    def act_t1(self, state: Dict) -> int:
        """Stage 1 action selection"""
        if self.level_l == 0:
            # Level-0: myopic
            q_values_Q_b_t1_k0 = state["vector_r"] - state["vector_d"]
            return self.select_action_a(q_values_Q_b_t1_k0)
        else:
            # Level-l: strategic
            q_values_Q_b_t1_kl = self.compute_Q_b_t1_kl(
                state["vector_d"], state["vector_r"], self.seller_model, self.beta
            )
            return self.select_action_a(q_values_Q_b_t1_kl)

    def act_t3(self, state: Dict) -> int:
        """Stage 3 action selection"""
        q_values_Q_b_t3 = state["vector_r"] - state["vector_m"]
        return self.select_action_a(q_values_Q_b_t3)

    def get_P_t1(self, vector_d: np.ndarray, vector_r: np.ndarray) -> np.ndarray:
        """Get stage 1 policy P^{b,t=1}_{k=l}(a_1 | r, d)"""
        if self.level_l == 0:
            q_values_Q_b_t1_k0 = vector_r - vector_d
            return self.compute_softmax_policy_P(q_values_Q_b_t1_k0)
        else:
            q_values_Q_b_t1_kl = self.compute_Q_b_t1_kl(
                vector_d, vector_r, self.seller_model, self.beta
            )
            return self.compute_softmax_policy_P(q_values_Q_b_t1_kl)

    def compute_Q_b_t1_kl(
        self,
        vector_d: np.ndarray,
        vector_r: np.ndarray,
        seller_model: SellerAgentProtocol,
        beta: float = DEFAULT_BETA,
    ) -> np.ndarray:
        """Compute strategic Q-values for level-l buyer"""
        q_values_Q_b_t1 = np.zeros(DEFAULT_N_ITEMS)

        for action_a1 in range(DEFAULT_N_ITEMS):
            # Immediate utility
            immediate_U_b_t1 = vector_r[action_a1] - vector_d[action_a1]

            # Simulate seller's response
            preference_grid_r, posterior_p_r_given_a = seller_model.p_r_given_a(
                action_a1, vector_d
            )
            vector_m_star = seller_model.compute_m_star(
                preference_grid_r, posterior_p_r_given_a
            )

            # Future expected utility
            q_values_stage3 = vector_r - vector_m_star
            P_b_t3 = NotationUtils.compute_softmax_policy_P(q_values_stage3, beta)
            expected_U_b_t3 = np.sum(P_b_t3 * q_values_stage3)

            q_values_Q_b_t1[action_a1] = immediate_U_b_t1 + expected_U_b_t3

        return q_values_Q_b_t1


class IncrementalBayesianUpdater:
    """매 observation마다 점진적 베이지안 업데이트"""

    def __init__(self, n_preference_points: int = 101):
        self.preference_grid_r = NotationUtils.create_preference_grid_r(
            n_preference_points
        )
        self.current_prior = NotationUtils.create_uniform_prior_p_r(n_preference_points)
        self.belief_history = []  # KL divergence 추적용

    def update_belief(
        self,
        observed_action: int,
        vector_d: np.ndarray,
        buyer_model: BuyerAgentProtocol,
    ):
        """베이지안 업데이트 후 prior 갱신"""
        # 현재 prior를 사용하여 posterior 계산
        posterior_p_r_given_a = np.zeros_like(self.current_prior)

        for i, r_apple in enumerate(self.preference_grid_r):
            r_orange = DEFAULT_MAX_VALUE - r_apple
            vector_r = np.array([r_apple, r_orange])

            # Buyer model로부터 likelihood 계산
            P_b_t1 = buyer_model.get_P_t1(vector_d, vector_r)
            likelihood_P_b_a1 = P_b_t1[observed_action]

            # Posterior ∝ likelihood × prior
            posterior_p_r_given_a[i] = likelihood_P_b_a1 * self.current_prior[i]

        # Normalize
        if np.sum(posterior_p_r_given_a) > EPSILON:
            posterior_p_r_given_a = posterior_p_r_given_a / np.sum(
                posterior_p_r_given_a
            )
        else:
            # Fallback to uniform if all probabilities are zero
            posterior_p_r_given_a = NotationUtils.create_uniform_prior_p_r(
                len(self.preference_grid_r)
            )

        # Update prior for next iteration (belief accumulation)
        self.current_prior = posterior_p_r_given_a

        # Record history
        self.belief_history.append(posterior_p_r_given_a.copy())

    def get_current_belief(self) -> Tuple[np.ndarray, np.ndarray]:
        """현재 belief 반환"""
        return self.preference_grid_r, self.current_prior


class LevelLSeller(BaseAgent):
    """수정된 Seller - 온라인 학습 지원"""

    def __init__(
        self,
        level_l: int,
        buyer_model: Optional[BuyerAgentProtocol] = None,
        beta: float = DEFAULT_BETA,
        bayesian_updater: Optional[IncrementalBayesianUpdater] = None,
    ):
        super().__init__(beta)
        self.level_l = level_l
        self.buyer_model = buyer_model
        self.bayesian_updater = bayesian_updater  # 외부 updater 사용

    def p_r_given_a(
        self, observed_action_a1: int, vector_d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """온라인 베이지안 업데이트 사용"""
        if self.bayesian_updater is not None:
            self.bayesian_updater.update_belief(
                observed_action_a1, vector_d, self.buyer_model
            )
            return self.bayesian_updater.get_current_belief()
        else:
            # 기존 방식 fallback
            return self.p_r_given_a_impl(observed_action_a1, vector_d, self.buyer_model)

    def p_r_given_a_impl(
        self,
        observed_action_a1: int,
        vector_d: np.ndarray,
        buyer_model: BuyerAgentProtocol,
        n_preference_points: int = DEFAULT_N_PREFERENCE_POINTS,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Implement Bayesian inference: p^s(r | a_1) ∝ P^b(a_1 | r, d) * p(r)"""
        preference_grid_r = NotationUtils.create_preference_grid_r(n_preference_points)
        posterior_p_r_given_a = np.zeros_like(preference_grid_r)
        prior_p_r = NotationUtils.create_uniform_prior_p_r(len(preference_grid_r))

        for i, r_apple in enumerate(preference_grid_r):
            r_orange = DEFAULT_MAX_VALUE - r_apple
            vector_r = np.array([r_apple, r_orange])

            P_b_t1 = buyer_model.get_P_t1(vector_d, vector_r)
            likelihood_P_b_a1 = P_b_t1[observed_action_a1]
            posterior_p_r_given_a[i] = likelihood_P_b_a1 * prior_p_r[i]

        # Normalize
        posterior_p_r_given_a = posterior_p_r_given_a / np.sum(posterior_p_r_given_a)
        return preference_grid_r, posterior_p_r_given_a

    def compute_m_star(
        self, preference_grid_r: np.ndarray, posterior_p_r_given_a: np.ndarray
    ) -> np.ndarray:
        """Compute optimal prices m*_{k=l}"""
        if self.level_l == 0:
            # Level-0: uniform pricing
            return np.array([DEFAULT_MAX_VALUE / 2, DEFAULT_MAX_VALUE / 2])
        else:
            # Level-l: optimize prices
            return self.optimize_vector_m_star(
                preference_grid_r, posterior_p_r_given_a, self.beta
            )

    def optimize_vector_m_star(
        self,
        preference_grid_r: np.ndarray,
        posterior_p_r_given_a: np.ndarray,
        beta: float = DEFAULT_BETA,
        n_price_points: int = DEFAULT_N_PRICE_POINTS,
    ) -> np.ndarray:
        """Compute optimal prices m* by maximizing E[U^{t=3}_s(m)]"""
        best_vector_m = np.array([DEFAULT_MAX_VALUE / 2, DEFAULT_MAX_VALUE / 2])
        best_expected_U_s = 0

        for m_apple in np.linspace(0, DEFAULT_MAX_VALUE, n_price_points):
            m_orange = DEFAULT_MAX_VALUE - m_apple
            test_vector_m = np.array([m_apple, m_orange])

            expected_U_s = self._calculate_E_U_s_t3(
                preference_grid_r, posterior_p_r_given_a, test_vector_m, beta
            )

            if expected_U_s > best_expected_U_s:
                best_expected_U_s = expected_U_s
                best_vector_m = test_vector_m

        return best_vector_m

    def _calculate_E_U_s_t3(
        self,
        preference_grid_r: np.ndarray,
        posterior_p_r_given_a: np.ndarray,
        vector_m: np.ndarray,
        beta: float,
    ) -> float:
        """Calculate expected seller utility E[U^{t=3}_s(m)]"""
        expected_U_s = 0
        for i, r_apple in enumerate(preference_grid_r):
            r_orange = DEFAULT_MAX_VALUE - r_apple
            vector_r = np.array([r_apple, r_orange])

            q_values_b_t3 = vector_r - vector_m
            P_b_t3 = NotationUtils.compute_softmax_policy_P(q_values_b_t3, beta)
            U_s_t3 = np.sum(P_b_t3 * vector_m)
            expected_U_s += posterior_p_r_given_a[i] * U_s_t3

        return expected_U_s


class SlidingWindowMemory:
    """관찰된 action들을 sliding window로 관리"""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.observations = []  # (action, distance) 쌍들

    def add_observation(self, action: int, vector_d: np.ndarray):
        """새로운 관찰 추가"""
        self.observations.append((action, vector_d.copy()))
        if len(self.observations) > self.window_size:
            self.observations.pop(0)

    def get_observation_count(self) -> int:
        """현재 관찰 수 반환"""
        return len(self.observations)


class ConvergenceTracker:
    """수렴 추적 및 성능 측정"""

    def __init__(
        self, true_preference: np.ndarray, convergence_threshold: float = 0.01
    ):
        self.true_preference = true_preference
        self.threshold = convergence_threshold
        self.best_loss = float("inf")
        self.convergence_data = []  # best_loss 갱신 시점의 데이터
        self.all_losses = []  # 모든 loss 기록

    def check_convergence(self, current_belief: np.ndarray) -> Tuple[bool, float]:
        """KL divergence 기반 수렴 확인"""
        # True preference를 grid 상의 분포로 변환
        true_dist = self._convert_preference_to_distribution(self.true_preference)

        # KL divergence 계산
        current_loss = InformationTheoryMetrics.D_KL(true_dist, current_belief)
        self.all_losses.append(current_loss)

        # 수렴 여부 확인
        converged = current_loss < self.threshold

        return converged, current_loss

    def update_best_and_save(self, current_loss: float, learning_data: Dict):
        """best_loss 갱신 시 데이터 저장"""
        if current_loss < self.best_loss:
            self.best_loss = current_loss

            # 현재 상태 저장
            snapshot = {
                "iteration": learning_data.get("iteration", 0),
                "loss": current_loss,
                "belief": (
                    learning_data.get("current_belief", []).copy()
                    if isinstance(learning_data.get("current_belief"), np.ndarray)
                    else []
                ),
                "observation_count": learning_data.get("observation_count", 0),
                "last_action": learning_data.get("last_action", -1),
                "last_distance": (
                    learning_data.get("last_distance", []).copy()
                    if isinstance(learning_data.get("last_distance"), np.ndarray)
                    else []
                ),
            }

            self.convergence_data.append(snapshot)

    def _convert_preference_to_distribution(self, preference: np.ndarray) -> np.ndarray:
        """True preference를 grid 상의 분포로 변환"""
        # Grid에서 가장 가까운 점 찾기
        grid_points = NotationUtils.create_preference_grid_r(
            DEFAULT_N_PREFERENCE_POINTS
        )

        # Apple preference에 해당하는 grid point 찾기
        apple_pref = preference[0]
        closest_idx = np.argmin(np.abs(grid_points - apple_pref))

        # Delta function (one-hot distribution)
        true_dist = np.zeros(len(grid_points))
        true_dist[closest_idx] = 1.0

        return true_dist

    def get_convergence_summary(self) -> Dict:
        """수렴 결과 요약"""
        return {
            "converged": self.best_loss < self.threshold,
            "best_loss": self.best_loss,
            "final_loss": self.all_losses[-1] if self.all_losses else float("inf"),
            "num_improvements": len(self.convergence_data),
            "all_losses": self.all_losses,
            "convergence_data": self.convergence_data,
        }


class OnlineLearningEnvironment:
    """온라인 학습 환경"""

    def __init__(
        self,
        buyer_level: int,
        seller_level: int,
        true_preference: np.ndarray,
        beta: float = 2.0,
        window_size: int = 50,
        convergence_threshold: float = 0.01,
    ):
        self.buyer_level = buyer_level
        self.seller_level = seller_level
        self.true_preference = true_preference
        self.beta = beta

        # 컴포넌트들 초기화
        self.memory = SlidingWindowMemory(window_size)
        self.bayesian_updater = IncrementalBayesianUpdater()
        self.convergence_tracker = ConvergenceTracker(
            true_preference, convergence_threshold
        )

        # Agent 초기화 (circular dependency 해결)
        self._initialize_agents()

    def _initialize_agents(self):
        """Agent들을 초기화 (circular dependency 해결)"""
        # 먼저 Level-0 buyer 생성 (seller model 불필요)
        level0_buyer = LevelLBuyer(level_l=0, seller_model=None, beta=self.beta)

        # Seller 생성 (buyer model과 bayesian updater 사용)
        if self.seller_level == 0:
            buyer_model = level0_buyer
        else:
            # Higher level sellers need appropriate buyer models
            buyer_model = level0_buyer  # 우선 level-0으로 시작

        self.seller = LevelLSeller(
            level_l=self.seller_level,
            buyer_model=buyer_model,
            beta=self.beta,
            bayesian_updater=self.bayesian_updater,
        )

        # Buyer 생성
        if self.buyer_level == 0:
            self.buyer = level0_buyer
        else:
            self.buyer = LevelLBuyer(
                level_l=self.buyer_level, seller_model=self.seller, beta=self.beta
            )

        # Seller의 buyer model을 실제 buyer로 업데이트 (더 정확한 모델링을 위해)
        if self.seller_level > 0:
            # Create appropriate level buyer model for seller
            if self.buyer_level > 0:
                seller_buyer_model = LevelLBuyer(
                    level_l=max(
                        0, self.buyer_level - 1
                    ),  # Seller assumes buyer is one level lower
                    seller_model=None,  # 순환 참조 방지
                    beta=self.beta,
                )
            else:
                seller_buyer_model = level0_buyer

            self.seller.buyer_model = seller_buyer_model

    def play_single_game(self) -> Dict:
        """단일 게임 실행"""
        # 랜덤한 distance 생성
        total_distance = np.random.uniform(2, 8)  # 적당한 범위
        d_apple = np.random.uniform(0.5, total_distance - 0.5)
        d_orange = total_distance - d_apple
        vector_d = np.array([d_apple, d_orange])

        # Stage 1: Buyer action
        state_t1 = {"vector_r": self.true_preference, "vector_d": vector_d}

        action_a1 = self.buyer.act_t1(state_t1)

        # Seller observes and updates belief
        preference_grid_r, posterior_belief = self.seller.p_r_given_a(
            action_a1, vector_d
        )

        # Stage 2: Seller sets prices
        optimal_prices = self.seller.compute_m_star(preference_grid_r, posterior_belief)

        # Stage 3: Buyer's final action
        state_t3 = {"vector_r": self.true_preference, "vector_m": optimal_prices}

        action_a3 = self.buyer.act_t3(state_t3)

        # Memory에 관찰 저장
        self.memory.add_observation(action_a1, vector_d)

        # 게임 결과 반환
        game_result = {
            "stage1_action": action_a1,
            "stage3_action": action_a3,
            "vector_d": vector_d,
            "optimal_prices": optimal_prices,
            "seller_belief": posterior_belief,
            "true_preference": self.true_preference,
        }

        return game_result

    def run_learning_experiment(self, max_iterations: int = 1000) -> Dict:
        """온라인 학습 실험 실행"""
        logger.info(
            f"온라인 학습 시작: 구매자 L{self.buyer_level} vs 판매자 L{self.seller_level}"
        )

        experiment_results = {
            "buyer_level": self.buyer_level,
            "seller_level": self.seller_level,
            "true_preference": self.true_preference.tolist(),
            "games": [],
            "convergence_info": {},
        }

        for iteration in range(max_iterations):
            # 게임 실행
            game_result = self.play_single_game()

            # 수렴 체크
            _, current_belief = self.bayesian_updater.get_current_belief()
            converged, current_loss = self.convergence_tracker.check_convergence(
                current_belief
            )

            # 학습 데이터 구성
            learning_data = {
                "iteration": iteration,
                "current_belief": current_belief,
                "observation_count": self.memory.get_observation_count(),
                "last_action": game_result["stage1_action"],
                "last_distance": game_result["vector_d"],
            }

            # Best loss 업데이트 및 데이터 저장
            self.convergence_tracker.update_best_and_save(current_loss, learning_data)

            # 게임 결과에 추가 정보 포함
            game_result.update(
                {
                    "iteration": iteration,
                    "current_loss": current_loss,
                    "converged": converged,
                    "observation_count": self.memory.get_observation_count(),
                }
            )

            experiment_results["games"].append(game_result)

            # 진행 상황 로깅
            if iteration % 100 == 0:
                logger.info(
                    f"반복 {iteration}: 손실 = {current_loss:.6f}, 최고 = {self.convergence_tracker.best_loss:.6f}"
                )

            # 조기 수렴 체크
            if converged:
                logger.info(f"반복 {iteration}에서 수렴됨, 손실 {current_loss:.6f}")
                break

        # 수렴 정보 추가
        experiment_results["convergence_info"] = (
            self.convergence_tracker.get_convergence_summary()
        )

        logger.info(f"실험 완료. 최종 손실: {current_loss:.6f}")

        return experiment_results


# numpy 배열을 리스트로 변환하여 JSON 직렬화 가능하게 만들기
def convert_arrays_to_lists(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_arrays_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_arrays_to_lists(item) for item in obj]
    else:
        return obj


def run_online_learning_experiment(
    buyer_level: int,
    seller_level: int,
    true_preference: np.ndarray,
    max_iterations: int = 1000,
    window_size: int = 50,
    convergence_threshold: float = 0.01,
) -> Dict:
    """온라인 학습 실험 실행"""

    # 환경 생성
    env = OnlineLearningEnvironment(
        buyer_level=buyer_level,
        seller_level=seller_level,
        true_preference=true_preference,
        window_size=window_size,
        convergence_threshold=convergence_threshold,
    )

    # 학습 실행
    results = env.run_learning_experiment(max_iterations)

    return results


if __name__ == "__main__":
    # 실험 설정
    true_preference = TRUE_PREFERENCE  # 고정된 true preference
    logger.info(f"실험 설정: 참조 선호도 = {true_preference}")

    # 다양한 level 조합 실험
    level_combinations = LEVEL_COMBINATIONS

    for buyer_level, seller_level in level_combinations:
        logger.info(f"실험 실행: 구매자 L{buyer_level} vs 판매자 L{seller_level}")

        results = run_online_learning_experiment(
            buyer_level=buyer_level,
            seller_level=seller_level,
            true_preference=true_preference,
            max_iterations=MAX_ITERATIONS,
            window_size=WINDOW_SIZE,
            convergence_threshold=CONVERGENCE_THRESHOLD,
        )

        # 결과 저장
        result_file = f"{RESULT_DIR}/L{buyer_level}B_vs_L{seller_level}S_learning.json"
        serializable_results = convert_arrays_to_lists(results)

        with open(result_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
