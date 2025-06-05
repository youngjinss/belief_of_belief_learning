import os
import json
from datetime import datetime
from typing import Dict, Tuple, Protocol, Optional
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import multiprocessing as mp

import numpy as np
from scipy.special import softmax

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_LEVEL = 2
N_SAMPLES = 15
DEFAULT_BETA = 2.0
DEFAULT_N_ITEMS = 2
DEFAULT_MAX_VALUE = 10.0
DEFAULT_N_PREFERENCE_POINTS = 81
DEFAULT_N_PRICE_POINTS = 51
EPSILON = 1e-10
EXPERIMENT_TIME = datetime.now().strftime("%Y%m%d_%H%M")
RESULT_DIR = f"./results/exp2/{EXPERIMENT_TIME}/"
PLOT_DIR = f"{RESULT_DIR}/analysis_plot.png"
DATA_DIR = f"{RESULT_DIR}/simulation_results.json"

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


class BuyerAgentProtocol(Protocol):
    """Protocol for buyer agents based on notation"""

    def get_P_t1(self, vector_d: np.ndarray, vector_r: np.ndarray) -> np.ndarray: ...


class SellerAgentProtocol(Protocol):
    """Protocol for seller agents based on notation"""

    def p_r_given_a(
        self, observed_action_a1: int, vector_d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]: ...

    def compute_m_star(
        self, preference_grid_r: np.ndarray, posterior_p_r_given_a: np.ndarray
    ) -> np.ndarray: ...


class NotationUtils:
    """Utility functions following mathematical notation"""

    @staticmethod
    def normalize_vector_to_sum(
        values: np.ndarray, target_sum: float = DEFAULT_MAX_VALUE
    ) -> np.ndarray:
        """Normalize vector to sum to target value (constraint: r + d + m = 10)"""
        return values * target_sum / values.sum()

    @staticmethod
    def create_preference_grid_r(
        n_points: int = DEFAULT_N_PREFERENCE_POINTS,
    ) -> np.ndarray:
        """Create preference grid for Bayesian inference over **r**"""
        return np.linspace(0, DEFAULT_MAX_VALUE, n_points)

    @staticmethod
    def create_uniform_prior_p_r(size: int) -> np.ndarray:
        """Create uniform prior distribution p(**r**)"""
        return np.ones(size) / size

    @staticmethod
    def compute_softmax_policy_P(
        q_values: np.ndarray, beta: float = DEFAULT_BETA
    ) -> np.ndarray:
        """Compute softmax policy P^{agent,t} following notation"""
        q_values = np.array(q_values)
        if len(q_values) == 2:
            diff = beta * (q_values[0] - q_values[1])
            prob_0 = 1 / (1 + np.exp(-diff))
            return np.array([prob_0, 1 - prob_0])
        else:
            return softmax(beta * q_values)


class BayesianIRLMixin:
    """Mixin for Bayesian IRL functionality following p^s(**r** | a_1) notation"""

    def p_r_given_a_impl(
        self,
        observed_action_a1: int,
        vector_d: np.ndarray,
        buyer_model: BuyerAgentProtocol,
        n_preference_points: int = DEFAULT_N_PREFERENCE_POINTS,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bayesian IRL: p^s(**r** | a_1) ∝ P^b(a_1 | **r**, **d**) * p(**r**)"""
        preference_grid_r = NotationUtils.create_preference_grid_r(n_preference_points)
        posterior_p_r_given_a = np.zeros_like(preference_grid_r)
        prior_p_r = NotationUtils.create_uniform_prior_p_r(len(preference_grid_r))

        for i, r_apple in enumerate(preference_grid_r):
            r_orange = DEFAULT_MAX_VALUE - r_apple
            vector_r = np.array([r_apple, r_orange])

            # Get P^b(a_1 | **r**, **d**) from buyer model
            P_b_t1 = buyer_model.get_P_t1(vector_d, vector_r)
            likelihood_P_b_a1 = P_b_t1[observed_action_a1]
            posterior_p_r_given_a[i] = likelihood_P_b_a1 * prior_p_r[i]

        # Normalize posterior
        posterior_p_r_given_a = posterior_p_r_given_a / np.sum(posterior_p_r_given_a)
        return preference_grid_r, posterior_p_r_given_a


class PriceOptimizationMixin:
    """Mixin for price optimization: **m***_{k} following notation"""

    def optimize_vector_m_star(
        self,
        preference_grid_r: np.ndarray,
        posterior_p_r_given_a: np.ndarray,
        beta: float = DEFAULT_BETA,
        n_price_points: int = DEFAULT_N_PRICE_POINTS,
    ) -> np.ndarray:
        """Optimize **m***_{k} with constraint that prices sum to max value"""
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
        """Calculate E[U^{t=3}_s(**m**)] for given prices"""
        expected_U_s = 0
        for i, r_apple in enumerate(preference_grid_r):
            r_orange = DEFAULT_MAX_VALUE - r_apple
            vector_r = np.array([r_apple, r_orange])

            # Q^{b,t=3} = **r** - **m**
            q_values_b_t3 = vector_r - vector_m
            P_b_t3 = NotationUtils.compute_softmax_policy_P(q_values_b_t3, beta)

            # U^{t=3}_s = **m** · P^b(a_3 | **m**)
            U_s_t3 = np.sum(P_b_t3 * vector_m)
            expected_U_s += posterior_p_r_given_a[i] * U_s_t3

        return expected_U_s


class StrategicPlanningMixin:
    """Mixin for strategic planning: Q^{b,t=1}_{k=l} following notation"""

    def compute_Q_b_t1_kl(
        self,
        vector_d: np.ndarray,
        vector_r: np.ndarray,
        seller_model: SellerAgentProtocol,
        beta: float = DEFAULT_BETA,
    ) -> np.ndarray:
        """Compute Q^{b,t=1}_{k=l} considering seller's response **m***_{k=(l-1)}"""
        q_values_Q_b_t1 = np.zeros(DEFAULT_N_ITEMS)

        for action_a1 in range(DEFAULT_N_ITEMS):
            # U^{t=1}_b = **r** - **d**
            immediate_U_b_t1 = vector_r[action_a1] - vector_d[action_a1]

            # Simulate seller's response: p^s(**r** | a_1) and **m***_{k=(l-1)}
            preference_grid_r, posterior_p_r_given_a = seller_model.p_r_given_a(
                action_a1, vector_d
            )
            vector_m_star = seller_model.compute_m_star(
                preference_grid_r, posterior_p_r_given_a
            )

            # E[U^{t=3}_b(**m***)]
            q_values_stage3 = vector_r - vector_m_star
            P_b_t3 = NotationUtils.compute_softmax_policy_P(q_values_stage3, beta)
            expected_U_b_t3 = np.sum(P_b_t3 * q_values_stage3)

            # Q^{b,t=1}_{k=l} = U^{t=1}_b + E[U^{t=3}_b]
            q_values_Q_b_t1[action_a1] = immediate_U_b_t1 + expected_U_b_t3

        return q_values_Q_b_t1


class BaseAgent:
    """Base agent class with common functionality"""

    def __init__(self, beta: float = DEFAULT_BETA):
        self.beta = beta

    def compute_softmax_policy_P(self, q_values: np.ndarray) -> np.ndarray:
        """Convert Q-values to P^{agent,t} using softmax"""
        return NotationUtils.compute_softmax_policy_P(q_values, self.beta)

    def select_action_a(self, q_values: np.ndarray) -> int:
        """Select action a based on softmax policy P^{agent,t}"""
        policy_probs = self.compute_softmax_policy_P(q_values)
        return np.random.choice(len(q_values), p=policy_probs)


class BuyerSellerEnvironment:
    """3-stage buyer-seller interaction environment following notation"""

    def __init__(
        self,
        n_items: int = DEFAULT_N_ITEMS,
        max_distance: float = DEFAULT_MAX_VALUE,
        max_price: float = DEFAULT_MAX_VALUE,
    ):
        self.n_items = n_items
        self.max_distance = max_distance
        self.max_price = max_price
        self.items = ["apple", "orange"]

    def reset(self, vector_r: np.ndarray, vector_d: np.ndarray):
        """Reset environment with buyer preferences **r** and distances **d**"""
        assert len(vector_r) == self.n_items
        assert len(vector_d) == self.n_items

        self.vector_r = NotationUtils.normalize_vector_to_sum(vector_r)
        self.vector_d = NotationUtils.normalize_vector_to_sum(vector_d)

        self.stage_t = 1
        self.buyer_choice_a1 = None
        self.vector_m = None

        return self.get_state()

    def get_state(self):
        """Get current environment state"""
        return {
            "stage_t": self.stage_t,
            "vector_d": self.vector_d,
            "vector_r": self.vector_r,
            "buyer_choice_a1": self.buyer_choice_a1,
            "vector_m": self.vector_m,
        }

    def step(self, action: Dict):
        """Execute action and move to next stage"""
        if self.stage_t == 1:
            self.buyer_choice_a1 = action["buyer_choice_a1"]
            self.stage_t = 2

        elif self.stage_t == 2:
            self.vector_m = action["vector_m"]
            self.stage_t = 3

        elif self.stage_t == 3:
            buyer_choice_a3 = action["buyer_choice_a3"]

            # Calculate U^{total}_b and U^{total}_s
            utility_U_b_t1 = (
                self.vector_r[self.buyer_choice_a1]
                - self.vector_d[self.buyer_choice_a1]
            )
            utility_U_b_t3 = (
                self.vector_r[buyer_choice_a3] - self.vector_m[buyer_choice_a3]
            )
            utility_U_b_total = utility_U_b_t1 + utility_U_b_t3

            utility_U_s_total = self.vector_m[buyer_choice_a3]

            self.stage_t = 4

            return {
                "utility_U_b_total": utility_U_b_total,
                "utility_U_s_total": utility_U_s_total,
                "done": True,
            }

        return {"done": False}


class LevelLBuyer(BaseAgent, StrategicPlanningMixin):
    """Level-l buyer: Q^{b,t}_{k=l} - generalized for any level l"""

    def __init__(
        self,
        level_l: int,
        seller_model: Optional[SellerAgentProtocol] = None,
        beta: float = DEFAULT_BETA,
    ):
        super().__init__(beta)
        self.level_l = level_l
        self.seller_model = seller_model  # Level-(l-1) seller model

    def act_t1(self, state: Dict) -> int:
        """Choose item in t=1 based on Q^{b,t=1}_{k=l}"""
        if self.level_l == 0:
            # Base case: Q^{b,t=1}_{k=0} = U^{t=1}_b
            q_values_Q_b_t1_k0 = state["vector_r"] - state["vector_d"]
            return self.select_action_a(q_values_Q_b_t1_k0)
        else:
            # Recursive case: Q^{b,t=1}_{k=l} using Level-(l-1) seller
            q_values_Q_b_t1_kl = self.compute_Q_b_t1_kl(
                state["vector_d"], state["vector_r"], self.seller_model, self.beta
            )
            return self.select_action_a(q_values_Q_b_t1_kl)

    def act_t3(self, state: Dict) -> int:
        """Choose item in t=3 based on Q^{b,t=3}_{k=l} = Q^{b,t=3}_{k=0}"""
        # All levels use same t=3 strategy: Q^{b,t=3} = **r** - **m**
        q_values_Q_b_t3 = state["vector_r"] - state["vector_m"]
        return self.select_action_a(q_values_Q_b_t3)

    def get_P_t1(self, vector_d: np.ndarray, vector_r: np.ndarray) -> np.ndarray:
        """Get P^{b,t=1}_{k=l}(a_1 | **r**, **d**)"""
        if self.level_l == 0:
            # Base case: P^{b,t=1}_{k=0}
            q_values_Q_b_t1_k0 = vector_r - vector_d
            return self.compute_softmax_policy_P(q_values_Q_b_t1_k0)
        else:
            # Recursive case: P^{b,t=1}_{k=l}
            q_values_Q_b_t1_kl = self.compute_Q_b_t1_kl(
                vector_d, vector_r, self.seller_model, self.beta
            )
            return self.compute_softmax_policy_P(q_values_Q_b_t1_kl)


class LevelLSeller(BaseAgent, BayesianIRLMixin, PriceOptimizationMixin):
    """Level-l seller: p^s(**r** | a_1) and **m***_{k=l} - generalized for any level l"""

    def __init__(
        self,
        level_l: int,
        buyer_model: Optional[BuyerAgentProtocol] = None,
        beta: float = DEFAULT_BETA,
        n_preference_points: int = DEFAULT_N_PREFERENCE_POINTS,
        defensive: bool = False,
    ):
        super().__init__(beta)
        self.level_l = level_l
        self.buyer_model = buyer_model  # Level-(l-1) buyer model
        self.n_preference_points = n_preference_points
        self.defensive = defensive

    def p_r_given_a(
        self, observed_action_a1: int, vector_d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform p^s(**r** | a_1) ∝ P^b_{k=(l-1)}(a_1 | **r**, **d**) * p(**r**)"""
        if self.level_l == 0:
            # Base case: random/uniform pricing (ignore buyer behavior)
            preference_grid_r = NotationUtils.create_preference_grid_r(
                self.n_preference_points
            )
            uniform_posterior = NotationUtils.create_uniform_prior_p_r(
                len(preference_grid_r)
            )
            return preference_grid_r, uniform_posterior
        else:
            # Recursive case: use Level-(l-1) buyer model for Bayesian IRL
            return self.p_r_given_a_impl(
                observed_action_a1, vector_d, self.buyer_model, self.n_preference_points
            )

    def compute_m_star(
        self, preference_grid_r: np.ndarray, posterior_p_r_given_a: np.ndarray
    ) -> np.ndarray:
        """Compute **m***_{k=l} based on posterior beliefs"""
        if self.level_l == 0:
            # Base case: random/uniform pricing
            return np.array([DEFAULT_MAX_VALUE / 2, DEFAULT_MAX_VALUE / 2])
        elif self.defensive:
            # Defensive pricing against strategic buyer
            return self._compute_defensive_pricing(
                preference_grid_r, posterior_p_r_given_a
            )
        else:
            # Standard pricing optimization
            return self.optimize_vector_m_star(
                preference_grid_r, posterior_p_r_given_a, self.beta
            )

    def _compute_defensive_pricing(
        self, preference_grid_r: np.ndarray, posterior_p_r_given_a: np.ndarray
    ) -> np.ndarray:
        """Compute defensive **m***_{k=l} against strategic buyer"""
        vector_m = np.zeros(DEFAULT_N_ITEMS)
        n_price_points = DEFAULT_N_PRICE_POINTS

        for item in range(DEFAULT_N_ITEMS):
            best_price = 0
            best_expected_U_s = 0

            for price in np.linspace(0, DEFAULT_MAX_VALUE, n_price_points):
                expected_U_s = 0

                for i, r_apple in enumerate(preference_grid_r):
                    r_orange = DEFAULT_MAX_VALUE - r_apple
                    vector_r = np.array([r_apple, r_orange])

                    # Test price vector **m**
                    test_vector_m = (
                        np.array([price, DEFAULT_MAX_VALUE - price])
                        if item == 0
                        else np.array([DEFAULT_MAX_VALUE - price, price])
                    )

                    # Q^{b,t=3} = **r** - **m**
                    q_values_Q_b_t3 = vector_r - test_vector_m
                    P_b_t3 = NotationUtils.compute_softmax_policy_P(
                        q_values_Q_b_t3, self.beta
                    )

                    expected_U_s += posterior_p_r_given_a[i] * P_b_t3[item] * price

                if expected_U_s > best_expected_U_s:
                    best_expected_U_s = expected_U_s
                    best_price = price

            vector_m[item] = best_price

        # Ensure constraint: sum(**m**) = max_value
        return NotationUtils.normalize_vector_to_sum(vector_m)


class InformationTheoryMetrics:
    """Calculate information-theoretic metrics I(**r**, a_1) and D_KL following notation"""

    @staticmethod
    def compute_I_r_a1(preference_values: np.ndarray, policies: np.ndarray) -> float:
        """Calculate I(**r**, a_1) between preferences and actions"""
        n_prefs, n_actions = policies.shape

        # Marginal distribution over actions
        action_probs = np.mean(policies, axis=0)

        # Calculate MI using definition
        mi = 0
        for i in range(n_prefs):
            for a in range(n_actions):
                if policies[i, a] > EPSILON:  # Avoid log(0)
                    mi += (
                        (1 / n_prefs)
                        * policies[i, a]
                        * np.log(policies[i, a] / action_probs[a])
                    )

        return mi

    @staticmethod
    def D_KL(p: np.ndarray, q: np.ndarray) -> float:
        """Calculate D_KL(p || q)"""
        # Avoid log(0) by adding small epsilon
        p = p + EPSILON
        q = q + EPSILON
        p = p / np.sum(p)
        q = q / np.sum(q)
        return np.sum(p * np.log(p / q))

    @staticmethod
    def update_belief(
        prior_p_r: np.ndarray, posterior_p_r_given_a: np.ndarray
    ) -> float:
        """Measure D_KL(p(**r** | a_1) || p(**r**))"""
        return InformationTheoryMetrics.D_KL(posterior_p_r_given_a, prior_p_r)


def create_agent_hierarchy(max_level: int, beta: float = DEFAULT_BETA) -> Dict:
    """Create agent hierarchy up to max_level recursively"""
    agents = {}

    # Base case: Level-0 buyer
    agents["buyer_l0"] = LevelLBuyer(level_l=0, seller_model=None, beta=beta)

    # Build hierarchy recursively
    for level in range(1, max_level + 1):
        # Level-l seller uses Level-(l-1) buyer model
        buyer_model_prev = agents[f"buyer_l{level-1}"]
        seller_standard = LevelLSeller(
            level_l=level, buyer_model=buyer_model_prev, beta=beta, defensive=False
        )
        seller_defensive = LevelLSeller(
            level_l=level, buyer_model=buyer_model_prev, beta=beta, defensive=True
        )

        agents[f"seller_l{level}"] = seller_standard
        agents[f"seller_l{level}_defensive"] = seller_defensive

        # Level-l buyer uses Level-l seller model
        buyer_standard = LevelLBuyer(
            level_l=level, seller_model=seller_standard, beta=beta
        )
        buyer_advanced = LevelLBuyer(
            level_l=level, seller_model=seller_defensive, beta=beta
        )

        agents[f"buyer_l{level}"] = buyer_standard
        agents[f"buyer_l{level}_advanced"] = buyer_advanced

    return agents


def compute_single_combination(args):
    """각 (r_apple, d_apple) 조합에 대한 계산을 수행하는 worker 함수"""
    r_apple, d_apple, beta, max_level = args

    # Create generalized agent hierarchy for this worker
    agents = create_agent_hierarchy(max_level=max_level, beta=beta)

    vector_r = np.array([r_apple, DEFAULT_MAX_VALUE - r_apple])
    vector_d = np.array([d_apple, DEFAULT_MAX_VALUE - d_apple])

    # Get policies P^{agent,t=1} from generalized agents
    P_level0_current = agents["buyer_l0"].get_P_t1(vector_d, vector_r)
    P_level1_current = agents["buyer_l1"].get_P_t1(vector_d, vector_r)
    P_level1_advanced_current = agents["buyer_l1_advanced"].get_P_t1(vector_d, vector_r)
    P_level2_advanced_current = agents["buyer_l2_advanced"].get_P_t1(vector_d, vector_r)

    # Calculate belief updates D_KL for both actions a_1
    belief_updates_level1 = []
    belief_updates_level1_def = []
    belief_updates_level2_def = []

    for action_a1 in range(DEFAULT_N_ITEMS):
        # Level1 seller belief update
        _, posterior_p_r_given_a_level1 = agents["seller_l1"].p_r_given_a(
            action_a1, vector_d
        )
        prior_p_r = NotationUtils.create_uniform_prior_p_r(
            len(posterior_p_r_given_a_level1)
        )
        belief_update_D_KL_level1 = InformationTheoryMetrics.update_belief(
            prior_p_r, posterior_p_r_given_a_level1
        )
        belief_updates_level1.append(belief_update_D_KL_level1)

        # Level1 defensive seller belief update
        _, posterior_p_r_given_a_defensive = agents["seller_l1_defensive"].p_r_given_a(
            action_a1, vector_d
        )
        belief_update_D_KL_defensive = InformationTheoryMetrics.update_belief(
            prior_p_r, posterior_p_r_given_a_defensive
        )
        belief_updates_level1_def.append(belief_update_D_KL_defensive)

        # Level2 defensive seller belief update
        _, posterior_p_r_given_a_level2_def = agents["seller_l2_defensive"].p_r_given_a(
            action_a1, vector_d
        )
        belief_update_D_KL_level2_def = InformationTheoryMetrics.update_belief(
            prior_p_r, posterior_p_r_given_a_level2_def
        )
        belief_updates_level2_def.append(belief_update_D_KL_level2_def)

    return {
        "r_apple": r_apple,
        "d_apple": d_apple,
        "policies": {
            "P_level0": P_level0_current,
            "P_level1": P_level1_current,
            "P_level1_advanced": P_level1_advanced_current,
            "P_level2_advanced": P_level2_advanced_current,
        },
        "belief_updates": {
            "level1": belief_updates_level1,
            "level1_def": belief_updates_level1_def,
            "level2_def": belief_updates_level2_def,
        },
    }


def run_systematic_analysis(
    beta: float = DEFAULT_BETA,
    n_samples: int = 20,
    max_level: int = 2,
    n_processes: int = None,
):
    """Run systematic analysis with generalized level-l agents using parallel processing"""

    if n_processes is None:
        n_processes = mp.cpu_count() - 1  # 하나의 CPU는 남겨둠

    # Create parameter grids for **r** and **d**
    candidate_vector_r = np.linspace(1, 9, n_samples)
    candidate_vector_d = np.linspace(1, 9, n_samples)

    # Store results
    results = {
        "L0B_vs_L1S": {
            "mutual_info_I_r_a1": [],
            "belief_updates_D_KL": [],
            "policies_P": [],
        },
        "L1B_vs_L1S": {
            "mutual_info_I_r_a1": [],
            "belief_updates_D_KL": [],
            "policies_P": [],
        },
        "L1B_vs_L1S_D": {
            "mutual_info_I_r_a1": [],
            "belief_updates_D_KL": [],
            "policies_P": [],
        },
        "L2B_vs_L2S_D": {
            "mutual_info_I_r_a1": [],
            "belief_updates_D_KL": [],
            "policies_P": [],
        },
    }

    logger.info(f"Starting parallel computation with {n_processes} processes")
    logger.info(
        f"Total combinations: {len(candidate_vector_r)} × {len(candidate_vector_d)} = {len(candidate_vector_r) * len(candidate_vector_d)}"
    )

    # 모든 (r_apple, d_apple) 조합 생성
    all_combinations = [
        (r_apple, d_apple, beta, max_level)
        for r_apple in candidate_vector_r
        for d_apple in candidate_vector_d
    ]

    # 병렬 처리
    with mp.Pool(processes=n_processes) as pool:
        # tqdm으로 진행 상황 표시
        parallel_results = list(
            tqdm(
                pool.imap(compute_single_combination, all_combinations),
                total=len(all_combinations),
                desc="Processing combinations in parallel",
            )
        )

    logger.info("Parallel computation completed. Aggregating results...")

    # 결과를 r_apple별로 그룹화하고 집계
    for pref_idx, r_apple in tqdm(
        enumerate(candidate_vector_r),
        desc="Aggregating results by preference",
        total=len(candidate_vector_r),
    ):

        # 현재 r_apple에 해당하는 결과들 필터링
        current_r_results = [
            r for r in parallel_results if np.isclose(r["r_apple"], r_apple)
        ]

        # 정렬 (d_apple 순서대로)
        current_r_results.sort(key=lambda x: x["d_apple"])

        # 각 레벨의 정책들 수집
        P_level0 = np.array([r["policies"]["P_level0"] for r in current_r_results])
        P_level1 = np.array([r["policies"]["P_level1"] for r in current_r_results])
        P_level1_advanced = np.array(
            [r["policies"]["P_level1_advanced"] for r in current_r_results]
        )
        P_level2_advanced = np.array(
            [r["policies"]["P_level2_advanced"] for r in current_r_results]
        )

        # Belief updates 수집
        belief_updates_D_KL_level1_seller = []
        belief_updates_D_KL_level1_seller_defensive = []
        belief_updates_D_KL_level2_seller_defensive = []

        for r in current_r_results:
            belief_updates_D_KL_level1_seller.extend(r["belief_updates"]["level1"])
            belief_updates_D_KL_level1_seller_defensive.extend(
                r["belief_updates"]["level1_def"]
            )
            belief_updates_D_KL_level2_seller_defensive.extend(
                r["belief_updates"]["level2_def"]
            )

        # Calculate mutual information I(**r**, a_1)
        mi_I_r_a1_level0 = InformationTheoryMetrics.compute_I_r_a1(
            np.array([r_apple]), P_level0
        )
        mi_I_r_a1_level1 = InformationTheoryMetrics.compute_I_r_a1(
            np.array([r_apple]), P_level1
        )
        mi_I_r_a1_level1_advanced = InformationTheoryMetrics.compute_I_r_a1(
            np.array([r_apple]), P_level1_advanced
        )
        mi_I_r_a1_level2_advanced = InformationTheoryMetrics.compute_I_r_a1(
            np.array([r_apple]), P_level2_advanced
        )

        # Store results
        results["L0B_vs_L1S"]["mutual_info_I_r_a1"].append(mi_I_r_a1_level0)
        results["L0B_vs_L1S"]["belief_updates_D_KL"].append(
            np.mean(belief_updates_D_KL_level1_seller)
        )
        results["L0B_vs_L1S"]["policies_P"].append(P_level0.mean(axis=0))

        results["L1B_vs_L1S"]["mutual_info_I_r_a1"].append(mi_I_r_a1_level1)
        results["L1B_vs_L1S"]["belief_updates_D_KL"].append(
            np.mean(belief_updates_D_KL_level1_seller)
        )
        results["L1B_vs_L1S"]["policies_P"].append(P_level1.mean(axis=0))

        results["L1B_vs_L1S_D"]["mutual_info_I_r_a1"].append(mi_I_r_a1_level1_advanced)
        results["L1B_vs_L1S_D"]["belief_updates_D_KL"].append(
            np.mean(belief_updates_D_KL_level1_seller_defensive)
        )
        results["L1B_vs_L1S_D"]["policies_P"].append(P_level1_advanced.mean(axis=0))

        results["L2B_vs_L2S_D"]["mutual_info_I_r_a1"].append(mi_I_r_a1_level2_advanced)
        results["L2B_vs_L2S_D"]["belief_updates_D_KL"].append(
            np.mean(belief_updates_D_KL_level2_seller_defensive)
        )
        results["L2B_vs_L2S_D"]["policies_P"].append(P_level2_advanced.mean(axis=0))

        logger.info(
            f"[Systematic Analysis] Aggregated results for preference {pref_idx + 1}/{len(candidate_vector_r)}"
        )

    # Save results with notation
    results_serializable = {}
    for key, value in results.items():
        results_serializable[key] = {
            "mutual_info_I_r_a1": [float(x) for x in value["mutual_info_I_r_a1"]],
            "belief_updates_D_KL": [float(x) for x in value["belief_updates_D_KL"]],
            "policies_P": [policy.tolist() for policy in value["policies_P"]],
        }

    with open(DATA_DIR, "w") as f:
        json.dump(results_serializable, f, indent=2)

    return results, candidate_vector_r


def plot_results(results: Dict, candidate_vector_r: np.ndarray):
    """Plot results following notation"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot mutual information I(**r**, a₁)
    ax = axes[0, 0]
    for key in results.keys():
        ax.plot(
            candidate_vector_r,
            results[key]["mutual_info_I_r_a1"],
            label=key,
            linewidth=2,
        )
    ax.set_xlabel("Buyer Preference for Apple (**r**[apple])")
    ax.set_ylabel("Mutual Information I(**r**, a₁)")
    ax.set_title("Mutual Information: Preference Revelation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot belief updates D_KL
    ax = axes[0, 1]
    for key in results.keys():
        ax.plot(
            candidate_vector_r,
            results[key]["belief_updates_D_KL"],
            label=key,
            linewidth=2,
        )
    ax.set_xlabel("Buyer Preference for Apple (**r**[apple])")
    ax.set_ylabel("KL Divergence D_KL(p(**r**|a₁)||p(**r**))")
    ax.set_title("Seller Skepticism: Belief Update Strength")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot action probabilities P^{agent,t=1}
    ax = axes[1, 0]
    for key in results.keys():
        policies_P = np.array(results[key]["policies_P"])
        ax.plot(
            candidate_vector_r,
            policies_P[:, 0],
            label=f"{key.split('_vs_')[0]} P(Apple|t=1)",
            linewidth=2,
        )
    ax.set_xlabel("Buyer Preference for Apple (**r**[apple])")
    ax.set_ylabel("P^{b,t=1}(Choose Apple)")
    ax.set_title("Strategic Behavior: Action Probabilities")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot deception effectiveness (ΔI)
    ax = axes[1, 1]
    naive_mi_I_r_a1 = results["L0B_vs_L1S"]["mutual_info_I_r_a1"]
    for key in results.keys():
        deception_delta_I = np.array(naive_mi_I_r_a1) - np.array(
            results[key]["mutual_info_I_r_a1"]
        )
        ax.plot(
            candidate_vector_r,
            deception_delta_I,
            label=key,
            linewidth=2,
        )
    ax.set_xlabel("Buyer Preference for Apple (**r**[apple])")
    ax.set_ylabel("Deception Effectiveness (ΔI)")
    ax.set_title("Information Hiding: Deception vs Naive")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {PLOT_DIR}")


# Example usage
if __name__ == "__main__":
    logger.info("Running systematic analysis with generalized level-l agents...")

    # Run analysis with level hierarchy up to 2 and parallel processing
    results, candidate_vector_r = run_systematic_analysis(
        beta=DEFAULT_BETA,
        n_samples=N_SAMPLES,
        max_level=MAX_LEVEL,
        n_processes=None,  # None = CPU 수 - 1
    )

    logger.info("Plotting results...")
    # Plot results
    plot_results(results, candidate_vector_r)

    # Log summary statistics
    logger.info("\n=== Summary Statistics (Generalized Level-l Notation) ===")
    for key in results.keys():
        mi_I_r_a1_avg = np.mean(results[key]["mutual_info_I_r_a1"])
        belief_D_KL_avg = np.mean(results[key]["belief_updates_D_KL"])
        logger.info(f"{key}:")
        logger.info(f"  Average I(**r**, a₁): {mi_I_r_a1_avg:.3f}")
        logger.info(f"  Average D_KL(p(**r**|a₁)||p(**r**)): {belief_D_KL_avg:.3f}")

    # Demonstrate specific example with generalized agents
    logger.info("\n=== Example Interaction (Generalized Level-l) ===")
    vector_r = np.array([7, 3])  # Strong preference for apple
    vector_d = np.array([3, 7])  # Apple is closer

    env = BuyerSellerEnvironment()

    # Create agents using generalized hierarchy
    agents = create_agent_hierarchy(max_level=2, beta=2.0)
    buyer_l1 = agents["buyer_l1"]
    seller_l1 = agents["seller_l1"]

    state = env.reset(vector_r, vector_d)

    # Stage t=1
    buyer_choice_a1 = buyer_l1.act_t1(state)
    env.step({"buyer_choice_a1": buyer_choice_a1})

    # Stage t=2
    preference_grid_r, posterior_p_r_given_a = seller_l1.p_r_given_a(
        buyer_choice_a1, vector_d
    )
    vector_m = seller_l1.compute_m_star(preference_grid_r, posterior_p_r_given_a)
    env.step({"vector_m": vector_m})

    # Calculate metrics
    prior_p_r = np.ones_like(posterior_p_r_given_a) / len(posterior_p_r_given_a)
    belief_update_D_KL = InformationTheoryMetrics.update_belief(
        prior_p_r, posterior_p_r_given_a
    )

    logger.info(f"Buyer **r**: Apple={vector_r[0]:.1f}, Orange={vector_r[1]:.1f}")
    logger.info(f"Distance **d**: Apple={vector_d[0]:.1f}, Orange={vector_d[1]:.1f}")
    logger.info(f"Stage t=1 choice a₁: {'Apple' if buyer_choice_a1 == 0 else 'Orange'}")
    logger.info(f"Seller **m**: Apple={vector_m[0]:.2f}, Orange={vector_m[1]:.2f}")
    logger.info(f"Belief update D_KL: {belief_update_D_KL:.3f}")

    # Demonstrate extensibility
    logger.info("\n=== Agent Hierarchy Extensibility ===")
    logger.info(f"Current hierarchy: {list(agents.keys())}")
    logger.info(
        "Can easily extend to level-3, level-4, etc. by changing max_level parameter"
    )
