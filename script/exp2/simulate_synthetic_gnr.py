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
DEFAULT_N_PREFERENCE_POINTS = 101
DEFAULT_N_PRICE_POINTS = 81
EPSILON = 1e-10
EXPERIMENT_TIME = datetime.now().strftime("%Y%m%d_%H%M")
RESULT_DIR = f"./results/exp2/{EXPERIMENT_TIME}/"
PLOT_DIR = f"{RESULT_DIR}/analysis_plot.png"
DATA_DIR = f"{RESULT_DIR}/simulation_results.json"

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
    def normalize_vector_to_sum(
        values: np.ndarray, target_sum: float = DEFAULT_MAX_VALUE
    ) -> np.ndarray:
        """Apply constraint: sum(v) = target_sum"""
        return values * target_sum / values.sum()

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


class BayesianIRLMixin:
    """Bayesian inverse reinforcement learning functionality"""

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


class PriceOptimizationMixin:
    """Price optimization functionality"""

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


class StrategicPlanningMixin:
    """Strategic planning functionality for level-l buyers"""

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


class LevelLBuyer(BaseAgent, StrategicPlanningMixin):
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


class LevelLSeller(BaseAgent, BayesianIRLMixin, PriceOptimizationMixin):
    """Level-l seller agent"""

    def __init__(
        self,
        level_l: int,
        buyer_model: Optional[BuyerAgentProtocol] = None,
        beta: float = DEFAULT_BETA,
        n_preference_points: int = DEFAULT_N_PREFERENCE_POINTS,
    ):
        super().__init__(beta)
        self.level_l = level_l
        self.buyer_model = buyer_model
        self.n_preference_points = n_preference_points

    def p_r_given_a(
        self, observed_action_a1: int, vector_d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bayesian preference inference: p^s(r | a_1)"""
        if self.level_l == 0:
            # Level-0: uniform distribution
            preference_grid_r = NotationUtils.create_preference_grid_r(
                self.n_preference_points
            )
            uniform_posterior = NotationUtils.create_uniform_prior_p_r(
                len(preference_grid_r)
            )
            return preference_grid_r, uniform_posterior
        else:
            # Level-l: use buyer model for inference
            return self.p_r_given_a_impl(
                observed_action_a1, vector_d, self.buyer_model, self.n_preference_points
            )

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


class InformationTheoryMetrics:
    """Information theory metrics calculation"""

    @staticmethod
    def compute_I_r_a1(preference_values: np.ndarray, policies: np.ndarray) -> float:
        """Compute mutual information I(r, a_1)"""
        n_prefs, n_actions = policies.shape
        action_probs = np.mean(policies, axis=0)

        mi = 0
        for i in range(n_prefs):
            for a in range(n_actions):
                if policies[i, a] > EPSILON:
                    mi += (
                        (1 / n_prefs)
                        * policies[i, a]
                        * np.log(policies[i, a] / action_probs[a])
                    )

        return mi

    @staticmethod
    def D_KL(p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence D_KL(p || q)"""
        p = p + EPSILON
        q = q + EPSILON
        p = p / np.sum(p)
        q = q / np.sum(q)
        return np.sum(p * np.log(p / q))

    @staticmethod
    def update_belief(
        prior_p_r: np.ndarray, posterior_p_r_given_a: np.ndarray
    ) -> float:
        """Measure belief update: D_KL(p(r | a_1) || p(r))"""
        return InformationTheoryMetrics.D_KL(posterior_p_r_given_a, prior_p_r)

    @staticmethod
    def policy_mismatch(assumed_policy: np.ndarray, actual_policy: np.ndarray) -> float:
        """Measure policy mismatch: D_KL(actual || assumed)"""
        return InformationTheoryMetrics.D_KL(actual_policy, assumed_policy)


def create_agent_hierarchy(max_level: int, beta: float = DEFAULT_BETA) -> Dict:
    """Create agent hierarchy from Level-0 to Level-max_level"""
    agents = {}

    # Base case: Level-0 buyer
    agents["buyer_l0"] = LevelLBuyer(level_l=0, seller_model=None, beta=beta)

    # Build hierarchy recursively
    for level in range(1, max_level + 1):
        # Level-l seller uses Level-(l-1) buyer model
        buyer_model_prev = agents[f"buyer_l{level-1}"]
        agents[f"seller_l{level}"] = LevelLSeller(
            level_l=level, buyer_model=buyer_model_prev, beta=beta
        )

        # Level-l buyer uses Level-l seller model
        seller_model = agents[f"seller_l{level}"]
        agents[f"buyer_l{level}"] = LevelLBuyer(
            level_l=level, seller_model=seller_model, beta=beta
        )

    return agents


def compute_single_combination(args):
    """Compute metrics for a single (r_apple, d_apple) combination"""
    r_apple, d_apple, beta, max_level = args

    agents = create_agent_hierarchy(max_level=max_level, beta=beta)

    vector_r = np.array([r_apple, DEFAULT_MAX_VALUE - r_apple])
    vector_d = np.array([d_apple, DEFAULT_MAX_VALUE - d_apple])

    # Get policies from agents
    P_level0_current = agents["buyer_l0"].get_P_t1(vector_d, vector_r)
    P_level1_current = agents["buyer_l1"].get_P_t1(vector_d, vector_r)
    P_level2_current = agents["buyer_l2"].get_P_t1(vector_d, vector_r)

    # Determine closer item
    a_1_buyer = 0 if d_apple < (DEFAULT_MAX_VALUE - d_apple) else 1

    # Calculate belief updates
    _, posterior_p_r_given_a_level1 = agents["seller_l1"].p_r_given_a(
        a_1_buyer, vector_d
    )
    prior_p_r = NotationUtils.create_uniform_prior_p_r(
        len(posterior_p_r_given_a_level1)
    )
    belief_update_level1 = InformationTheoryMetrics.update_belief(
        prior_p_r, posterior_p_r_given_a_level1
    )

    _, posterior_p_r_given_a_level2 = agents["seller_l2"].p_r_given_a(
        a_1_buyer, vector_d
    )
    belief_update_level2 = InformationTheoryMetrics.update_belief(
        prior_p_r, posterior_p_r_given_a_level2
    )

    # Calculate policy mismatches
    policy_mismatch_l0_vs_l1 = InformationTheoryMetrics.policy_mismatch(
        P_level0_current, P_level1_current
    )
    policy_mismatch_l1_vs_l2 = InformationTheoryMetrics.policy_mismatch(
        P_level1_current, P_level2_current
    )

    return {
        "r_apple": r_apple,
        "d_apple": d_apple,
        "policies": {
            "P_level0": P_level0_current,
            "P_level1": P_level1_current,
            "P_level2": P_level2_current,
        },
        "belief_updates": {
            "level1": belief_update_level1,
            "level2": belief_update_level2,
        },
        "policy_mismatches": {
            "l0_vs_l1": policy_mismatch_l0_vs_l1,
            "l1_vs_l2": policy_mismatch_l1_vs_l2,
        },
        "closer_item": a_1_buyer,
    }


def run_systematic_analysis(
    beta: float = DEFAULT_BETA,
    n_samples: int = 20,
    max_level: int = 2,
    n_processes: int = None,
):
    """Run systematic analysis across preference/distance combinations"""

    if n_processes is None:
        n_processes = mp.cpu_count() - 1

    candidate_vector_r = np.linspace(1, 9, n_samples)
    candidate_vector_d = np.linspace(1, 9, n_samples)

    # Store results
    results = {
        "L0B_vs_L1S": {
            "mutual_info_I_r_a1": [],
            "belief_updates_D_KL": [],
            "policies_P": [],
            "seller_error": [],
        },
        "L1B_vs_L1S": {
            "mutual_info_I_r_a1": [],
            "belief_updates_D_KL": [],
            "policies_P": [],
            "seller_error": [],
        },
        "L1B_vs_L2S": {
            "mutual_info_I_r_a1": [],
            "belief_updates_D_KL": [],
            "policies_P": [],
            "seller_error": [],
        },
    }

    logger.info(f"Starting parallel computation with {n_processes} processes")

    all_combinations = [
        (r_apple, d_apple, beta, max_level)
        for r_apple in candidate_vector_r
        for d_apple in candidate_vector_d
    ]

    with mp.Pool(processes=n_processes) as pool:
        parallel_results = list(
            tqdm(
                pool.imap(compute_single_combination, all_combinations),
                total=len(all_combinations),
                desc="Processing combinations",
            )
        )

    logger.info("Aggregating results...")

    # Aggregate results by distance
    for dist_idx, d_apple in tqdm(
        enumerate(candidate_vector_d),
        desc="Aggregating by distance",
        total=len(candidate_vector_d),
    ):
        current_d_results = [
            r for r in parallel_results if np.isclose(r["d_apple"], d_apple)
        ]
        current_d_results.sort(key=lambda x: x["r_apple"])

        # Collect policies
        P_level0_all_prefs = np.array(
            [r["policies"]["P_level0"] for r in current_d_results]
        )
        P_level1_all_prefs = np.array(
            [r["policies"]["P_level1"] for r in current_d_results]
        )
        P_level2_all_prefs = np.array(
            [r["policies"]["P_level2"] for r in current_d_results]
        )

        # Calculate mutual information
        preference_values = np.array([r["r_apple"] for r in current_d_results])

        mi_I_r_a1_level0 = InformationTheoryMetrics.compute_I_r_a1(
            preference_values, P_level0_all_prefs
        )
        mi_I_r_a1_level1 = InformationTheoryMetrics.compute_I_r_a1(
            preference_values, P_level1_all_prefs
        )
        mi_I_r_a1_level2 = InformationTheoryMetrics.compute_I_r_a1(
            preference_values, P_level2_all_prefs
        )

        # Get averages
        belief_updates_level1 = np.mean(
            [r["belief_updates"]["level1"] for r in current_d_results]
        )
        belief_updates_level2 = np.mean(
            [r["belief_updates"]["level2"] for r in current_d_results]
        )

        policy_mismatch_l0_vs_l1 = np.mean(
            [r["policy_mismatches"]["l0_vs_l1"] for r in current_d_results]
        )
        policy_mismatch_l1_vs_l2 = np.mean(
            [r["policy_mismatches"]["l1_vs_l2"] for r in current_d_results]
        )

        # Store results
        results["L0B_vs_L1S"]["mutual_info_I_r_a1"].append(mi_I_r_a1_level0)
        results["L0B_vs_L1S"]["belief_updates_D_KL"].append(belief_updates_level1)
        results["L0B_vs_L1S"]["policies_P"].append(P_level0_all_prefs.mean(axis=0))
        results["L0B_vs_L1S"]["seller_error"].append(0.0)

        results["L1B_vs_L1S"]["mutual_info_I_r_a1"].append(mi_I_r_a1_level1)
        results["L1B_vs_L1S"]["belief_updates_D_KL"].append(belief_updates_level1)
        results["L1B_vs_L1S"]["policies_P"].append(P_level1_all_prefs.mean(axis=0))
        results["L1B_vs_L1S"]["seller_error"].append(policy_mismatch_l0_vs_l1)

        results["L1B_vs_L2S"]["mutual_info_I_r_a1"].append(mi_I_r_a1_level2)
        results["L1B_vs_L2S"]["belief_updates_D_KL"].append(belief_updates_level2)
        results["L1B_vs_L2S"]["policies_P"].append(P_level2_all_prefs.mean(axis=0))
        results["L1B_vs_L2S"]["seller_error"].append(policy_mismatch_l1_vs_l2)

    # Save results
    results_serializable = {}
    for key, value in results.items():
        results_serializable[key] = {
            "mutual_info_I_r_a1": [float(x) for x in value["mutual_info_I_r_a1"]],
            "belief_updates_D_KL": [float(x) for x in value["belief_updates_D_KL"]],
            "policies_P": [policy.tolist() for policy in value["policies_P"]],
            "seller_error": [float(x) for x in value["seller_error"]],
        }

    with open(DATA_DIR, "w") as f:
        json.dump(results_serializable, f, indent=2)

    return results, candidate_vector_d


def plot_results(results: Dict, candidate_vector_d: np.ndarray):
    """Plot results - Figure 3F, 3G, 3H"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Figure 3F: Mutual Information I(r, a₁)
    ax = axes[0]
    ax.plot(
        candidate_vector_d,
        results["L0B_vs_L1S"]["mutual_info_I_r_a1"],
        label="Lv0 buyer",
        linewidth=2,
        color="darkgreen",
    )
    ax.plot(
        candidate_vector_d,
        results["L1B_vs_L1S"]["mutual_info_I_r_a1"],
        label="Lv1 buyer",
        linewidth=2,
        color="orange",
    )
    ax.plot(
        candidate_vector_d,
        results["L1B_vs_L2S"]["mutual_info_I_r_a1"],
        label="Lv1 buyer (vs Lv2 seller)",
        linewidth=2,
        color="lightgreen",
    )
    ax.set_xlabel("Distance to Apple")
    ax.set_ylabel("Mutual Information I(r, a₁)")
    ax.set_title("Figure 3F: Preference-Policy MI")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Figure 3G: Seller Skepticism
    ax = axes[1]
    ax.plot(
        candidate_vector_d,
        results["L0B_vs_L1S"]["belief_updates_D_KL"],
        label="Lv1 seller",
        linewidth=2,
        color="darkblue",
    )
    ax.plot(
        candidate_vector_d,
        results["L1B_vs_L2S"]["belief_updates_D_KL"],
        label="Lv2 seller",
        linewidth=2,
        color="lightblue",
    )
    ax.set_xlabel("Distance to Apple")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Figure 3G: Seller Skepticism")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Figure 3H: Seller Error
    ax = axes[2]
    ax.plot(
        candidate_vector_d,
        results["L1B_vs_L1S"]["seller_error"],
        label="Lv1 seller error",
        linewidth=2,
        color="darkgreen",
    )
    ax.plot(
        candidate_vector_d,
        results["L1B_vs_L2S"]["seller_error"],
        label="Lv2 seller error",
        linewidth=2,
        color="lightgreen",
    )
    ax.set_xlabel("Distance to Apple")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Figure 3H: Seller Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {PLOT_DIR}")


if __name__ == "__main__":
    logger.info("Running systematic analysis...")

    # Run analysis
    results, candidate_vector_d = run_systematic_analysis(
        beta=DEFAULT_BETA,
        n_samples=N_SAMPLES,
        max_level=MAX_LEVEL,
        n_processes=None,
    )

    logger.info("Plotting results...")
    plot_results(results, candidate_vector_d)

    # Log summary statistics
    logger.info("\n=== Summary Statistics ===")
    for key in results.keys():
        mi_I_r_a1_avg = np.mean(results[key]["mutual_info_I_r_a1"])
        belief_D_KL_avg = np.mean(results[key]["belief_updates_D_KL"])
        seller_error_avg = np.mean(results[key]["seller_error"])
        logger.info(f"{key}:")
        logger.info(f"  Average I(r, a₁): {mi_I_r_a1_avg:.3f}")
        logger.info(f"  Average D_KL(p(r|a₁)||p(r)): {belief_D_KL_avg:.3f}")
        logger.info(f"  Average Seller Error: {seller_error_avg:.3f}")
