import os
import json
from datetime import datetime
from typing import Dict, Tuple, Protocol
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm, trange

import numpy as np
from scipy.special import softmax

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_BETA = 2.0
DEFAULT_N_ITEMS = 2
DEFAULT_MAX_VALUE = 10.0
DEFAULT_N_PREFERENCE_POINTS = 101
DEFAULT_N_PRICE_POINTS = 51
EPSILON = 1e-10
EXPERIMENT_TIME = datetime.now().strftime("%Y%m%d_%H%M")
RESULT_DIR = f"./results/exp2/{EXPERIMENT_TIME}/"
PLOT_DIR = f"{RESULT_DIR}/analysis_plot.png"
DATA_DIR = f"{RESULT_DIR}/simulation_results.json"

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


class BuyerProtocol(Protocol):
    """Protocol for buyer agents"""

    def get_policy_stage1(
        self, distances: np.ndarray, preferences: np.ndarray
    ) -> np.ndarray: ...


class Utils:
    """Utility functions for common operations"""

    @staticmethod
    def normalize_to_sum(
        values: np.ndarray, target_sum: float = DEFAULT_MAX_VALUE
    ) -> np.ndarray:
        """Normalize array to sum to target value"""
        return values * target_sum / values.sum()

    @staticmethod
    def create_preference_grid(
        n_points: int = DEFAULT_N_PREFERENCE_POINTS,
    ) -> np.ndarray:
        """Create preference grid for Bayesian inference"""
        return np.linspace(0, DEFAULT_MAX_VALUE, n_points)

    @staticmethod
    def create_uniform_prior(size: int) -> np.ndarray:
        """Create uniform prior distribution"""
        return np.ones(size) / size

    @staticmethod
    def safe_softmax(q_values: np.ndarray, beta: float = DEFAULT_BETA) -> np.ndarray:
        """Numerically stable softmax computation"""
        q_values = np.array(q_values)
        if len(q_values) == 2:
            diff = beta * (q_values[0] - q_values[1])
            prob_0 = 1 / (1 + np.exp(-diff))
            return np.array([prob_0, 1 - prob_0])
        else:
            return softmax(beta * q_values)


class BayesianIRLMixin:
    """Mixin for Bayesian Inverse Reinforcement Learning functionality"""

    def perform_bayesian_irl(
        self,
        observed_action: int,
        distances: np.ndarray,
        buyer_model: BuyerProtocol,
        n_preference_points: int = DEFAULT_N_PREFERENCE_POINTS,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """General Bayesian IRL implementation"""
        preference_grid = Utils.create_preference_grid(n_preference_points)
        posterior = np.zeros_like(preference_grid)
        prior = Utils.create_uniform_prior(len(preference_grid))

        for i, pref_apple in enumerate(preference_grid):
            pref_orange = DEFAULT_MAX_VALUE - pref_apple
            preferences = np.array([pref_apple, pref_orange])

            # Get policy from buyer model
            policy = buyer_model.get_policy_stage1(distances, preferences)
            likelihood = policy[observed_action]
            posterior[i] = likelihood * prior[i]

        # Normalize posterior
        posterior = posterior / np.sum(posterior)
        return preference_grid, posterior


class PriceOptimizationMixin:
    """Mixin for price optimization functionality"""

    def optimize_prices_basic(
        self,
        preference_grid: np.ndarray,
        posterior: np.ndarray,
        beta: float = DEFAULT_BETA,
        n_price_points: int = DEFAULT_N_PRICE_POINTS,
    ) -> np.ndarray:
        """Basic price optimization with constraint that prices sum to max value"""
        best_prices = np.array([DEFAULT_MAX_VALUE / 2, DEFAULT_MAX_VALUE / 2])
        best_expected_revenue = 0

        for price_apple in np.linspace(0, DEFAULT_MAX_VALUE, n_price_points):
            price_orange = DEFAULT_MAX_VALUE - price_apple
            test_prices = np.array([price_apple, price_orange])

            expected_revenue = self._calculate_expected_revenue(
                preference_grid, posterior, test_prices, beta
            )

            if expected_revenue > best_expected_revenue:
                best_expected_revenue = expected_revenue
                best_prices = test_prices

        return best_prices

    def _calculate_expected_revenue(
        self,
        preference_grid: np.ndarray,
        posterior: np.ndarray,
        prices: np.ndarray,
        beta: float,
    ) -> float:
        """Calculate expected revenue for given prices"""
        expected_revenue = 0
        for i, pref_apple in enumerate(preference_grid):
            pref_orange = DEFAULT_MAX_VALUE - pref_apple
            preferences = np.array([pref_apple, pref_orange])

            q_values = preferences - prices
            purchase_probs = Utils.safe_softmax(q_values, beta)
            revenue = np.sum(purchase_probs * prices)
            expected_revenue += posterior[i] * revenue

        return expected_revenue


class StrategicBuyerMixin:
    """Mixin for strategic buyer functionality"""

    def compute_strategic_q_values(
        self,
        distances: np.ndarray,
        preferences: np.ndarray,
        seller_model,
        beta: float = DEFAULT_BETA,
    ) -> np.ndarray:
        """Compute Q-values considering seller's response"""
        q_values = np.zeros(DEFAULT_N_ITEMS)

        for action in range(DEFAULT_N_ITEMS):
            # Immediate utility from stage 1
            immediate_utility = preferences[action] - distances[action]

            # Simulate seller's response
            preference_grid, posterior = seller_model.bayesian_irl(action, distances)
            prices = seller_model.set_optimal_prices(preference_grid, posterior)

            # Expected utility from stage 3
            stage3_q_values = preferences - prices
            stage3_probs = Utils.safe_softmax(stage3_q_values, beta)
            expected_stage3_utility = np.sum(stage3_probs * stage3_q_values)

            q_values[action] = immediate_utility + expected_stage3_utility

        return q_values


class BaseAgent:
    """Base agent class with common functionality"""

    def __init__(self, beta: float = DEFAULT_BETA):
        self.beta = beta

    def softmax_policy(self, q_values: np.ndarray) -> np.ndarray:
        """Convert Q-values to action probabilities using softmax"""
        return Utils.safe_softmax(q_values, self.beta)

    def select_action(self, q_values: np.ndarray) -> int:
        """Select action based on softmax policy"""
        probs = self.softmax_policy(q_values)
        return np.random.choice(len(q_values), p=probs)


class BuyerSellerEnvironment:
    """3-stage buyer-seller interaction environment"""

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

    def reset(self, buyer_preferences: np.ndarray, distances: np.ndarray):
        """Reset environment with buyer preferences and item distances"""
        assert len(buyer_preferences) == self.n_items
        assert len(distances) == self.n_items

        self.buyer_preferences = Utils.normalize_to_sum(buyer_preferences)
        self.distances = Utils.normalize_to_sum(distances)

        self.stage = 1
        self.buyer_choice_stage1 = None
        self.seller_prices = None

        return self.get_state()

    def get_state(self):
        """Get current environment state"""
        return {
            "stage": self.stage,
            "distances": self.distances,
            "buyer_preferences": self.buyer_preferences,
            "buyer_choice_stage1": self.buyer_choice_stage1,
            "seller_prices": self.seller_prices,
        }

    def step(self, action: Dict):
        """Execute action and move to next stage"""
        if self.stage == 1:
            self.buyer_choice_stage1 = action["buyer_choice"]
            self.stage = 2

        elif self.stage == 2:
            self.seller_prices = action["seller_prices"]
            self.stage = 3

        elif self.stage == 3:
            buyer_choice_stage3 = action["buyer_choice"]

            # Calculate utilities
            buyer_utility_stage1 = (
                self.buyer_preferences[self.buyer_choice_stage1]
                - self.distances[self.buyer_choice_stage1]
            )
            buyer_utility_stage3 = (
                self.buyer_preferences[buyer_choice_stage3]
                - self.seller_prices[buyer_choice_stage3]
            )
            buyer_total_utility = buyer_utility_stage1 + buyer_utility_stage3

            seller_utility = self.seller_prices[buyer_choice_stage3]

            self.stage = 4

            return {
                "buyer_utility": buyer_total_utility,
                "seller_utility": seller_utility,
                "done": True,
            }

        return {"done": False}


class ToMNeg1Buyer(BaseAgent):
    """ToM(-1) buyer: naive, maximizes utility at each stage independently"""

    def act_stage1(self, state: Dict) -> int:
        """Choose item in stage 1 based on immediate utility"""
        q_values = state["buyer_preferences"] - state["distances"]
        return self.select_action(q_values)

    def act_stage3(self, state: Dict) -> int:
        """Choose item in stage 3 based on immediate utility"""
        q_values = state["buyer_preferences"] - state["seller_prices"]
        return self.select_action(q_values)

    def get_policy_stage1(
        self, distances: np.ndarray, preferences: np.ndarray
    ) -> np.ndarray:
        """Get policy probabilities for stage 1"""
        q_values = preferences - distances
        return self.softmax_policy(q_values)


class ToM0Seller(BaseAgent, BayesianIRLMixin, PriceOptimizationMixin):
    """ToM(0) seller: performs Bayesian IRL on ToM(-1) buyer"""

    def __init__(
        self,
        beta: float = DEFAULT_BETA,
        n_preference_points: int = DEFAULT_N_PREFERENCE_POINTS,
    ):
        super().__init__(beta)
        self.n_preference_points = n_preference_points
        self.tom_neg1_buyer = ToMNeg1Buyer(beta)

    def bayesian_irl(
        self, observed_action: int, distances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform Bayesian inverse reinforcement learning"""
        return self.perform_bayesian_irl(
            observed_action, distances, self.tom_neg1_buyer, self.n_preference_points
        )

    def set_optimal_prices(
        self, preference_grid: np.ndarray, posterior: np.ndarray
    ) -> np.ndarray:
        """Set prices based on posterior beliefs about buyer preferences"""
        return self.optimize_prices_basic(preference_grid, posterior, self.beta)


class ToM1Buyer(BaseAgent, StrategicBuyerMixin):
    """ToM(1) buyer: strategic, plans through ToM(0) seller's inference"""

    def __init__(self, beta: float = DEFAULT_BETA):
        super().__init__(beta)
        self.tom0_seller = ToM0Seller(beta)

    def act_stage1(self, state: Dict) -> int:
        """Choose item strategically, considering seller's inference"""
        q_values = self.compute_strategic_q_values(
            state["distances"], state["buyer_preferences"], self.tom0_seller, self.beta
        )
        return self.select_action(q_values)

    def act_stage3(self, state: Dict) -> int:
        """Same as ToM(-1) for stage 3"""
        return ToMNeg1Buyer(self.beta).act_stage3(state)

    def get_policy_stage1(
        self, distances: np.ndarray, preferences: np.ndarray
    ) -> np.ndarray:
        """Get strategic policy for stage 1"""
        q_values = self.compute_strategic_q_values(
            distances, preferences, self.tom0_seller, self.beta
        )
        return self.softmax_policy(q_values)


class ToM2Seller(BaseAgent, BayesianIRLMixin):
    """ToM(2) seller: skeptical, performs Bayesian IRL on ToM(1) buyer"""

    def __init__(
        self,
        beta: float = DEFAULT_BETA,
        n_preference_points: int = DEFAULT_N_PREFERENCE_POINTS,
    ):
        super().__init__(beta)
        self.n_preference_points = n_preference_points
        self.tom1_buyer = ToM1Buyer(beta)

    def bayesian_irl(
        self, observed_action: int, distances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform Bayesian IRL assuming strategic ToM(1) buyer"""
        return self.perform_bayesian_irl(
            observed_action, distances, self.tom1_buyer, self.n_preference_points
        )

    def set_optimal_prices(
        self, preference_grid: np.ndarray, posterior: np.ndarray
    ) -> np.ndarray:
        """Set prices defensively against strategic buyer"""
        prices = np.zeros(DEFAULT_N_ITEMS)
        n_price_points = DEFAULT_N_PRICE_POINTS

        for item in range(DEFAULT_N_ITEMS):
            best_price = 0
            best_expected_revenue = 0

            for price in np.linspace(0, DEFAULT_MAX_VALUE, n_price_points):
                expected_revenue = 0

                for i, pref_apple in enumerate(preference_grid):
                    pref_orange = DEFAULT_MAX_VALUE - pref_apple
                    preferences = np.array([pref_apple, pref_orange])

                    # Test prices
                    test_prices = (
                        np.array([price, DEFAULT_MAX_VALUE - price])
                        if item == 0
                        else np.array([DEFAULT_MAX_VALUE - price, price])
                    )

                    # Use softmax for stage 3 choice
                    q_values = preferences - test_prices
                    purchase_probs = Utils.safe_softmax(q_values, self.beta)

                    expected_revenue += posterior[i] * purchase_probs[item] * price

                if expected_revenue > best_expected_revenue:
                    best_expected_revenue = expected_revenue
                    best_price = price

            prices[item] = best_price

        # Ensure prices sum to max value
        return Utils.normalize_to_sum(prices)


class ToM3Buyer(BaseAgent, StrategicBuyerMixin):
    """ToM(3) buyer: plans through ToM(2) seller's defensive inference"""

    def __init__(self, beta: float = DEFAULT_BETA):
        super().__init__(beta)
        self.tom2_seller = ToM2Seller(beta)

    def act_stage1(self, state: Dict) -> int:
        """Choose item strategically against skeptical seller"""
        q_values = self.compute_strategic_q_values(
            state["distances"], state["buyer_preferences"], self.tom2_seller, self.beta
        )
        return self.select_action(q_values)

    def act_stage3(self, state: Dict) -> int:
        """Same as ToM(-1) for stage 3"""
        return ToMNeg1Buyer(self.beta).act_stage3(state)

    def get_policy_stage1(
        self, distances: np.ndarray, preferences: np.ndarray
    ) -> np.ndarray:
        """Get strategic policy for stage 1"""
        q_values = self.compute_strategic_q_values(
            distances, preferences, self.tom2_seller, self.beta
        )
        return self.softmax_policy(q_values)


class InformationTheoryMetrics:
    """Calculate information-theoretic metrics for analysis"""

    @staticmethod
    def mutual_information_continuous(
        preference_values: np.ndarray, policies: np.ndarray
    ) -> float:
        """Calculate mutual information between preferences and actions"""
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
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Calculate KL divergence D_KL(p || q)"""
        # Avoid log(0) by adding small epsilon
        p = p + EPSILON
        q = q + EPSILON
        p = p / np.sum(p)
        q = q / np.sum(q)
        return np.sum(p * np.log(p / q))

    @staticmethod
    def belief_update_strength(prior: np.ndarray, posterior: np.ndarray) -> float:
        """Measure strength of belief update using KL divergence"""
        return InformationTheoryMetrics.kl_divergence(posterior, prior)


def run_systematic_analysis(beta: float = DEFAULT_BETA, n_samples: int = 20):
    """Run systematic analysis as in the paper"""

    # Create parameter grids
    preference_values = np.linspace(1, 9, n_samples)
    distance_values = np.linspace(1, 9, n_samples)

    # Store results
    results = {
        "ToM(-1)_vs_ToM(0)": {"mutual_info": [], "belief_updates": [], "policies": []},
        "ToM(1)_vs_ToM(0)": {"mutual_info": [], "belief_updates": [], "policies": []},
        "ToM(1)_vs_ToM(2)": {"mutual_info": [], "belief_updates": [], "policies": []},
        "ToM(3)_vs_ToM(2)": {"mutual_info": [], "belief_updates": [], "policies": []},
    }

    # Create agents
    agents = {
        "ToM(-1)": ToMNeg1Buyer(beta),
        "ToM(1)": ToM1Buyer(beta),
        "ToM(3)": ToM3Buyer(beta),
        "ToM(0)": ToM0Seller(beta),
        "ToM(2)": ToM2Seller(beta),
    }

    for pref_idx, pref in tqdm(
        enumerate(preference_values),
        desc="Processing preferences",
        total=len(preference_values),
    ):
        preferences = np.array([pref, DEFAULT_MAX_VALUE - pref])

        # Calculate policies for different distances
        policies_tom_neg1 = []
        policies_tom1 = []
        policies_tom3 = []

        belief_updates_tom0 = []
        belief_updates_tom2 = []

        for dist_idx in trange(len(distance_values), desc="Processing distances"):
            dist = distance_values[dist_idx]
            distances = np.array([dist, DEFAULT_MAX_VALUE - dist])

            # Get policies
            policy_neg1 = agents["ToM(-1)"].get_policy_stage1(distances, preferences)
            policy_1 = agents["ToM(1)"].get_policy_stage1(distances, preferences)
            policy_3 = agents["ToM(3)"].get_policy_stage1(distances, preferences)

            policies_tom_neg1.append(policy_neg1)
            policies_tom1.append(policy_1)
            policies_tom3.append(policy_3)

            # Calculate belief updates for both actions
            for action in range(DEFAULT_N_ITEMS):
                # ToM(0) seller's belief update
                _, posterior_tom0 = agents["ToM(0)"].bayesian_irl(action, distances)
                prior = Utils.create_uniform_prior(len(posterior_tom0))
                belief_update_tom0 = InformationTheoryMetrics.belief_update_strength(
                    prior, posterior_tom0
                )
                belief_updates_tom0.append(belief_update_tom0)

                # ToM(2) seller's belief update
                _, posterior_tom2 = agents["ToM(2)"].bayesian_irl(action, distances)
                belief_update_tom2 = InformationTheoryMetrics.belief_update_strength(
                    prior, posterior_tom2
                )
                belief_updates_tom2.append(belief_update_tom2)

        # Calculate mutual information
        policies_tom_neg1 = np.array(policies_tom_neg1)
        policies_tom1 = np.array(policies_tom1)
        policies_tom3 = np.array(policies_tom3)

        mi_neg1 = InformationTheoryMetrics.mutual_information_continuous(
            np.array([pref]), policies_tom_neg1
        )
        mi_1 = InformationTheoryMetrics.mutual_information_continuous(
            np.array([pref]), policies_tom1
        )
        mi_3 = InformationTheoryMetrics.mutual_information_continuous(
            np.array([pref]), policies_tom3
        )

        # Store results
        results["ToM(-1)_vs_ToM(0)"]["mutual_info"].append(mi_neg1)
        results["ToM(-1)_vs_ToM(0)"]["belief_updates"].append(
            np.mean(belief_updates_tom0)
        )
        results["ToM(-1)_vs_ToM(0)"]["policies"].append(policies_tom_neg1.mean(axis=0))

        results["ToM(1)_vs_ToM(0)"]["mutual_info"].append(mi_1)
        results["ToM(1)_vs_ToM(0)"]["belief_updates"].append(
            np.mean(belief_updates_tom0)
        )
        results["ToM(1)_vs_ToM(0)"]["policies"].append(policies_tom1.mean(axis=0))

        results["ToM(1)_vs_ToM(2)"]["mutual_info"].append(mi_1)
        results["ToM(1)_vs_ToM(2)"]["belief_updates"].append(
            np.mean(belief_updates_tom2)
        )
        results["ToM(1)_vs_ToM(2)"]["policies"].append(policies_tom1.mean(axis=0))

        results["ToM(3)_vs_ToM(2)"]["mutual_info"].append(mi_3)
        results["ToM(3)_vs_ToM(2)"]["belief_updates"].append(
            np.mean(belief_updates_tom2)
        )
        results["ToM(3)_vs_ToM(2)"]["policies"].append(policies_tom3.mean(axis=0))

        logger.info(
            f"[Systematic Analysis] Completed {pref_idx + 1}/{len(preference_values)}"
        )

    # save result data in json file
    with open(DATA_DIR, "w") as f:
        json.dump(results, f)

    return results, preference_values


def plot_results(results: Dict, preference_values: np.ndarray):
    """Plot results as in the paper"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot mutual information
    ax = axes[0, 0]
    for key in results.keys():
        ax.plot(
            preference_values,
            results[key]["mutual_info"],
            label=key.replace("_vs_", " vs "),
            linewidth=2,
        )
    ax.set_xlabel("Buyer Preference for Apple")
    ax.set_ylabel("Mutual Information I(r, a₁)")
    ax.set_title("Mutual Information: Preference Revelation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot belief updates
    ax = axes[0, 1]
    for key in results.keys():
        ax.plot(
            preference_values,
            results[key]["belief_updates"],
            label=key.replace("_vs_", " vs "),
            linewidth=2,
        )
    ax.set_xlabel("Buyer Preference for Apple")
    ax.set_ylabel("KL Divergence D_KL(p(r|a₁)||p(r))")
    ax.set_title("Seller Skepticism: Belief Update Strength")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot action probabilities
    ax = axes[1, 0]
    for key in results.keys():
        policies = np.array(results[key]["policies"])
        ax.plot(
            preference_values,
            policies[:, 0],
            label=f"{key.split('_vs_')[0]} (Apple)",
            linewidth=2,
        )
    ax.set_xlabel("Buyer Preference for Apple")
    ax.set_ylabel("P(Choose Apple in Stage 1)")
    ax.set_title("Strategic Behavior: Action Probabilities")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot deception effectiveness (difference from naive)
    ax = axes[1, 1]
    naive_mi = results["ToM(-1)_vs_ToM(0)"]["mutual_info"]
    for key in results.keys():
        if "ToM(1)" in key or "ToM(3)" in key:
            deception = np.array(naive_mi) - np.array(results[key]["mutual_info"])
            ax.plot(
                preference_values,
                deception,
                label=key.replace("_vs_", " vs "),
                linewidth=2,
            )
    ax.set_xlabel("Buyer Preference for Apple")
    ax.set_ylabel("Deception Effectiveness (ΔI)")
    ax.set_title("Information Hiding: Deception vs Naive")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {PLOT_DIR}")


# Example usage
if __name__ == "__main__":
    logger.info("Running systematic analysis to reproduce paper results...")

    # Run analysis
    results, preference_values = run_systematic_analysis(beta=2.0, n_samples=15)

    logger.info("Plotting results...")
    # Plot results
    plot_results(results, preference_values)

    # Log summary statistics
    logger.info("\n=== Summary Statistics ===")
    for key in results.keys():
        mi_avg = np.mean(results[key]["mutual_info"])
        belief_avg = np.mean(results[key]["belief_updates"])
        logger.info(f"{key}:")
        logger.info(f"  Average Mutual Information: {mi_avg:.3f}")
        logger.info(f"  Average Belief Update: {belief_avg:.3f}")

    # Demonstrate specific example
    logger.info("\n=== Example Interaction ===")
    preferences = np.array([7, 3])  # Strong preference for apple
    distances = np.array([3, 7])  # Apple is closer

    env = BuyerSellerEnvironment()

    # ToM(1) vs ToM(0)
    buyer = ToM1Buyer(beta=2.0)
    seller = ToM0Seller(beta=2.0)

    state = env.reset(preferences, distances)

    # Stage 1
    buyer_choice = buyer.act_stage1(state)
    env.step({"buyer_choice": buyer_choice})

    # Stage 2
    preference_grid, posterior = seller.bayesian_irl(buyer_choice, distances)
    prices = seller.set_optimal_prices(preference_grid, posterior)
    env.step({"seller_prices": prices})

    # Calculate metrics
    prior = np.ones_like(posterior) / len(posterior)
    belief_update = InformationTheoryMetrics.belief_update_strength(prior, posterior)

    logger.info(
        f"Buyer preferences: Apple={preferences[0]:.1f}, Orange={preferences[1]:.1f}"
    )
    logger.info(f"Distances: Apple={distances[0]:.1f}, Orange={distances[1]:.1f}")
    logger.info(f"Stage 1 choice: {'Apple' if buyer_choice == 0 else 'Orange'}")
    logger.info(f"Seller prices: Apple={prices[0]:.2f}, Orange={prices[1]:.2f}")
    logger.info(f"Belief update strength: {belief_update:.3f}")
    logger.info(f"Belief update strength: {belief_update:.3f}")
