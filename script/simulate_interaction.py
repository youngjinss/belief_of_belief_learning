import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from typing import Dict, Tuple, Optional, List
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import quad

class BuyerSellerEnvironment:
    """3-stage buyer-seller interaction environment"""
    
    def __init__(self, n_items: int = 2, max_distance: float = 10.0, max_price: float = 10.0):
        self.n_items = n_items
        self.max_distance = max_distance
        self.max_price = max_price
        self.items = ['apple', 'orange']
        
    def reset(self, buyer_preferences: np.ndarray, distances: np.ndarray):
        """Reset environment with buyer preferences and item distances"""
        assert len(buyer_preferences) == self.n_items
        assert len(distances) == self.n_items
        
        # Normalize to sum to 10 as in the paper
        self.buyer_preferences = buyer_preferences * 10 / buyer_preferences.sum()
        self.distances = distances * 10 / distances.sum()
        
        self.stage = 1
        self.buyer_choice_stage1 = None
        self.seller_prices = None
        
        return self.get_state()
    
    def get_state(self):
        """Get current environment state"""
        return {
            'stage': self.stage,
            'distances': self.distances,
            'buyer_preferences': self.buyer_preferences,
            'buyer_choice_stage1': self.buyer_choice_stage1,
            'seller_prices': self.seller_prices
        }
    
    def step(self, action: Dict):
        """Execute action and move to next stage"""
        if self.stage == 1:
            self.buyer_choice_stage1 = action['buyer_choice']
            self.stage = 2
            
        elif self.stage == 2:
            self.seller_prices = action['seller_prices']
            self.stage = 3
            
        elif self.stage == 3:
            buyer_choice_stage3 = action['buyer_choice']
            
            # Calculate utilities
            buyer_utility_stage1 = self.buyer_preferences[self.buyer_choice_stage1] - self.distances[self.buyer_choice_stage1]
            buyer_utility_stage3 = self.buyer_preferences[buyer_choice_stage3] - self.seller_prices[buyer_choice_stage3]
            buyer_total_utility = buyer_utility_stage1 + buyer_utility_stage3
            
            seller_utility = self.seller_prices[buyer_choice_stage3]
            
            self.stage = 4
            
            return {
                'buyer_utility': buyer_total_utility,
                'seller_utility': seller_utility,
                'done': True
            }
        
        return {'done': False}


class BaseAgent:
    """Base agent class"""
    
    def __init__(self, beta: float = 2.0):  # Higher beta for more deterministic behavior
        self.beta = beta
        
    def softmax_policy(self, q_values: np.ndarray) -> np.ndarray:
        """Convert Q-values to action probabilities using softmax"""
        # Handle numerical stability
        q_values = np.array(q_values)
        if len(q_values) == 2:
            diff = self.beta * (q_values[0] - q_values[1])
            prob_0 = 1 / (1 + np.exp(-diff))
            return np.array([prob_0, 1 - prob_0])
        else:
            return softmax(self.beta * q_values)


class ToMNeg1Buyer(BaseAgent):
    """ToM(-1) buyer: naive, maximizes utility at each stage independently"""
    
    def act_stage1(self, state: Dict) -> int:
        """Choose item in stage 1 based on immediate utility"""
        distances = state['distances']
        preferences = state['buyer_preferences']
        
        q_values = preferences - distances
        probs = self.softmax_policy(q_values)
        return np.random.choice(len(q_values), p=probs)
    
    def act_stage3(self, state: Dict) -> int:
        """Choose item in stage 3 based on immediate utility"""
        prices = state['seller_prices']
        preferences = state['buyer_preferences']
        
        q_values = preferences - prices
        probs = self.softmax_policy(q_values)
        return np.random.choice(len(q_values), p=probs)
    
    def get_policy_stage1(self, distances: np.ndarray, preferences: np.ndarray) -> np.ndarray:
        """Get policy probabilities for stage 1"""
        q_values = preferences - distances
        return self.softmax_policy(q_values)


class ToM0Seller(BaseAgent):
    """ToM(0) seller: performs Bayesian IRL on ToM(-1) buyer"""
    
    def __init__(self, beta: float = 2.0, n_preference_points: int = 101):
        super().__init__(beta)
        self.n_preference_points = n_preference_points
        self.tom_neg1_buyer = ToMNeg1Buyer(beta)
        
    def bayesian_irl(self, observed_action: int, distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform Bayesian inverse reinforcement learning"""
        # Create preference grid
        preference_grid = np.linspace(0, 10, self.n_preference_points)
        posterior = np.zeros_like(preference_grid)
        
        # Uniform prior
        prior = np.ones_like(preference_grid) / len(preference_grid)
        
        for i, pref_apple in enumerate(preference_grid):
            pref_orange = 10 - pref_apple
            preferences = np.array([pref_apple, pref_orange])
            
            # Get ToM(-1) policy for these preferences
            policy = self.tom_neg1_buyer.get_policy_stage1(distances, preferences)
            
            # Likelihood of observed action
            likelihood = policy[observed_action]
            
            # Posterior ∝ likelihood × prior
            posterior[i] = likelihood * prior[i]
        
        # Normalize
        posterior = posterior / np.sum(posterior)
        
        return preference_grid, posterior
    
    def set_optimal_prices(self, preference_grid: np.ndarray, posterior: np.ndarray) -> np.ndarray:
        """Set prices based on posterior beliefs about buyer preferences"""
        prices = np.zeros(2)
        n_price_points = 51
        
        for item in range(2):
            best_price = 0
            best_expected_revenue = 0
            
            for price in np.linspace(0, 10, n_price_points):
                expected_revenue = 0
                
                for i, pref_apple in enumerate(preference_grid):
                    pref_orange = 10 - pref_apple
                    preferences = np.array([pref_apple, pref_orange])
                    
                    # Test prices (other item gets complementary price)
                    test_prices = np.array([price, 10 - price]) if item == 0 else np.array([10 - price, price])
                    
                    # Get purchase probability using ToM(-1) model
                    q_values = preferences - test_prices
                    purchase_probs = self.tom_neg1_buyer.softmax_policy(q_values)
                    
                    # Expected revenue for this preference
                    expected_revenue += posterior[i] * purchase_probs[item] * price
                
                if expected_revenue > best_expected_revenue:
                    best_expected_revenue = expected_revenue
                    best_price = price
            
            prices[item] = best_price
        
        # Ensure prices sum to 10
        prices = prices * 10 / np.sum(prices)
        
        return prices


class ToM1Buyer(BaseAgent):
    """ToM(1) buyer: strategic, plans through ToM(0) seller's inference"""
    
    def __init__(self, beta: float = 2.0):
        super().__init__(beta)
        self.tom0_seller = ToM0Seller(beta)
        
    def act_stage1(self, state: Dict) -> int:
        """Choose item strategically, considering seller's inference"""
        distances = state['distances']
        preferences = state['buyer_preferences']
        
        q_values = self._compute_q_values(distances, preferences)
        probs = self.softmax_policy(q_values)
        return np.random.choice(len(q_values), p=probs)
    
    def _compute_q_values(self, distances: np.ndarray, preferences: np.ndarray) -> np.ndarray:
        """Compute Q-values considering both immediate and future utility"""
        q_values = np.zeros(2)
        
        for action in range(2):
            # Immediate utility from stage 1
            immediate_utility = preferences[action] - distances[action]
            
            # Simulate seller's response to this action
            preference_grid, posterior = self.tom0_seller.bayesian_irl(action, distances)
            prices = self.tom0_seller.set_optimal_prices(preference_grid, posterior)
            
            # Expected utility from stage 3
            stage3_q_values = preferences - prices
            stage3_probs = self.softmax_policy(stage3_q_values)
            expected_stage3_utility = np.sum(stage3_probs * stage3_q_values)
            
            q_values[action] = immediate_utility + expected_stage3_utility
        
        return q_values
    
    def act_stage3(self, state: Dict) -> int:
        """Same as ToM(-1) for stage 3"""
        return ToMNeg1Buyer(self.beta).act_stage3(state)
    
    def get_policy_stage1(self, distances: np.ndarray, preferences: np.ndarray) -> np.ndarray:
        """Get strategic policy for stage 1"""
        q_values = self._compute_q_values(distances, preferences)
        return self.softmax_policy(q_values)


class ToM2Seller(BaseAgent):
    """ToM(2) seller: skeptical, performs Bayesian IRL on ToM(1) buyer"""
    
    def __init__(self, beta: float = 2.0, n_preference_points: int = 101):
        super().__init__(beta)
        self.n_preference_points = n_preference_points
        self.tom1_buyer = ToM1Buyer(beta)
        
    def bayesian_irl(self, observed_action: int, distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform Bayesian IRL assuming strategic ToM(1) buyer"""
        preference_grid = np.linspace(0, 10, self.n_preference_points)
        posterior = np.zeros_like(preference_grid)
        
        # Uniform prior
        prior = np.ones_like(preference_grid) / len(preference_grid)
        
        for i, pref_apple in enumerate(preference_grid):
            pref_orange = 10 - pref_apple
            preferences = np.array([pref_apple, pref_orange])
            
            # Get ToM(1) strategic policy
            policy = self.tom1_buyer.get_policy_stage1(distances, preferences)
            
            # Likelihood of observed action
            likelihood = policy[observed_action]
            
            # Posterior ∝ likelihood × prior
            posterior[i] = likelihood * prior[i]
        
        # Normalize
        posterior = posterior / np.sum(posterior)
        
        return preference_grid, posterior
    
    def set_optimal_prices(self, preference_grid: np.ndarray, posterior: np.ndarray) -> np.ndarray:
        """Set prices defensively against strategic buyer"""
        prices = np.zeros(2)
        n_price_points = 51
        
        for item in range(2):
            best_price = 0
            best_expected_revenue = 0
            
            for price in np.linspace(0, 10, n_price_points):
                expected_revenue = 0
                
                for i, pref_apple in enumerate(preference_grid):
                    pref_orange = 10 - pref_apple
                    preferences = np.array([pref_apple, pref_orange])
                    
                    # Test prices
                    test_prices = np.array([price, 10 - price]) if item == 0 else np.array([10 - price, price])
                    
                    # Use ToM(-1) buyer model for stage 3 (same as ToM(1) at stage 3)
                    q_values = preferences - test_prices
                    purchase_probs = softmax(self.beta * q_values)
                    
                    expected_revenue += posterior[i] * purchase_probs[item] * price
                
                if expected_revenue > best_expected_revenue:
                    best_expected_revenue = expected_revenue
                    best_price = price
            
            prices[item] = best_price
        
        # Ensure prices sum to 10
        prices = prices * 10 / np.sum(prices)
        
        return prices


class ToM3Buyer(BaseAgent):
    """ToM(3) buyer: plans through ToM(2) seller's defensive inference"""
    
    def __init__(self, beta: float = 2.0):
        super().__init__(beta)
        self.tom2_seller = ToM2Seller(beta)
        
    def act_stage1(self, state: Dict) -> int:
        """Choose item strategically against skeptical seller"""
        distances = state['distances']
        preferences = state['buyer_preferences']
        
        q_values = self._compute_q_values(distances, preferences)
        probs = self.softmax_policy(q_values)
        return np.random.choice(len(q_values), p=probs)
    
    def _compute_q_values(self, distances: np.ndarray, preferences: np.ndarray) -> np.ndarray:
        """Compute Q-values against skeptical seller"""
        q_values = np.zeros(2)
        
        for action in range(2):
            # Immediate utility
            immediate_utility = preferences[action] - distances[action]
            
            # Simulate skeptical seller's response
            preference_grid, posterior = self.tom2_seller.bayesian_irl(action, distances)
            prices = self.tom2_seller.set_optimal_prices(preference_grid, posterior)
            
            # Expected future utility
            stage3_q_values = preferences - prices
            stage3_probs = self.softmax_policy(stage3_q_values)
            expected_stage3_utility = np.sum(stage3_probs * stage3_q_values)
            
            q_values[action] = immediate_utility + expected_stage3_utility
        
        return q_values
    
    def act_stage3(self, state: Dict) -> int:
        """Same as ToM(-1) for stage 3"""
        return ToMNeg1Buyer(self.beta).act_stage3(state)
    
    def get_policy_stage1(self, distances: np.ndarray, preferences: np.ndarray) -> np.ndarray:
        """Get strategic policy for stage 1"""
        q_values = self._compute_q_values(distances, preferences)
        return self.softmax_policy(q_values)


class InformationTheoryMetrics:
    """Calculate information-theoretic metrics for analysis"""
    
    @staticmethod
    def mutual_information_continuous(preference_values: np.ndarray, policies: np.ndarray) -> float:
        """Calculate mutual information between preferences and actions"""
        n_prefs, n_actions = policies.shape
        
        # Marginal distribution over actions
        action_probs = np.mean(policies, axis=0)
        
        # Calculate MI using definition
        mi = 0
        for i in range(n_prefs):
            for a in range(n_actions):
                if policies[i, a] > 1e-10:  # Avoid log(0)
                    mi += (1/n_prefs) * policies[i, a] * np.log(policies[i, a] / action_probs[a])
        
        return mi
    
    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Calculate KL divergence D_KL(p || q)"""
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        p = p / np.sum(p)
        q = q / np.sum(q)
        return np.sum(p * np.log(p / q))
    
    @staticmethod
    def belief_update_strength(prior: np.ndarray, posterior: np.ndarray) -> float:
        """Measure strength of belief update using KL divergence"""
        return InformationTheoryMetrics.kl_divergence(posterior, prior)


def run_systematic_analysis(beta: float = 2.0, n_samples: int = 20):
    """Run systematic analysis as in the paper"""
    
    # Create parameter grids
    preference_values = np.linspace(1, 9, n_samples)  # Preference for apple (orange = 10 - apple)
    distance_values = np.linspace(1, 9, n_samples)    # Distance to apple (orange = 10 - apple)
    
    # Store results
    results = {
        'ToM(-1)_vs_ToM(0)': {'mutual_info': [], 'belief_updates': [], 'policies': []},
        'ToM(1)_vs_ToM(0)': {'mutual_info': [], 'belief_updates': [], 'policies': []},
        'ToM(1)_vs_ToM(2)': {'mutual_info': [], 'belief_updates': [], 'policies': []},
        'ToM(3)_vs_ToM(2)': {'mutual_info': [], 'belief_updates': [], 'policies': []}
    }
    
    # Create agents
    agents = {
        'ToM(-1)': ToMNeg1Buyer(beta),
        'ToM(1)': ToM1Buyer(beta),
        'ToM(3)': ToM3Buyer(beta),
        'ToM(0)': ToM0Seller(beta),
        'ToM(2)': ToM2Seller(beta)
    }
    
    for pref in preference_values:
        preferences = np.array([pref, 10 - pref])
        
        # Calculate policies for different distances
        policies_tom_neg1 = []
        policies_tom1 = []
        policies_tom3 = []
        
        belief_updates_tom0 = []
        belief_updates_tom2 = []
        
        for dist in distance_values:
            distances = np.array([dist, 10 - dist])
            
            # Get policies
            policy_neg1 = agents['ToM(-1)'].get_policy_stage1(distances, preferences)
            policy_1 = agents['ToM(1)'].get_policy_stage1(distances, preferences)
            policy_3 = agents['ToM(3)'].get_policy_stage1(distances, preferences)
            
            policies_tom_neg1.append(policy_neg1)
            policies_tom1.append(policy_1)
            policies_tom3.append(policy_3)
            
            # Calculate belief updates for both actions
            for action in [0, 1]:
                # ToM(0) seller's belief update
                _, posterior_tom0 = agents['ToM(0)'].bayesian_irl(action, distances)
                prior = np.ones_like(posterior_tom0) / len(posterior_tom0)
                belief_update_tom0 = InformationTheoryMetrics.belief_update_strength(prior, posterior_tom0)
                belief_updates_tom0.append(belief_update_tom0)
                
                # ToM(2) seller's belief update
                _, posterior_tom2 = agents['ToM(2)'].bayesian_irl(action, distances)
                belief_update_tom2 = InformationTheoryMetrics.belief_update_strength(prior, posterior_tom2)
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
        results['ToM(-1)_vs_ToM(0)']['mutual_info'].append(mi_neg1)
        results['ToM(-1)_vs_ToM(0)']['belief_updates'].append(np.mean(belief_updates_tom0))
        results['ToM(-1)_vs_ToM(0)']['policies'].append(policies_tom_neg1.mean(axis=0))
        
        results['ToM(1)_vs_ToM(0)']['mutual_info'].append(mi_1)
        results['ToM(1)_vs_ToM(0)']['belief_updates'].append(np.mean(belief_updates_tom0))
        results['ToM(1)_vs_ToM(0)']['policies'].append(policies_tom1.mean(axis=0))
        
        results['ToM(1)_vs_ToM(2)']['mutual_info'].append(mi_1)
        results['ToM(1)_vs_ToM(2)']['belief_updates'].append(np.mean(belief_updates_tom2))
        results['ToM(1)_vs_ToM(2)']['policies'].append(policies_tom1.mean(axis=0))
        
        results['ToM(3)_vs_ToM(2)']['mutual_info'].append(mi_3)
        results['ToM(3)_vs_ToM(2)']['belief_updates'].append(np.mean(belief_updates_tom2))
        results['ToM(3)_vs_ToM(2)']['policies'].append(policies_tom3.mean(axis=0))
    
    return results, preference_values


def plot_results(results: Dict, preference_values: np.ndarray):
    """Plot results as in the paper"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot mutual information
    ax = axes[0, 0]
    for key in results.keys():
        ax.plot(preference_values, results[key]['mutual_info'], 
                label=key.replace('_vs_', ' vs '), linewidth=2)
    ax.set_xlabel('Buyer Preference for Apple')
    ax.set_ylabel('Mutual Information I(r, a₁)')
    ax.set_title('Mutual Information: Preference Revelation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot belief updates
    ax = axes[0, 1]
    for key in results.keys():
        ax.plot(preference_values, results[key]['belief_updates'], 
                label=key.replace('_vs_', ' vs '), linewidth=2)
    ax.set_xlabel('Buyer Preference for Apple')
    ax.set_ylabel('KL Divergence D_KL(p(r|a₁)||p(r))')
    ax.set_title('Seller Skepticism: Belief Update Strength')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot action probabilities
    ax = axes[1, 0]
    for key in results.keys():
        policies = np.array(results[key]['policies'])
        ax.plot(preference_values, policies[:, 0], 
                label=f"{key.split('_vs_')[0]} (Apple)", linewidth=2)
    ax.set_xlabel('Buyer Preference for Apple')
    ax.set_ylabel('P(Choose Apple in Stage 1)')
    ax.set_title('Strategic Behavior: Action Probabilities')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot deception effectiveness (difference from naive)
    ax = axes[1, 1]
    naive_mi = results['ToM(-1)_vs_ToM(0)']['mutual_info']
    for key in results.keys():
        if 'ToM(1)' in key or 'ToM(3)' in key:
            deception = np.array(naive_mi) - np.array(results[key]['mutual_info'])
            ax.plot(preference_values, deception, 
                    label=key.replace('_vs_', ' vs '), linewidth=2)
    ax.set_xlabel('Buyer Preference for Apple')
    ax.set_ylabel('Deception Effectiveness (ΔI)')
    ax.set_title('Information Hiding: Deception vs Naive')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    print("Running systematic analysis to reproduce paper results...")
    
    # Run analysis
    results, preference_values = run_systematic_analysis(beta=2.0, n_samples=15)
    
    # Plot results
    plot_results(results, preference_values)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for key in results.keys():
        mi_avg = np.mean(results[key]['mutual_info'])
        belief_avg = np.mean(results[key]['belief_updates'])
        print(f"{key}:")
        print(f"  Average Mutual Information: {mi_avg:.3f}")
        print(f"  Average Belief Update: {belief_avg:.3f}")
    
    # Demonstrate specific example
    print("\n=== Example Interaction ===")
    preferences = np.array([7, 3])  # Strong preference for apple
    distances = np.array([3, 7])    # Apple is closer
    
    env = BuyerSellerEnvironment()
    
    # ToM(1) vs ToM(0)
    buyer = ToM1Buyer(beta=2.0)
    seller = ToM0Seller(beta=2.0)
    
    state = env.reset(preferences, distances)
    
    # Stage 1
    buyer_choice = buyer.act_stage1(state)
    env.step({'buyer_choice': buyer_choice})
    
    # Stage 2
    preference_grid, posterior = seller.bayesian_irl(buyer_choice, distances)
    prices = seller.set_optimal_prices(preference_grid, posterior)
    env.step({'seller_prices': prices})
    
    # Calculate metrics
    prior = np.ones_like(posterior) / len(posterior)
    belief_update = InformationTheoryMetrics.belief_update_strength(prior, posterior)
    
    print(f"Buyer preferences: Apple={preferences[0]:.1f}, Orange={preferences[1]:.1f}")
    print(f"Distances: Apple={distances[0]:.1f}, Orange={distances[1]:.1f}")
    print(f"Stage 1 choice: {'Apple' if buyer_choice == 0 else 'Orange'}")
    print(f"Seller prices: Apple={prices[0]:.2f}, Orange={prices[1]:.2f}")
    print(f"Belief update strength: {belief_update:.3f}")