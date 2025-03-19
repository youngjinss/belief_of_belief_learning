# policy 정의
import numpy as np
from collections import deque


class policyPassive:
    """
    패시브 정책은 항상 1을 반환 (시장 가격에 영향을 최소화, 소극적으로 주문을 실행하는 전략)
    """

    def __init__(self):
        self.name = "passive"

    def get_action(self, state):
        return 1


class policyAggressive:
    """
    공격적 정책은 항상 0을 반환 (적극적으로 주문을 처리하는 전략, 시장 충격이 클 수 있음)
    """

    def __init__(self):
        self.name = "aggressive"

    def get_action(self, state):
        return 0


class policyRandom:
    """
    무작위 정책은 0과 1 중에서 무작위로 선택 (패시브와 공격적 전략을 무작위로 혼합하는 방식)
    """

    def __init__(self):
        self.name = "random"

    def get_action(self, state):
        return np.random.choice([0, 1])


class policyRandomWithNoAction:
    """
    완전 무작위 정책은 -2 ~ 2 중에서 무작위로 선택 (관망하며 시장 상황을 지켜보는 옵션을 포함)
    """

    def __init__(self):
        self.name = "random_no_action"

    def get_action(self, state):
        action = np.random.choice([-2, -1, 0, 1, 2])
        if state[0] < 0 and action < 0:
            action = 0
        return action


class PolicyTobinToCAPM:
    """
    Tobin to CAPM policy implementation based on Section 3.2 and Algorithms 2 & 3
    in Albers and Mathew (2024). This policy adapts the Capital Asset Pricing Model
    using principles from Tobin's separation theorem to determine optimal trading actions.
    """

    def __init__(self, window_size=20, rf_rate=0.01, risk_aversion=1.0):
        """
        Initialize the Tobin to CAPM policy.

        Args:
            window_size (int): Size of the rolling window for estimating parameters
            rf_rate (float): Risk-free rate (annualized)
            risk_aversion (float): Risk aversion parameter (λ) for the utility function
        """
        self.name = "tobin_to_capm"
        self.window_size = window_size
        self.return_history = deque(maxlen=window_size)
        self.market_return_history = deque(maxlen=window_size)
        self.rf_rate = rf_rate / 252  # Daily risk-free rate (assuming 252 trading days)
        self.risk_aversion = risk_aversion
        self.beta = 1.0  # Initialize beta
        self.previous_action = 0
        self.previous_optimal_weight = 0.0  # Store previous optimal weight
        self.trade_proportion = 0.0  # Store proportion to trade

    def update_histories(self, returns, market_returns=None):
        """
        Update return histories

        Args:
            returns (float or array): Asset returns
            market_returns (float or array): Market returns (if available)
        """
        # Handle both single value and array of returns with NaN checking
        if isinstance(returns, (list, np.ndarray)):
            for ret in returns:
                # Only add finite values to history
                if ret is not None and not (
                    isinstance(ret, (float, np.floating))
                    and (np.isnan(ret) or np.isinf(ret))
                ):
                    self.return_history.append(ret)
        elif returns is not None and not (
            isinstance(returns, (float, np.floating))
            and (np.isnan(returns) or np.isinf(returns))
        ):
            self.return_history.append(returns)

        # Update market return history if provided
        if market_returns is not None:
            if isinstance(market_returns, (list, np.ndarray)):
                for ret in market_returns:
                    # Only add finite values to history
                    if ret is not None and not (
                        isinstance(ret, (float, np.floating))
                        and (np.isnan(ret) or np.isinf(ret))
                    ):
                        self.market_return_history.append(ret)
            elif market_returns is not None and not (
                isinstance(market_returns, (float, np.floating))
                and (np.isnan(market_returns) or np.isinf(market_returns))
            ):
                self.market_return_history.append(market_returns)
        # If no market returns provided, use the asset returns as a proxy
        elif len(self.return_history) > 0:
            if isinstance(returns, (list, np.ndarray)):
                # Filter out NaN and infinite values before calculating mean
                valid_returns = [
                    r
                    for r in returns
                    if r is not None
                    and not (
                        isinstance(r, (float, np.floating))
                        and (np.isnan(r) or np.isinf(r))
                    )
                ]
                if valid_returns:
                    avg_return = np.mean(valid_returns)
                    if not np.isnan(avg_return) and not np.isinf(avg_return):
                        self.market_return_history.append(avg_return)
            elif returns is not None and not (
                isinstance(returns, (float, np.floating))
                and (np.isnan(returns) or np.isinf(returns))
            ):
                avg_return = returns
                self.market_return_history.append(avg_return)

    def estimate_parameters(self):
        """
        Estimate CAPM parameters: beta, expected_return, and volatility

        Returns:
            tuple: (beta, expected_return, volatility)
        """
        # Need sufficient history to estimate parameters
        if len(self.return_history) < self.window_size // 2:
            return 1.0, self.rf_rate, 0.001

        # Convert histories to numpy arrays for calculations
        returns = np.array(self.return_history)
        market_returns = np.array(self.market_return_history)

        # Filter out NaN or infinite values
        returns = returns[np.isfinite(returns)]
        market_returns = market_returns[np.isfinite(market_returns)]

        # If we don't have enough valid data after filtering, return default values
        if len(returns) < 2:
            return 1.0, self.rf_rate, 0.001

        if len(market_returns) == 0:
            # If market returns not available, use default values
            beta = 1.0
            # Handle potential NaN values
            mean_return = np.nanmean(returns) if len(returns) > 0 else 0
            if np.isnan(mean_return):
                mean_return = 0

            volatility = np.nanstd(returns) if len(returns) > 0 else 0.001
            if np.isnan(volatility) or volatility < 0.0001:
                volatility = 0.001

            expected_return = self.rf_rate + beta * (mean_return - self.rf_rate)
        else:
            # Ensure the arrays have the same length for calculation
            min_length = min(len(returns), len(market_returns))
            if min_length < 2:  # Need at least 2 data points for meaningful calculation
                return 1.0, self.rf_rate, 0.001

            returns = returns[-min_length:]
            market_returns = market_returns[-min_length:]

            # Calculate beta using covariance with error handling
            try:
                market_std = np.std(market_returns)
                if market_std > 0:
                    cov_matrix = np.cov(returns, market_returns)
                    if cov_matrix.shape == (
                        2,
                        2,
                    ):  # Ensure proper covariance matrix shape
                        beta = cov_matrix[0, 1] / np.var(market_returns)
                    else:
                        beta = 1.0
                else:
                    beta = 1.0
            except:
                beta = 1.0

            # Calculate mean returns and volatility
            try:
                mean_return = np.nanmean(returns)
                mean_market_return = np.nanmean(market_returns)
                if np.isnan(mean_return) or np.isnan(mean_market_return):
                    mean_return = 0
                    mean_market_return = 0
            except:
                mean_return = 0
                mean_market_return = 0

            # Calculate volatility and expected return
            try:
                volatility = np.nanstd(returns)
                if np.isnan(volatility) or volatility < 0.0001:
                    volatility = 0.001
            except:
                volatility = 0.001

            try:
                expected_return = self.rf_rate + beta * (
                    mean_market_return - self.rf_rate
                )
                if np.isnan(expected_return):
                    expected_return = self.rf_rate
            except:
                expected_return = self.rf_rate

        # Limit beta to reasonable range to avoid extreme values
        beta = max(0.1, min(3.0, beta))

        # Handle any remaining NaN values
        if np.isnan(beta):
            beta = 1.0
        if np.isnan(expected_return):
            expected_return = self.rf_rate
        if np.isnan(volatility):
            volatility = 0.001

        return beta, expected_return, volatility

    def calculate_optimal_weight(self, expected_excess_return, volatility):
        """
        Calculate optimal weight in risky asset based on Tobin's separation theorem

        Args:
            expected_excess_return (float): Expected excess return (over risk-free rate)
            volatility (float): Estimated volatility of returns

        Returns:
            float: Optimal weight between -1 and 1
        """
        # Check for NaN values
        if np.isnan(expected_excess_return) or np.isnan(volatility):
            return 0.0

        # Avoid division by zero
        if volatility < 0.0001:
            volatility = 0.0001

        try:
            # Calculate optimal weight based on Tobin's theorem (maximizing utility)
            # w* = (E[r] - rf) / (λ * σ²)
            optimal_weight = expected_excess_return / (
                self.risk_aversion * (volatility**2)
            )

            # Check for NaN or infinite results
            if np.isnan(optimal_weight) or np.isinf(optimal_weight):
                optimal_weight = 0.0

            # Limit weight to reasonable range
            optimal_weight = max(-1.0, min(1.0, optimal_weight))

            return optimal_weight
        except Exception as e:
            # If any error occurs, return a safe default
            print(f"Error in calculate_optimal_weight: {e}")
            return 0.0

    def determine_action(self, beta, expected_return, volatility):
        """
        Determine trading action based on CAPM parameters and Tobin's theorem

        Args:
            beta (float): Beta (systematic risk) estimate
            expected_return (float): Expected return based on CAPM
            volatility (float): Volatility estimate

        Returns:
            int: Action value between -2 and 2
            float: Trade proportion relative to current holdings
        """
        # Check for NaN values and set defaults if needed
        if np.isnan(expected_return) or np.isnan(volatility):
            return 0, 0.0  # Default to no action if we have NaN values

        # Calculate expected excess return
        expected_excess_return = expected_return - self.rf_rate

        # Calculate optimal portfolio weight using Tobin's theorem
        optimal_weight = self.calculate_optimal_weight(
            expected_excess_return, volatility
        )

        # Calculate trade proportion based on difference from previous optimal weight
        self.trade_proportion = optimal_weight - self.previous_optimal_weight

        # Update previous optimal weight for next round
        self.previous_optimal_weight = optimal_weight

        # Determine action based on weight difference
        if abs(self.trade_proportion) < 0.05:  # Small threshold for meaningful action
            action = 0  # No significant change needed
        else:
            if self.trade_proportion > 0:  # Need to increase position
                action = 2
            else:  # Need to decrease position
                action = -2

        # Apply mean reversion adjustment based on beta
        if not np.isnan(beta):
            if beta > 1.5 and action > 0:
                # High beta stocks are more volatile, be more cautious with buys
                action = max(0, action - 1)
            elif beta < 0.5 and action < 0:
                # Low beta stocks are less volatile, be more cautious with sells
                action = min(0, action + 1)

        return action

    def get_action(self, state):
        """
        Determine trading action based on current state.

        Args:
            state: The current state observation from the environment

        Returns:
            int: Action value between -2 and 2
            float: Trade proportion to execute
        """
        try:
            # Extract relevant information from state
            current_holdings = state[0][0] if not np.isnan(state[0][0]) else 0

            # Get returns directly from state[8:]
            returns = state[8:] if len(state) > 8 else []

            # Filter out NaN values
            returns = [
                r
                for r in returns
                if r is not None
                and not (
                    isinstance(r, (float, np.floating)) and (np.isnan(r) or np.isinf(r))
                )
            ]

            # Update histories with returns
            if returns:
                self.update_histories(returns)

            # Estimate CAPM parameters
            beta, expected_return, volatility = self.estimate_parameters()

            # Determine action and trade proportion based on the parameters
            action = self.determine_action(beta, expected_return, volatility)

            # Adjust action based on current holdings
            # Prevent short selling if not allowed (similar to the random policy)
            if current_holdings <= 0 and action < 0:
                action = 0
                self.trade_proportion = 0.0

            self.previous_action = action

            return action
        except Exception as e:
            # If any error occurs, log it and return a safe default action
            print(f"Error in get_action: {e}")
            return 0, 0.0
