# Albers, Evan, Mohammad T. Irfan, and Matthew J. Botsch. "Beliefs, Shocks, and the Emergence of Roles in Asset Markets: An Agent-Based Modeling Approach." Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems. 2024
# 구현 --> 안맞는 것 같음

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class Asset:
    """
    Asset class representing risky assets in the market
    """

    def __init__(self, asset_id, payoff, variance, initial_price=1000):
        self.asset_id = asset_id
        self.payoff = payoff  # Expected payoff in dollars
        self.variance = variance  # Variance of payoff in dollars squared
        self.price = initial_price  # Current market price
        self.trade_history = []  # List of (timestamp, price) tuples

    def update_price(self, new_price, timestamp):
        """Update the price of the asset and record in history"""
        self.price = new_price
        self.trade_history.append((timestamp, new_price))

    def get_expected_return(self):
        """Calculate expected return of the asset"""
        return self.payoff / self.price


class Agent:
    """
    Agent class representing investors in the market
    Following the model described in Section 3.2
    """

    def __init__(self, agent_id, risk_coefficient, initial_cash,
                 portfolio=None, beliefs=None, refresh_rate=10, step_size=1.0):
        self.agent_id = agent_id
        self.risk_coefficient = risk_coefficient  # Risk aversion parameter r
        self.cash = initial_cash  # Quantity of risk-free asset (τ_A)
        # Θ_A = {a0, a1, a2, ...aN}
        self.portfolio = portfolio if portfolio is not None else {}
        self.beliefs = beliefs if beliefs is not None else {}  # ϕ_A = {p0, p1, p2...pN}
        self.refresh_rate = refresh_rate  # Time interval between trade evaluations
        self.step_size = step_size  # Amount to adjust limit orders by
        self.orders = []  # Current outstanding orders

    def calculate_portfolio_value(self, market):
        """Calculate the total value of the agent's portfolio including cash"""
        portfolio_value = self.cash
        for asset_id, quantity in self.portfolio.items():
            asset_price = market.get_asset_price(asset_id)
            portfolio_value += quantity * asset_price
        return portfolio_value

    def calculate_optimal_portfolio(self, market):
        """
        Calculate the optimal portfolio weights according to Tobin's separation theorem
        Returns a dictionary with asset IDs as keys and optimal quantities as values
        """
        assets = market.assets
        risk_free_rate = market.risk_free_rate

        # Calculate the market portfolio
        asset_expected_returns = {}
        asset_variances = {}

        # Get expected returns and variances based on agent's beliefs
        for asset_id, asset in assets.items():
            if asset_id in self.beliefs:
                # Use agent's belief about asset price
                expected_return = asset.payoff / self.beliefs[asset_id]
            else:
                # Use current market price
                expected_return = asset.payoff / asset.price

            asset_expected_returns[asset_id] = expected_return
            asset_variances[asset_id] = asset.variance / (asset.price ** 2)

        # For simplicity, we're using a diagonal covariance matrix (no correlations)
        # In a more complex model, we would calculate the full covariance matrix

        # Calculate the market portfolio weights
        er_minus_rf = {asset_id: er - risk_free_rate for asset_id,
                       er in asset_expected_returns.items()}

        # Calculate optimal weights based on formula: w = (ER_M - R_f) / (r * σ²_M)
        total_portfolio_value = self.calculate_portfolio_value(market)

        optimal_portfolio = {}
        for asset_id in assets:
            # Calculate optimal weight for this asset
            if asset_id in asset_variances and asset_variances[asset_id] > 0:
                weight = er_minus_rf[asset_id] / \
                    (self.risk_coefficient * asset_variances[asset_id])

                # Convert weight to quantity (number of shares)
                asset_price = self.beliefs.get(
                    asset_id, assets[asset_id].price)
                optimal_quantity = int(
                    (weight * total_portfolio_value) / asset_price)

                # Ensure non-negative holdings
                optimal_portfolio[asset_id] = max(0, optimal_quantity)
            else:
                optimal_portfolio[asset_id] = 0

        return optimal_portfolio

    def generate_orders(self, market, optimal_portfolio):
        """
        Generate buy/sell orders based on the difference between current and optimal portfolio
        Following Algorithm 2 in the paper
        """
        orders = []

        for asset_id, optimal_quantity in optimal_portfolio.items():
            current_quantity = self.portfolio.get(asset_id, 0)

            if current_quantity < optimal_quantity:
                # Buy order
                quantity_to_buy = optimal_quantity - current_quantity
                price = self.beliefs.get(
                    asset_id, market.assets[asset_id].price)

                orders.append({
                    'agent_id': self.agent_id,
                    'asset_id': asset_id,
                    'type': 'buy',
                    'quantity': quantity_to_buy,
                    'price': price,
                    'timestamp': market.current_time
                })

            elif current_quantity > optimal_quantity:
                # Sell order
                quantity_to_sell = current_quantity - optimal_quantity
                price = self.beliefs.get(
                    asset_id, market.assets[asset_id].price)

                orders.append({
                    'agent_id': self.agent_id,
                    'asset_id': asset_id,
                    'type': 'sell',
                    'quantity': quantity_to_sell,
                    'price': price,
                    'timestamp': market.current_time
                })

        return orders

    def update_beliefs(self, asset_id, trade_type, trade_price=None, timeout=False):
        """
        Update beliefs about asset prices based on market information
        Following Algorithm 3 in the paper
        """
        if trade_price is not None:
            # If a trade occurred, update belief to the trade price
            self.beliefs[asset_id] = trade_price
        elif timeout:
            # If order timed out, adjust price accordingly
            if trade_type == 'buy':
                # Increase buy price
                current_belief = self.beliefs.get(asset_id)
                if current_belief:
                    self.beliefs[asset_id] = current_belief * \
                        (1 + self.step_size / 100)
            else:  # sell order
                # Decrease sell price
                current_belief = self.beliefs.get(asset_id)
                if current_belief:
                    self.beliefs[asset_id] = current_belief * \
                        (1 - self.step_size / 100)

    def execute_trade(self, order, trade_price):
        """Execute a completed trade"""
        asset_id = order['asset_id']
        quantity = order['quantity']

        if order['type'] == 'buy':
            # Buy order execution
            cost = quantity * trade_price
            if self.cash >= cost:
                self.cash -= cost
                self.portfolio[asset_id] = self.portfolio.get(
                    asset_id, 0) + quantity
                # Update belief
                self.update_beliefs(asset_id, 'buy', trade_price)
                return True
            else:
                return False  # Not enough cash
        else:
            # Sell order execution
            if self.portfolio.get(asset_id, 0) >= quantity:
                proceeds = quantity * trade_price
                self.cash += proceeds
                self.portfolio[asset_id] = self.portfolio.get(
                    asset_id, 0) - quantity
                # Update belief
                self.update_beliefs(asset_id, 'sell', trade_price)
                return True
            else:
                return False  # Not enough assets


class NoiseTrader(Agent):
    """
    A noise trader is an agent with imperfect (biased) information about asset payoffs
    """

    def __init__(self, agent_id, risk_coefficient, initial_cash,
                 portfolio=None, beliefs=None, refresh_rate=10, step_size=1.0,
                 payoff_bias=None):
        super().__init__(agent_id, risk_coefficient, initial_cash,
                         portfolio, beliefs, refresh_rate, step_size)
        self.payoff_bias = payoff_bias if payoff_bias is not None else {}

    def calculate_optimal_portfolio(self, market):
        """Override to use biased payoff beliefs"""
        assets = market.assets
        risk_free_rate = market.risk_free_rate

        asset_expected_returns = {}
        asset_variances = {}

        for asset_id, asset in assets.items():
            # Apply payoff bias if available
            if asset_id in self.payoff_bias:
                payoff = asset.payoff * (1 + self.payoff_bias[asset_id])
            else:
                payoff = asset.payoff

            if asset_id in self.beliefs:
                # Use agent's belief about asset price
                expected_return = payoff / self.beliefs[asset_id]
            else:
                # Use current market price
                expected_return = payoff / asset.price

            asset_expected_returns[asset_id] = expected_return
            asset_variances[asset_id] = asset.variance / (asset.price ** 2)

        # Calculate the market portfolio weights
        er_minus_rf = {asset_id: er - risk_free_rate for asset_id,
                       er in asset_expected_returns.items()}

        # Calculate optimal weights
        total_portfolio_value = self.calculate_portfolio_value(market)

        optimal_portfolio = {}
        for asset_id in assets:
            if asset_id in asset_variances and asset_variances[asset_id] > 0:
                weight = er_minus_rf[asset_id] / \
                    (self.risk_coefficient * asset_variances[asset_id])
                asset_price = self.beliefs.get(
                    asset_id, assets[asset_id].price)
                optimal_quantity = int(
                    (weight * total_portfolio_value) / asset_price)
                optimal_portfolio[asset_id] = max(0, optimal_quantity)
            else:
                optimal_portfolio[asset_id] = 0

        return optimal_portfolio


class AssetMarket:
    """
    Main market class that handles trading between agents
    Simulates a ProRata exchange algorithm similar to MAXE
    """

    def __init__(self, risk_free_rate=0.01, order_timeout=100):
        self.assets = {}  # Dictionary of assets by ID
        self.agents = []  # List of agents
        self.order_book = {'buy': defaultdict(list), 'sell': defaultdict(list)}
        self.risk_free_rate = risk_free_rate
        self.current_time = 0
        self.order_timeout = order_timeout
        self.trade_history = []
        self.running = False

    def add_asset(self, asset):
        """Add an asset to the market"""
        self.assets[asset.asset_id] = asset

    def add_agent(self, agent):
        """Add an agent to the market"""
        self.agents.append(agent)

    def get_asset_price(self, asset_id):
        """Get the current price of an asset"""
        if asset_id in self.assets:
            return self.assets[asset_id].price
        else:
            return None

    def submit_order(self, order):
        """Submit an order to the order book"""
        asset_id = order['asset_id']
        order_type = order['type']
        self.order_book[order_type][asset_id].append(order)

    def match_orders(self, asset_id):
        """
        Match buy and sell orders for an asset
        Using a simplified ProRata matching algorithm
        """
        buy_orders = sorted(self.order_book['buy'][asset_id],
                            key=lambda x: x['price'], reverse=True)
        sell_orders = sorted(self.order_book['sell'][asset_id],
                             key=lambda x: x['price'])

        if not buy_orders or not sell_orders:
            return []

        # Check if highest bid >= lowest ask
        if buy_orders[0]['price'] >= sell_orders[0]['price']:
            # Determine trading price (midpoint)
            trade_price = (buy_orders[0]['price'] + sell_orders[0]['price']) / 2

            # Match orders (simplified version)
            trades = []
            remaining_buy = buy_orders[0]['quantity']
            remaining_sell = sell_orders[0]['quantity']

            # Execute the trade with minimum of buy and sell quantities
            trade_quantity = min(remaining_buy, remaining_sell)

            trades.append({
                'buyer_id': buy_orders[0]['agent_id'],
                'seller_id': sell_orders[0]['agent_id'],
                'asset_id': asset_id,
                'quantity': trade_quantity,
                'price': trade_price,
                'timestamp': self.current_time
            })

            # Update order quantities
            buy_orders[0]['quantity'] -= trade_quantity
            sell_orders[0]['quantity'] -= trade_quantity

            # Remove filled orders
            if buy_orders[0]['quantity'] == 0:
                self.order_book['buy'][asset_id].remove(buy_orders[0])
            if sell_orders[0]['quantity'] == 0:
                self.order_book['sell'][asset_id].remove(sell_orders[0])

            # Update asset price
            self.assets[asset_id].update_price(trade_price, self.current_time)

            return trades

        return []

    def execute_trades(self, trades):
        """Execute the trades and update agent portfolios"""
        for trade in trades:
            buyer = next(
                (agent for agent in self.agents if agent.agent_id == trade['buyer_id']), None)
            seller = next(
                (agent for agent in self.agents if agent.agent_id == trade['seller_id']), None)

            if buyer and seller:
                buyer_order = {
                    'asset_id': trade['asset_id'],
                    'type': 'buy',
                    'quantity': trade['quantity']
                }

                seller_order = {
                    'asset_id': trade['asset_id'],
                    'type': 'sell',
                    'quantity': trade['quantity']
                }

                buyer_success = buyer.execute_trade(
                    buyer_order, trade['price'])
                seller_success = seller.execute_trade(
                    seller_order, trade['price'])

                if buyer_success and seller_success:
                    self.trade_history.append(trade)
                    # Update both agents' beliefs
                    buyer.update_beliefs(
                        trade['asset_id'], 'buy', trade['price'])
                    seller.update_beliefs(
                        trade['asset_id'], 'sell', trade['price'])

    def handle_timeouts(self):
        """Handle orders that have timed out"""
        current_time = self.current_time

        for order_type in ['buy', 'sell']:
            for asset_id in list(self.order_book[order_type].keys()):
                for order in list(self.order_book[order_type][asset_id]):
                    if current_time - order['timestamp'] > self.order_timeout:
                        # Find the agent who placed the order
                        agent = next(
                            (a for a in self.agents if a.agent_id == order['agent_id']), None)

                        if agent:
                            # Update agent's beliefs based on timeout
                            agent.update_beliefs(
                                asset_id, order_type, timeout=True)

                        # Remove the timed out order
                        self.order_book[order_type][asset_id].remove(order)

    def simulation_step(self):
        """
        Execute one step of the simulation
        Following Algorithm 1 in the paper
        """
        self.current_time += 1

        for agent in self.agents:
            # Check if it's time for this agent to refresh
            if self.current_time % agent.refresh_rate == 0:
                # Calculate optimal portfolio
                optimal_portfolio = agent.calculate_optimal_portfolio(self)

                # Check if current portfolio differs from optimal
                for asset_id, optimal_quantity in optimal_portfolio.items():
                    current_quantity = agent.portfolio.get(asset_id, 0)

                    if current_quantity != optimal_quantity:
                        # Generate and submit orders (Algorithm 2)
                        orders = agent.generate_orders(self, optimal_portfolio)

                        for order in orders:
                            self.submit_order(order)

        # Match orders for each asset
        for asset_id in self.assets:
            trades = self.match_orders(asset_id)
            self.execute_trades(trades)

        # Handle order timeouts (Algorithm 3)
        self.handle_timeouts()

    def run_simulation(self, num_steps):
        """Run the simulation for a specified number of steps"""
        self.running = True
        for _ in range(num_steps):
            if not self.running:
                break
            self.simulation_step()
        self.running = False

    def apply_shock(self, asset_id, payoff_change_pct):
        """Apply a shock to an asset's payoff"""
        if asset_id in self.assets:
            asset = self.assets[asset_id]
            original_payoff = asset.payoff
            asset.payoff = original_payoff * (1 + payoff_change_pct / 100)
            return True
        return False

    def get_asset_returns(self):
        """Calculate returns for all assets over time"""
        returns = {}
        for asset_id, asset in self.assets.items():
            if len(asset.trade_history) > 1:
                prices = [price for _, price in asset.trade_history]
                returns[asset_id] = [
                    100 * asset.payoff / price for price in prices]
            else:
                returns[asset_id] = []
        return returns

    def analyze_portfolio_convergence(self):
        """Analyze how agent portfolios converge over time"""
        # Calculate mean-variance characteristics of each agent's portfolio
        portfolio_characteristics = []

        for agent in self.agents:
            portfolio_value = agent.calculate_portfolio_value(self)

            # Skip agents with no value
            if portfolio_value <= 0:
                continue

            # Calculate expected return and variance of portfolio
            expected_return = 0
            variance = 0

            for asset_id, quantity in agent.portfolio.items():
                if asset_id in self.assets and quantity > 0:
                    asset = self.assets[asset_id]
                    asset_weight = (quantity * asset.price) / portfolio_value
                    expected_return += asset_weight * \
                        (asset.payoff / asset.price)
                    variance += (asset_weight ** 2) * \
                        (asset.variance / (asset.price ** 2))

            # Add risk-free component
            rf_weight = agent.cash / portfolio_value
            expected_return += rf_weight * self.risk_free_rate

            portfolio_characteristics.append({
                'agent_id': agent.agent_id,
                'expected_return': expected_return,
                'variance': variance,
                'risk_coef': agent.risk_coefficient
            })

        return portfolio_characteristics

    def create_asset_flow_network(self, asset_id):
        """
        Create an asset flow network for visualization
        Returns the net trade volume between agents
        """
        # Filter trades for the specified asset
        asset_trades = [
            t for t in self.trade_history if t['asset_id'] == asset_id]

        # Calculate net trade volume between agents
        flow_network = defaultdict(float)

        for trade in asset_trades:
            buyer_id = trade['buyer_id']
            seller_id = trade['seller_id']
            quantity = trade['quantity']

            # Create a directed edge from seller to buyer
            edge = (seller_id, buyer_id)
            flow_network[edge] += quantity

        return flow_network


# Example usage

def create_simple_market():
    """Create a simple market with 3 assets and 30 agents"""
    market = AssetMarket(risk_free_rate=0.01)

    # Create assets
    asset0 = Asset(0, payoff=100, variance=400, initial_price=1000)
    asset1 = Asset(1, payoff=80, variance=200, initial_price=800)
    asset2 = Asset(2, payoff=120, variance=600, initial_price=1200)

    market.add_asset(asset0)
    market.add_asset(asset1)
    market.add_asset(asset2)

    # Create agents with different risk coefficients and initial portfolios
    for i in range(30):
        # Randomize agent parameters
        # Risk coefficient between 3 and 6
        risk_coef = 3 + np.random.uniform(0, 3)
        initial_cash = np.random.uniform(5000, 10000)

        # Create initial portfolio
        portfolio = {}
        for asset_id in range(3):
            # Random initial holdings
            quantity = np.random.randint(0, 10)
            if quantity > 0:
                portfolio[asset_id] = quantity

        # Initial beliefs are based on market prices
        beliefs = {
            0: asset0.price * np.random.uniform(0.95, 1.05),
            1: asset1.price * np.random.uniform(0.95, 1.05),
            2: asset2.price * np.random.uniform(0.95, 1.05)
        }

        agent = Agent(i, risk_coef, initial_cash, portfolio, beliefs)
        market.add_agent(agent)

    return market


def create_market_with_noise_traders():
    """Create a market with some noise traders"""
    market = AssetMarket(risk_free_rate=0.01)

    # Create assets
    asset0 = Asset(0, payoff=100, variance=400, initial_price=1000)
    asset1 = Asset(1, payoff=80, variance=200, initial_price=800)
    asset2 = Asset(2, payoff=120, variance=600, initial_price=1200)

    market.add_asset(asset0)
    market.add_asset(asset1)
    market.add_asset(asset2)

    # Create 25 regular agents
    for i in range(25):
        risk_coef = 3 + np.random.uniform(0, 3)
        initial_cash = np.random.uniform(5000, 10000)

        portfolio = {}
        for asset_id in range(3):
            quantity = np.random.randint(0, 10)
            if quantity > 0:
                portfolio[asset_id] = quantity

        beliefs = {
            0: asset0.price * np.random.uniform(0.95, 1.05),
            1: asset1.price * np.random.uniform(0.95, 1.05),
            2: asset2.price * np.random.uniform(0.95, 1.05)
        }

        agent = Agent(i, risk_coef, initial_cash, portfolio, beliefs)
        market.add_agent(agent)

    # Create 5 noise traders with biased beliefs about asset 0
    for i in range(25, 30):
        risk_coef = 3 + np.random.uniform(0, 3)
        initial_cash = np.random.uniform(5000, 10000)

        portfolio = {}
        for asset_id in range(3):
            quantity = np.random.randint(0, 10)
            if quantity > 0:
                portfolio[asset_id] = quantity

        beliefs = {
            0: asset0.price * np.random.uniform(0.95, 1.05),
            1: asset1.price * np.random.uniform(0.95, 1.05),
            2: asset2.price * np.random.uniform(0.95, 1.05)
        }

        # Create payoff bias - 10% higher payoff belief for asset 0
        payoff_bias = {0: 0.10}

        noise_trader = NoiseTrader(
            i, risk_coef, initial_cash, portfolio, beliefs, payoff_bias=payoff_bias)
        market.add_agent(noise_trader)

    return market


def run_shock_experiment():
    """Run an experiment with shocks to an asset"""
    market = create_simple_market()

    # Run the simulation for a while to reach initial equilibrium
    market.run_simulation(5000)

    # Apply negative shock to asset 2
    market.apply_shock(2, -10)  # -10% payoff shock

    # Continue simulation
    market.run_simulation(5000)

    # Apply positive shock to asset 2
    market.apply_shock(2, 11.11)  # +11.11% payoff shock

    # Continue simulation
    market.run_simulation(5000)

    # Get asset returns
    returns = market.get_asset_returns()

    # Plot the results
    for asset_id, asset_returns in returns.items():
        plt.figure(figsize=(10, 6))
        plt.plot(asset_returns)
        plt.axvline(x=5000, color='r', linestyle='--', label='Negative Shock')
        plt.axvline(x=10000, color='g', linestyle='--', label='Positive Shock')
        plt.title(f'Asset {asset_id} Returns')
        plt.ylabel('Return (%)')
        plt.xlabel('Time')
        plt.legend()
        plt.show()


# Example of how to run a simulation and visualize results
if __name__ == "__main__":
    # Create and run a simple market simulation
    market = create_simple_market()
    market.run_simulation(3000)

    # Analyze portfolio convergence
    portfolios = market.analyze_portfolio_convergence()

    # Plot portfolio characteristics
    plt.figure(figsize=(10, 6))
    for p in portfolios:
        plt.scatter(p['variance'], p['expected_return'], alpha=0.5)
    plt.title('Portfolio Mean-Variance Characteristics')
    plt.xlabel('Portfolio Variance')
    plt.ylabel('Expected Portfolio Return')
    plt.show()

    # Run simulation with noise traders
    noise_market = create_market_with_noise_traders()
    noise_market.run_simulation(3000)

    # Analyze portfolio convergence with noise traders
    noise_portfolios = noise_market.analyze_portfolio_convergence()

    # Plot portfolio characteristics with noise traders
    plt.figure(figsize=(10, 6))
    for p in noise_portfolios:
        if p['agent_id'] < 25:
            plt.scatter(p['variance'], p['expected_return'],
                        color='blue', alpha=0.5)
        else:
            plt.scatter(p['variance'], p['expected_return'],
                        color='red', alpha=0.5)
    plt.title('Portfolio Mean-Variance Characteristics with Noise Traders')
    plt.xlabel('Portfolio Variance')
    plt.ylabel('Expected Portfolio Return')
    plt.legend(['Regular Agents', 'Noise Traders'])
    plt.show()

    # Run the shock experiment
    run_shock_experiment()
