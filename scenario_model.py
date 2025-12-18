"""
Scenario Modeling Module
Implements stress testing and forward-looking simulations using GBM.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class StressScenarioConfig:
    """Configuration for a stress scenario."""
    name: str
    start_date: datetime
    duration_days: int
    affected_tickers: List[str]
    expected_drop: float  # e.g., 0.10 for 10% drop
    drop_std: float  # Standard deviation of drop, e.g., 0.02 for ±2%
    volatility_increase: float  # e.g., 0.10 for 10% volatility increase


@dataclass
class SimulationResult:
    """Container for simulation results."""
    simulated_prices: pd.DataFrame
    simulated_returns: pd.DataFrame
    portfolio_returns: Dict  # Dict with 'daily' and 'cumulative' DataFrames
    scenario_config: StressScenarioConfig
    statistics: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert results to dictionary format."""
        return {
            'scenario_name': self.scenario_config.name,
            'scenario_config': {
                'start_date': str(self.scenario_config.start_date.date()),
                'duration_days': self.scenario_config.duration_days,
                'affected_tickers': self.scenario_config.affected_tickers,
                'expected_drop': self.scenario_config.expected_drop,
                'drop_std': self.scenario_config.drop_std,
                'volatility_increase': self.scenario_config.volatility_increase
            },
            'statistics': self.statistics,
            'portfolio_returns_summary': {
                'mean_final_return': float(self.portfolio_returns['cumulative'].iloc[-1].mean()),
                'std_final_return': float(self.portfolio_returns['cumulative'].iloc[-1].std()),
                'percentile_5': float(np.percentile(self.portfolio_returns['cumulative'].iloc[-1], 5)),
                'percentile_25': float(np.percentile(self.portfolio_returns['cumulative'].iloc[-1], 25)),
                'percentile_50': float(np.percentile(self.portfolio_returns['cumulative'].iloc[-1], 50)),
                'percentile_75': float(np.percentile(self.portfolio_returns['cumulative'].iloc[-1], 75)),
                'percentile_95': float(np.percentile(self.portfolio_returns['cumulative'].iloc[-1], 95))
            }
        }


class ScenarioSimulator:
    """
    Simulates forward-looking scenarios using Geometric Brownian Motion (GBM).

    GBM equation: dS = μS dt + σS dW
    Where:
        - S: Asset price
        - μ: Drift (expected return)
        - σ: Volatility
        - dW: Wiener process increment
    """

    def __init__(self,
                 historical_returns: pd.DataFrame,
                 last_prices: pd.Series,
                 trading_days_per_year: int = 252):
        """
        Initialize the simulator.

        Args:
            historical_returns: DataFrame of historical daily returns
            last_prices: Series of last observed prices
            trading_days_per_year: Number of trading days per year
        """
        self.historical_returns = historical_returns
        self.last_prices = last_prices
        self.trading_days = trading_days_per_year
        self.tickers = historical_returns.columns.tolist()

        # Calculate baseline statistics from historical data
        self.base_mean = historical_returns.mean() * self.trading_days
        self.base_std = historical_returns.std() * np.sqrt(self.trading_days)
        self.correlation = historical_returns.corr()

    def simulate_gbm(self,
                     n_days: int,
                     n_simulations: int,
                     drift: Optional[pd.Series] = None,
                     volatility: Optional[pd.Series] = None,
                     initial_shock: Optional[Dict[str, Tuple[float, float]]] = None,
                     random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate asset prices using correlated GBM.

        Args:
            n_days: Number of days to simulate
            n_simulations: Number of Monte Carlo simulations
            drift: Annualized drift for each asset (uses historical if None)
            volatility: Annualized volatility for each asset (uses historical if None)
            initial_shock: Dict of ticker -> (mean_shock, shock_std) for initial day
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (simulated_prices, simulated_returns) arrays
            Shape: (n_simulations, n_days, n_assets)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        n_assets = len(self.tickers)

        # Use provided or baseline parameters
        if drift is None:
            drift = self.base_mean
        if volatility is None:
            volatility = self.base_std

        # Convert to daily parameters
        daily_drift = drift.values / self.trading_days
        daily_vol = volatility.values / np.sqrt(self.trading_days)

        # Cholesky decomposition for correlated random numbers
        cholesky = np.linalg.cholesky(self.correlation.values)

        # Initialize arrays
        prices = np.zeros((n_simulations, n_days + 1, n_assets))
        returns = np.zeros((n_simulations, n_days, n_assets))

        # Set initial prices
        prices[:, 0, :] = self.last_prices.values

        # Apply initial shock if specified
        if initial_shock is not None:
            for i, ticker in enumerate(self.tickers):
                if ticker in initial_shock:
                    mean_shock, shock_std = initial_shock[ticker]
                    # Generate random shocks for each simulation
                    shocks = np.random.normal(mean_shock, shock_std, n_simulations)
                    prices[:, 0, i] = prices[:, 0, i] * (1 + shocks)

        # Simulate GBM paths
        for t in range(n_days):
            # Generate correlated standard normal random numbers
            z = np.random.standard_normal((n_simulations, n_assets))
            correlated_z = z @ cholesky.T

            # GBM discrete approximation
            # ln(S_t+1/S_t) = (μ - σ²/2)dt + σ√dt * Z
            drift_term = (daily_drift - 0.5 * daily_vol ** 2)
            diffusion_term = daily_vol * correlated_z

            log_returns = drift_term + diffusion_term
            returns[:, t, :] = np.exp(log_returns) - 1

            prices[:, t + 1, :] = prices[:, t, :] * np.exp(log_returns)

        return prices[:, 1:, :], returns

    def apply_stress_scenario(self,
                              scenario: StressScenarioConfig,
                              n_simulations: int = 1000,
                              forward_horizon_days: int = 252,
                              random_seed: Optional[int] = None) -> SimulationResult:
        """
        Apply a stress scenario and simulate forward.

        Args:
            scenario: Stress scenario configuration
            n_simulations: Number of Monte Carlo simulations
            forward_horizon_days: Total number of days to simulate forward
            random_seed: Random seed for reproducibility

        Returns:
            SimulationResult with simulated data
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Calculate initial shock for affected tickers
        initial_shock = {}
        for ticker in scenario.affected_tickers:
            if ticker in self.tickers:
                # Negative shock (drop)
                initial_shock[ticker] = (-scenario.expected_drop, scenario.drop_std)

        # Adjust volatility for affected tickers
        adjusted_volatility = self.base_std.copy()
        for ticker in scenario.affected_tickers:
            if ticker in self.tickers:
                adjusted_volatility[ticker] *= (1 + scenario.volatility_increase)

        # Simulate with stress parameters
        prices, returns = self.simulate_gbm(
            n_days=forward_horizon_days,
            n_simulations=n_simulations,
            volatility=adjusted_volatility,
            initial_shock=initial_shock,
            random_seed=random_seed
        )

        # Convert to DataFrames for easier analysis
        date_range = pd.date_range(
            start=scenario.start_date,
            periods=forward_horizon_days,
            freq='B'  # Business days
        )

        # Create multi-index DataFrames
        sim_prices_df = pd.DataFrame(
            prices.mean(axis=0),  # Mean across simulations
            index=date_range,
            columns=self.tickers
        )

        sim_returns_df = pd.DataFrame(
            returns.mean(axis=0),
            index=date_range,
            columns=self.tickers
        )

        # Calculate statistics
        statistics = self._calculate_simulation_statistics(prices, returns, scenario)

        # Placeholder for portfolio returns (will be filled by caller)
        portfolio_returns_df = pd.DataFrame(index=date_range)

        return SimulationResult(
            simulated_prices=sim_prices_df,
            simulated_returns=sim_returns_df,
            portfolio_returns=portfolio_returns_df,
            scenario_config=scenario,
            statistics=statistics
        )

    def simulate_portfolio_returns(self,
                                   weights: np.ndarray,
                                   tickers: List[str],
                                   scenario: StressScenarioConfig,
                                   n_simulations: int = 1000,
                                   forward_horizon_days: int = 252,
                                   random_seed: Optional[int] = None) -> SimulationResult:
        """
        Simulate portfolio returns under a stress scenario.

        Args:
            weights: Portfolio weights
            tickers: Tickers corresponding to weights
            scenario: Stress scenario configuration
            n_simulations: Number of Monte Carlo simulations
            forward_horizon_days: Total number of days to simulate
            random_seed: Random seed for reproducibility

        Returns:
            SimulationResult with portfolio-level results
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Calculate initial shock for affected tickers
        initial_shock = {}
        for ticker in scenario.affected_tickers:
            if ticker in self.tickers:
                initial_shock[ticker] = (-scenario.expected_drop, scenario.drop_std)

        # Adjust volatility for affected tickers
        adjusted_volatility = self.base_std.copy()
        for ticker in scenario.affected_tickers:
            if ticker in self.tickers:
                adjusted_volatility[ticker] *= (1 + scenario.volatility_increase)

        # Simulate
        prices, returns = self.simulate_gbm(
            n_days=forward_horizon_days,
            n_simulations=n_simulations,
            volatility=adjusted_volatility,
            initial_shock=initial_shock,
            random_seed=random_seed
        )

        # Map weights to simulation indices
        weight_vector = np.zeros(len(self.tickers))
        for w, t in zip(weights, tickers):
            if t in self.tickers:
                idx = self.tickers.index(t)
                weight_vector[idx] = w

        # Calculate portfolio returns for each simulation
        portfolio_daily_returns = np.sum(returns * weight_vector, axis=2)

        # Calculate cumulative returns
        portfolio_cumulative = np.cumprod(1 + portfolio_daily_returns, axis=1) - 1

        # Create date range
        date_range = pd.date_range(
            start=scenario.start_date,
            periods=forward_horizon_days,
            freq='B'
        )

        # Create DataFrames with all simulations
        # Each column is a simulation, each row is a day
        daily_returns_df = pd.DataFrame(
            portfolio_daily_returns.T,
            index=date_range,
            columns=[f'sim_{i}' for i in range(n_simulations)]
        )
        cumulative_returns_df = pd.DataFrame(
            portfolio_cumulative.T,
            index=date_range,
            columns=[f'sim_{i}' for i in range(n_simulations)]
        )

        # Create a container dict for both
        portfolio_returns_df = {
            'daily': daily_returns_df,
            'cumulative': cumulative_returns_df
        }

        # Calculate price and return DataFrames (mean across simulations)
        sim_prices_df = pd.DataFrame(
            prices.mean(axis=0),
            index=date_range,
            columns=self.tickers
        )

        sim_returns_df = pd.DataFrame(
            returns.mean(axis=0),
            index=date_range,
            columns=self.tickers
        )

        # Statistics
        statistics = self._calculate_simulation_statistics(prices, returns, scenario)
        statistics['portfolio'] = {
            'mean_total_return': float(portfolio_cumulative[:, -1].mean()),
            'std_total_return': float(portfolio_cumulative[:, -1].std()),
            'var_95': float(np.percentile(portfolio_cumulative[:, -1], 5)),
            'var_99': float(np.percentile(portfolio_cumulative[:, -1], 1)),
            'probability_loss': float(np.mean(portfolio_cumulative[:, -1] < 0)),
            'max_simulated_loss': float(portfolio_cumulative[:, -1].min()),
            'max_simulated_gain': float(portfolio_cumulative[:, -1].max())
        }

        return SimulationResult(
            simulated_prices=sim_prices_df,
            simulated_returns=sim_returns_df,
            portfolio_returns=portfolio_returns_df,
            scenario_config=scenario,
            statistics=statistics
        )

    def _calculate_simulation_statistics(self,
                                         prices: np.ndarray,
                                         returns: np.ndarray,
                                         scenario: StressScenarioConfig) -> Dict:
        """Calculate statistics from simulation results."""
        # Final prices relative to initial
        price_changes = (prices[:, -1, :] / prices[:, 0, :]) - 1

        stats = {
            'per_asset': {},
            'scenario_impact': {}
        }

        for i, ticker in enumerate(self.tickers):
            stats['per_asset'][ticker] = {
                'mean_return': float(price_changes[:, i].mean()),
                'std_return': float(price_changes[:, i].std()),
                'percentile_5': float(np.percentile(price_changes[:, i], 5)),
                'percentile_95': float(np.percentile(price_changes[:, i], 95))
            }

        # Impact on affected vs non-affected
        affected_idx = [self.tickers.index(t) for t in scenario.affected_tickers if t in self.tickers]
        non_affected_idx = [i for i in range(len(self.tickers)) if i not in affected_idx]

        if affected_idx:
            stats['scenario_impact']['affected_mean_return'] = float(
                price_changes[:, affected_idx].mean()
            )
        if non_affected_idx:
            stats['scenario_impact']['non_affected_mean_return'] = float(
                price_changes[:, non_affected_idx].mean()
            )

        return stats


def create_tariff_scenario(start_date: Optional[datetime] = None,
                           duration_days: int = 252,
                           thai_tickers: Optional[List[str]] = None,
                           expected_drop: float = 0.10,
                           drop_std: float = 0.02,
                           volatility_increase: float = 0.10) -> StressScenarioConfig:
    """
    Create a Trump tariffs stress scenario.

    Args:
        start_date: Start date of scenario (defaults to tomorrow)
        duration_days: Duration of the stress scenario in trading days
        thai_tickers: List of Thai-related tickers affected by tariffs
        expected_drop: Expected drop in Thai assets (e.g., 0.10 for 10%)
        drop_std: Standard deviation of the drop (e.g., 0.02 for ±2%)
        volatility_increase: Increase in volatility (e.g., 0.10 for 10%)

    Returns:
        Configured StressScenarioConfig
    """
    if start_date is None:
        start_date = datetime.now() + timedelta(days=1)

    if thai_tickers is None:
        thai_tickers = [
            # Thai Export
            'DELTA.BK', 'HANA.BK', 'STA.BK', 'IVL.BK', 'PTTGC.BK',
            # Thai Domestic
            'CPALL.BK', 'AOT.BK', 'BDMS.BK', 'SCB.BK', 'CPN.BK', 'MINT.BK',
            # Thai ETF
            'THD',
            # Thai currency
            'THB=X'
        ]

    return StressScenarioConfig(
        name='Trump Tariffs Impact',
        start_date=start_date,
        duration_days=duration_days,
        affected_tickers=thai_tickers,
        expected_drop=expected_drop,
        drop_std=drop_std,
        volatility_increase=volatility_increase
    )


if __name__ == "__main__":
    # Test the scenario simulator
    np.random.seed(42)

    # Create sample data
    tickers = ['THAI_1', 'THAI_2', 'GLOBAL_1', 'GLOBAL_2']
    n_days = 252
    n_assets = len(tickers)

    # Generate sample historical returns
    returns = pd.DataFrame(
        np.random.normal(0.0005, 0.02, (n_days, n_assets)),
        columns=tickers
    )

    last_prices = pd.Series([100, 150, 200, 250], index=tickers)

    # Create simulator
    simulator = ScenarioSimulator(returns, last_prices)

    # Create stress scenario
    scenario = StressScenarioConfig(
        name='Test Stress',
        start_date=datetime.now(),
        duration_days=252,
        affected_tickers=['THAI_1', 'THAI_2'],
        expected_drop=0.10,
        drop_std=0.02,
        volatility_increase=0.10
    )

    # Run simulation
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    result = simulator.simulate_portfolio_returns(
        weights=weights,
        tickers=tickers,
        scenario=scenario,
        n_simulations=100,
        forward_horizon_days=252,
        random_seed=42
    )

    print("Simulation Statistics:")
    print(f"  Mean Portfolio Return: {result.statistics['portfolio']['mean_total_return']:.2%}")
    print(f"  VaR 95%: {result.statistics['portfolio']['var_95']:.2%}")
    print(f"  Probability of Loss: {result.statistics['portfolio']['probability_loss']:.2%}")
