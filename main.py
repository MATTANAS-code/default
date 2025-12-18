"""
Main Module
Orchestrates portfolio optimization and stress testing pipeline.
Generates JSON output with all results.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import argparse

from data_loader import DataLoader, load_and_prepare_data
from portfolio_optimizer import (
    PortfolioOptimizer,
    backtest_portfolio,
    calculate_backtest_metrics
)
from scenario_model import (
    ScenarioSimulator,
    StressScenarioConfig,
    create_tariff_scenario
)


# Default asset universe
DEFAULT_TICKERS = [
    # Thai Export
    'DELTA.BK', 'HANA.BK', 'STA.BK', 'IVL.BK', 'PTTGC.BK',
    # Thai Domestic
    'CPALL.BK', 'AOT.BK', 'BDMS.BK', 'SCB.BK', 'CPN.BK', 'MINT.BK',
    # Global
    'WDC', 'THD',
    # Fixed Income
    'LEMB', 'VWOB', 'EMLC',
    # FX
    'THB=X'
]

# Thai-related assets for stress testing
THAI_TICKERS = [
    'DELTA.BK', 'HANA.BK', 'STA.BK', 'IVL.BK', 'PTTGC.BK',
    'CPALL.BK', 'AOT.BK', 'BDMS.BK', 'SCB.BK', 'CPN.BK', 'MINT.BK',
    'THD', 'THB=X'
]


class PortfolioAnalysisPipeline:
    """
    End-to-end pipeline for portfolio optimization and stress testing.
    """

    def __init__(self,
                 tickers: List[str] = None,
                 years_history: int = 10,
                 forward_horizon_days: int = 252,
                 risk_free_rate: float = 0.02):
        """
        Initialize the pipeline.

        Args:
            tickers: List of ticker symbols
            years_history: Years of historical data to use
            forward_horizon_days: Days for forward simulation
            risk_free_rate: Risk-free rate for Sharpe ratio
        """
        self.tickers = tickers or DEFAULT_TICKERS
        self.years_history = years_history
        self.forward_horizon_days = forward_horizon_days
        self.risk_free_rate = risk_free_rate

        self.data_loader: Optional[DataLoader] = None
        self.optimizer: Optional[PortfolioOptimizer] = None
        self.simulator: Optional[ScenarioSimulator] = None

        self.results: Dict = {}

    def run_data_loading(self) -> None:
        """Step 1: Load and clean historical data."""
        print("=" * 60)
        print("STEP 1: Loading and Cleaning Data")
        print("=" * 60)

        self.data_loader = load_and_prepare_data(
            tickers=self.tickers,
            years_history=self.years_history,
            outlier_std_threshold=4.0,
            min_data_coverage=0.8
        )

        # Store metadata
        self.results['data_info'] = {
            'requested_tickers': self.tickers,
            'available_tickers': self.data_loader.get_available_tickers(),
            'removed_tickers': list(set(self.tickers) - set(self.data_loader.get_available_tickers())),
            'data_start_date': str(self.data_loader.clean_data.index[0].date()),
            'data_end_date': str(self.data_loader.clean_data.index[-1].date()),
            'total_trading_days': len(self.data_loader.clean_data)
        }

        print(f"\nData Summary:")
        print(f"  Available tickers: {len(self.results['data_info']['available_tickers'])}")
        print(f"  Date range: {self.results['data_info']['data_start_date']} to {self.results['data_info']['data_end_date']}")
        print(f"  Trading days: {self.results['data_info']['total_trading_days']}")

    def run_optimization(self) -> None:
        """Step 2: Run mean-variance portfolio optimization."""
        print("\n" + "=" * 60)
        print("STEP 2: Portfolio Optimization")
        print("=" * 60)

        mean_returns, cov_matrix = self.data_loader.get_statistics()

        self.optimizer = PortfolioOptimizer(
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            risk_free_rate=self.risk_free_rate
        )

        # Optimize for maximum Sharpe ratio (no short selling)
        opt_result = self.optimizer.optimize_sharpe(allow_short=False)

        self.results['optimization'] = opt_result.to_dict()
        self.results['optimization']['historical_mean_returns'] = mean_returns.to_dict()

        print(f"\nOptimal Portfolio (Max Sharpe, No Shorting):")
        print(f"  Expected Return: {opt_result.expected_return:.2%}")
        print(f"  Volatility: {opt_result.volatility:.2%}")
        print(f"  Sharpe Ratio: {opt_result.sharpe_ratio:.2f}")
        print(f"\n  Weights:")
        for ticker, weight in sorted(zip(opt_result.tickers, opt_result.weights),
                                     key=lambda x: -x[1]):
            if weight > 0.001:
                print(f"    {ticker}: {weight:.2%}")

        # Store optimization result for later use
        self._optimal_weights = opt_result.weights
        self._optimal_tickers = opt_result.tickers

    def run_backtest(self) -> None:
        """Step 3: Backtest the optimal portfolio on historical data."""
        print("\n" + "=" * 60)
        print("STEP 3: Historical Backtest")
        print("=" * 60)

        returns = self.data_loader.calculate_returns()

        backtest_results = backtest_portfolio(
            weights=self._optimal_weights,
            returns=returns,
            tickers=self._optimal_tickers
        )

        metrics = calculate_backtest_metrics(backtest_results, self.risk_free_rate)

        # Convert daily returns to list for JSON
        daily_returns_list = [
            {
                'date': str(idx.date()),
                'daily_return': float(row['daily_return']),
                'cumulative_return': float(row['cumulative_return'])
            }
            for idx, row in backtest_results.iterrows()
        ]

        self.results['backtest'] = {
            'metrics': metrics,
            'daily_returns': daily_returns_list
        }

        print(f"\nBacktest Results:")
        print(f"  Period: {metrics['start_date']} to {metrics['end_date']}")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"  Annualized Volatility: {metrics['annualized_volatility']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")

    def run_stress_test(self,
                        scenario_config: Optional[StressScenarioConfig] = None,
                        n_simulations: int = 1000,
                        random_seed: int = 42) -> None:
        """
        Step 4: Run stress test simulation.

        Args:
            scenario_config: Custom scenario config (uses default tariff scenario if None)
            n_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
        """
        print("\n" + "=" * 60)
        print("STEP 4: Stress Test Simulation")
        print("=" * 60)

        # Get available Thai tickers
        available_tickers = self.data_loader.get_available_tickers()
        available_thai = [t for t in THAI_TICKERS if t in available_tickers]

        if scenario_config is None:
            scenario_config = create_tariff_scenario(
                start_date=datetime.now() + timedelta(days=1),
                duration_days=self.forward_horizon_days,
                thai_tickers=available_thai,
                expected_drop=0.10,
                drop_std=0.02,
                volatility_increase=0.10
            )

        print(f"\nScenario: {scenario_config.name}")
        print(f"  Affected tickers: {len(scenario_config.affected_tickers)} assets")
        print(f"  Expected drop: {scenario_config.expected_drop:.1%} ± {scenario_config.drop_std:.1%}")
        print(f"  Volatility increase: {scenario_config.volatility_increase:.1%}")
        print(f"  Forward horizon: {scenario_config.duration_days} days")
        print(f"  Simulations: {n_simulations}")

        # Initialize simulator
        returns = self.data_loader.calculate_returns()
        last_prices = self.data_loader.clean_data.iloc[-1]

        self.simulator = ScenarioSimulator(
            historical_returns=returns,
            last_prices=last_prices
        )

        # Run simulation
        sim_result = self.simulator.simulate_portfolio_returns(
            weights=self._optimal_weights,
            tickers=self._optimal_tickers,
            scenario=scenario_config,
            n_simulations=n_simulations,
            forward_horizon_days=self.forward_horizon_days,
            random_seed=random_seed
        )

        # Extract simulation path data (mean, percentiles)
        cumulative_returns = sim_result.portfolio_returns['cumulative']

        simulation_paths = []
        dates = cumulative_returns.index.tolist()

        for i, date in enumerate(dates):
            row_data = cumulative_returns.iloc[i].values
            simulation_paths.append({
                'date': str(date.date()),
                'mean_return': float(np.mean(row_data)),
                'std_return': float(np.std(row_data)),
                'percentile_5': float(np.percentile(row_data, 5)),
                'percentile_25': float(np.percentile(row_data, 25)),
                'percentile_50': float(np.percentile(row_data, 50)),
                'percentile_75': float(np.percentile(row_data, 75)),
                'percentile_95': float(np.percentile(row_data, 95))
            })

        self.results['stress_test'] = {
            'scenario': sim_result.to_dict()['scenario_config'],
            'portfolio_summary': sim_result.to_dict()['portfolio_returns_summary'],
            'statistics': sim_result.statistics,
            'simulation_paths': simulation_paths
        }

        print(f"\nStress Test Results:")
        print(f"  Mean 1Y Return: {sim_result.statistics['portfolio']['mean_total_return']:.2%}")
        print(f"  Std 1Y Return: {sim_result.statistics['portfolio']['std_total_return']:.2%}")
        print(f"  VaR 95%: {sim_result.statistics['portfolio']['var_95']:.2%}")
        print(f"  VaR 99%: {sim_result.statistics['portfolio']['var_99']:.2%}")
        print(f"  Probability of Loss: {sim_result.statistics['portfolio']['probability_loss']:.2%}")

    def run_full_pipeline(self,
                          scenario_config: Optional[StressScenarioConfig] = None,
                          n_simulations: int = 1000,
                          random_seed: int = 42) -> Dict:
        """
        Run the complete analysis pipeline.

        Args:
            scenario_config: Custom stress scenario (optional)
            n_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary with all results
        """
        self.run_data_loading()
        self.run_optimization()
        self.run_backtest()
        self.run_stress_test(scenario_config, n_simulations, random_seed)

        # Add metadata
        self.results['metadata'] = {
            'run_timestamp': datetime.now().isoformat(),
            'parameters': {
                'years_history': self.years_history,
                'forward_horizon_days': self.forward_horizon_days,
                'risk_free_rate': self.risk_free_rate,
                'n_simulations': n_simulations,
                'random_seed': random_seed
            }
        }

        return self.results

    def save_results(self, filepath: str = 'portfolio_analysis_results.json') -> None:
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to: {filepath}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Portfolio Optimization with Stress Testing'
    )
    parser.add_argument(
        '--output', '-o',
        default='portfolio_analysis_results.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--years', '-y',
        type=int,
        default=10,
        help='Years of historical data (default: 10)'
    )
    parser.add_argument(
        '--forward-days', '-f',
        type=int,
        default=252,
        help='Forward simulation days (default: 252)'
    )
    parser.add_argument(
        '--simulations', '-s',
        type=int,
        default=1000,
        help='Number of Monte Carlo simulations (default: 1000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--drop',
        type=float,
        default=0.10,
        help='Expected drop for Thai assets (default: 0.10)'
    )
    parser.add_argument(
        '--drop-std',
        type=float,
        default=0.02,
        help='Std deviation of drop (default: 0.02)'
    )
    parser.add_argument(
        '--vol-increase',
        type=float,
        default=0.10,
        help='Volatility increase for Thai assets (default: 0.10)'
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = PortfolioAnalysisPipeline(
        tickers=DEFAULT_TICKERS,
        years_history=args.years,
        forward_horizon_days=args.forward_days
    )

    # Create custom scenario if parameters differ from defaults
    available_tickers = None  # Will be determined after data loading

    # Run pipeline
    print("\n" + "=" * 60)
    print("PORTFOLIO OPTIMIZATION WITH STRESS TESTING")
    print("=" * 60)

    # Run data loading first to get available tickers
    pipeline.run_data_loading()

    # Get available Thai tickers for scenario
    available_thai = [t for t in THAI_TICKERS if t in pipeline.data_loader.get_available_tickers()]

    # Create scenario
    scenario = create_tariff_scenario(
        start_date=datetime.now() + timedelta(days=1),
        duration_days=args.forward_days,
        thai_tickers=available_thai,
        expected_drop=args.drop,
        drop_std=args.drop_std,
        volatility_increase=args.vol_increase
    )

    # Run remaining steps
    pipeline.run_optimization()
    pipeline.run_backtest()
    pipeline.run_stress_test(scenario, args.simulations, args.seed)

    # Add metadata
    pipeline.results['metadata'] = {
        'run_timestamp': datetime.now().isoformat(),
        'parameters': {
            'years_history': args.years,
            'forward_horizon_days': args.forward_days,
            'risk_free_rate': pipeline.risk_free_rate,
            'n_simulations': args.simulations,
            'random_seed': args.seed
        }
    }

    # Save results
    pipeline.save_results(args.output)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return pipeline.results


if __name__ == "__main__":
    results = main()
