"""
Portfolio Optimizer Module
Implements mean-variance portfolio optimization using scipy.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    weights: np.ndarray
    tickers: List[str]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    optimization_status: str

    def to_dict(self) -> Dict:
        """Convert to dictionary with ticker-weight mapping."""
        return {
            'weights': {ticker: float(weight) for ticker, weight in zip(self.tickers, self.weights)},
            'expected_return': float(self.expected_return),
            'volatility': float(self.volatility),
            'sharpe_ratio': float(self.sharpe_ratio),
            'status': self.optimization_status
        }


class PortfolioOptimizer:
    """
    Mean-variance portfolio optimizer using scipy.

    Supports:
    - Maximize Sharpe ratio
    - Minimize volatility for target return
    - Maximize return for target volatility
    """

    def __init__(self,
                 mean_returns: pd.Series,
                 cov_matrix: pd.DataFrame,
                 risk_free_rate: float = 0.02):
        """
        Initialize the optimizer.

        Args:
            mean_returns: Annualized expected returns for each asset
            cov_matrix: Annualized covariance matrix
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.tickers = mean_returns.index.tolist()
        self.n_assets = len(self.tickers)

        # Convert to numpy for faster computation
        self._returns = mean_returns.values
        self._cov = cov_matrix.values

    def _portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate expected portfolio return."""
        return np.dot(weights, self._returns)

    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility (standard deviation)."""
        return np.sqrt(np.dot(weights.T, np.dot(self._cov, weights)))

    def _sharpe_ratio(self, weights: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        port_return = self._portfolio_return(weights)
        port_vol = self._portfolio_volatility(weights)
        return (port_return - self.risk_free_rate) / port_vol

    def _negative_sharpe(self, weights: np.ndarray) -> float:
        """Negative Sharpe ratio for minimization."""
        return -self._sharpe_ratio(weights)

    def optimize_sharpe(self,
                        allow_short: bool = False,
                        max_weight: float = 1.0) -> OptimizationResult:
        """
        Find the portfolio that maximizes the Sharpe ratio.

        Args:
            allow_short: Whether to allow short selling (default: False)
            max_weight: Maximum weight for any single asset

        Returns:
            OptimizationResult with optimal weights and metrics
        """
        # Initial guess: equal weights
        init_weights = np.ones(self.n_assets) / self.n_assets

        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        # Bounds
        if allow_short:
            bounds = tuple((-1, max_weight) for _ in range(self.n_assets))
        else:
            bounds = tuple((0, max_weight) for _ in range(self.n_assets))

        # Optimize
        result = minimize(
            self._negative_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )

        # Extract results
        optimal_weights = result.x
        port_return = self._portfolio_return(optimal_weights)
        port_vol = self._portfolio_volatility(optimal_weights)
        sharpe = self._sharpe_ratio(optimal_weights)

        status = 'success' if result.success else f'failed: {result.message}'

        return OptimizationResult(
            weights=optimal_weights,
            tickers=self.tickers,
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            optimization_status=status
        )

    def optimize_min_volatility(self,
                                target_return: Optional[float] = None,
                                allow_short: bool = False,
                                max_weight: float = 1.0) -> OptimizationResult:
        """
        Find the minimum volatility portfolio, optionally with a target return.

        Args:
            target_return: Target annualized return (optional)
            allow_short: Whether to allow short selling
            max_weight: Maximum weight for any single asset

        Returns:
            OptimizationResult with optimal weights and metrics
        """
        init_weights = np.ones(self.n_assets) / self.n_assets

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: self._portfolio_return(w) - target_return
            })

        if allow_short:
            bounds = tuple((-1, max_weight) for _ in range(self.n_assets))
        else:
            bounds = tuple((0, max_weight) for _ in range(self.n_assets))

        result = minimize(
            self._portfolio_volatility,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )

        optimal_weights = result.x
        port_return = self._portfolio_return(optimal_weights)
        port_vol = self._portfolio_volatility(optimal_weights)
        sharpe = self._sharpe_ratio(optimal_weights)

        status = 'success' if result.success else f'failed: {result.message}'

        return OptimizationResult(
            weights=optimal_weights,
            tickers=self.tickers,
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            optimization_status=status
        )

    def get_efficient_frontier(self,
                               n_points: int = 50,
                               allow_short: bool = False) -> pd.DataFrame:
        """
        Calculate the efficient frontier.

        Args:
            n_points: Number of points on the frontier
            allow_short: Whether to allow short selling

        Returns:
            DataFrame with return, volatility, and weights for each point
        """
        # Find return bounds
        min_vol_result = self.optimize_min_volatility(allow_short=allow_short)
        max_sharpe_result = self.optimize_sharpe(allow_short=allow_short)

        min_return = min(min_vol_result.expected_return, self._returns.min())
        max_return = max(max_sharpe_result.expected_return, self._returns.max())

        target_returns = np.linspace(min_return, max_return, n_points)

        frontier_data = []
        for target_ret in target_returns:
            try:
                result = self.optimize_min_volatility(
                    target_return=target_ret,
                    allow_short=allow_short
                )
                if result.optimization_status == 'success':
                    frontier_data.append({
                        'return': result.expected_return,
                        'volatility': result.volatility,
                        'sharpe': result.sharpe_ratio,
                        'weights': result.weights
                    })
            except Exception:
                continue

        return pd.DataFrame(frontier_data)


def backtest_portfolio(weights: np.ndarray,
                       returns: pd.DataFrame,
                       tickers: List[str]) -> pd.DataFrame:
    """
    Backtest a portfolio with given weights on historical returns.

    Args:
        weights: Portfolio weights
        returns: DataFrame of daily returns
        tickers: List of ticker symbols corresponding to weights

    Returns:
        DataFrame with daily portfolio returns and cumulative returns
    """
    # Align returns with weights
    aligned_returns = returns[tickers]

    # Calculate portfolio daily returns
    portfolio_returns = (aligned_returns * weights).sum(axis=1)

    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1

    result = pd.DataFrame({
        'daily_return': portfolio_returns,
        'cumulative_return': cumulative_returns
    })

    return result


def calculate_backtest_metrics(backtest_results: pd.DataFrame,
                               risk_free_rate: float = 0.02) -> Dict:
    """
    Calculate performance metrics from backtest results.

    Args:
        backtest_results: DataFrame from backtest_portfolio
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary of performance metrics
    """
    daily_returns = backtest_results['daily_return']

    # Annualized metrics
    annualized_return = daily_returns.mean() * 252
    annualized_vol = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol

    # Drawdown analysis
    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Total return
    total_return = backtest_results['cumulative_return'].iloc[-1]

    return {
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'annualized_volatility': float(annualized_vol),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'start_date': str(backtest_results.index[0].date()),
        'end_date': str(backtest_results.index[-1].date()),
        'n_days': len(backtest_results)
    }


if __name__ == "__main__":
    # Test the optimizer with sample data
    np.random.seed(42)

    # Generate sample data
    tickers = ['ASSET_A', 'ASSET_B', 'ASSET_C', 'ASSET_D']
    n_assets = len(tickers)

    # Random mean returns (annualized)
    mean_returns = pd.Series(
        np.random.uniform(0.05, 0.15, n_assets),
        index=tickers
    )

    # Random covariance matrix (make it positive definite)
    random_matrix = np.random.randn(n_assets, n_assets)
    cov_matrix = pd.DataFrame(
        np.dot(random_matrix, random_matrix.T) * 0.04,
        index=tickers,
        columns=tickers
    )

    # Test optimization
    optimizer = PortfolioOptimizer(mean_returns, cov_matrix)

    print("Max Sharpe Portfolio (no shorting):")
    result = optimizer.optimize_sharpe(allow_short=False)
    print(f"  Weights: {dict(zip(result.tickers, np.round(result.weights, 4)))}")
    print(f"  Expected Return: {result.expected_return:.2%}")
    print(f"  Volatility: {result.volatility:.2%}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")

    print("\nMin Volatility Portfolio:")
    result = optimizer.optimize_min_volatility(allow_short=False)
    print(f"  Weights: {dict(zip(result.tickers, np.round(result.weights, 4)))}")
    print(f"  Volatility: {result.volatility:.2%}")
