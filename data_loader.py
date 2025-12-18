"""
Data Loader Module
Downloads and cleans financial data from Yahoo Finance.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple, Optional


class DataLoader:
    """Handles downloading and cleaning of financial data from Yahoo Finance."""

    def __init__(self, tickers: List[str], years_history: int = 10):
        """
        Initialize the data loader.

        Args:
            tickers: List of ticker symbols to download
            years_history: Number of years of historical data to fetch
        """
        self.tickers = tickers
        self.years_history = years_history
        self.raw_data: Optional[pd.DataFrame] = None
        self.clean_data: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None

    def download_data(self, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Download price data from Yahoo Finance.

        Args:
            end_date: End date for data download (defaults to today)

        Returns:
            DataFrame with adjusted close prices
        """
        if end_date is None:
            end_date = datetime.now()

        start_date = end_date - timedelta(days=self.years_history * 365)

        print(f"Downloading data from {start_date.date()} to {end_date.date()}...")

        # Download data for all tickers
        data = yf.download(
            self.tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )

        # Extract adjusted close prices
        if len(self.tickers) > 1:
            prices = data['Close']
        else:
            prices = data['Close'].to_frame(self.tickers[0])

        self.raw_data = prices
        print(f"Downloaded {len(prices)} days of data for {len(self.tickers)} tickers")

        return prices

    def clean_data_pipeline(self,
                            outlier_std_threshold: float = 4.0,
                            min_data_coverage: float = 0.8) -> pd.DataFrame:
        """
        Clean the downloaded data by removing NAs and outliers.

        Args:
            outlier_std_threshold: Number of standard deviations for outlier detection
            min_data_coverage: Minimum fraction of non-NA data required for a ticker

        Returns:
            Cleaned DataFrame with prices
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call download_data() first.")

        prices = self.raw_data.copy()

        # Step 1: Remove tickers with too many missing values
        coverage = prices.notna().sum() / len(prices)
        valid_tickers = coverage[coverage >= min_data_coverage].index.tolist()
        removed_tickers = set(prices.columns) - set(valid_tickers)

        if removed_tickers:
            print(f"Removing tickers with low coverage: {removed_tickers}")

        prices = prices[valid_tickers]

        # Step 2: Forward fill missing values (for small gaps)
        prices = prices.ffill(limit=5)

        # Step 3: Drop remaining rows with NA values
        initial_rows = len(prices)
        prices = prices.dropna()
        removed_rows = initial_rows - len(prices)

        if removed_rows > 0:
            print(f"Removed {removed_rows} rows with missing data")

        # Step 4: Calculate returns for outlier detection
        returns = prices.pct_change().dropna()

        # Step 5: Remove outliers based on z-score
        outliers_removed = 0
        for col in returns.columns:
            mean = returns[col].mean()
            std = returns[col].std()

            # Identify outlier indices
            outlier_mask = np.abs(returns[col] - mean) > outlier_std_threshold * std
            outlier_indices = returns[outlier_mask].index

            # Replace outlier returns with the median
            if len(outlier_indices) > 0:
                median_return = returns[col].median()
                returns.loc[outlier_indices, col] = median_return
                outliers_removed += len(outlier_indices)

        if outliers_removed > 0:
            print(f"Replaced {outliers_removed} outlier returns with median values")

        # Reconstruct prices from cleaned returns
        initial_prices = prices.iloc[0]
        clean_prices = (1 + returns).cumprod() * initial_prices
        clean_prices = pd.concat([prices.iloc[[0]], clean_prices])

        self.clean_data = clean_prices
        self.returns = returns

        print(f"Clean data: {len(clean_prices)} days, {len(clean_prices.columns)} tickers")

        return clean_prices

    def calculate_returns(self) -> pd.DataFrame:
        """
        Calculate daily returns from price data.

        Returns:
            DataFrame with daily returns
        """
        if self.clean_data is None:
            raise ValueError("No clean data available. Run clean_data_pipeline() first.")

        if self.returns is None:
            self.returns = self.clean_data.pct_change().dropna()

        return self.returns

    def get_statistics(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculate mean returns and covariance matrix.

        Returns:
            Tuple of (annualized mean returns, annualized covariance matrix)
        """
        returns = self.calculate_returns()

        # Annualize statistics (252 trading days)
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        return mean_returns, cov_matrix

    def get_available_tickers(self) -> List[str]:
        """Return list of tickers that have valid data after cleaning."""
        if self.clean_data is None:
            raise ValueError("No clean data available. Run clean_data_pipeline() first.")
        return self.clean_data.columns.tolist()


def load_and_prepare_data(tickers: List[str],
                          years_history: int = 10,
                          outlier_std_threshold: float = 4.0,
                          min_data_coverage: float = 0.8) -> DataLoader:
    """
    Convenience function to load and prepare data in one step.

    Args:
        tickers: List of ticker symbols
        years_history: Number of years of historical data
        outlier_std_threshold: Threshold for outlier detection
        min_data_coverage: Minimum data coverage required

    Returns:
        Configured DataLoader instance with clean data
    """
    loader = DataLoader(tickers, years_history)
    loader.download_data()
    loader.clean_data_pipeline(outlier_std_threshold, min_data_coverage)

    return loader


if __name__ == "__main__":
    # Test the data loader
    tickers = [
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

    loader = load_and_prepare_data(tickers)
    mean_returns, cov_matrix = loader.get_statistics()

    print("\nAnnualized Mean Returns:")
    print(mean_returns)
    print("\nCovariance Matrix Shape:", cov_matrix.shape)
