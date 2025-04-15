"""
Data Manager Module

Handles data acquisition, preprocessing, and management for statistical arbitrage trading.
Fetches S&P 500 constituent data and prepares it for analysis.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class DataManager:
    """Manages data acquisition and preprocessing for statistical arbitrage trading."""
    
    def __init__(self, start_date: str = "2020-01-01", end_date: str = None):
        """
        Initialize DataManager.
        
        Args:
            start_date: Start date for data fetching (YYYY-MM-DD)
            end_date: End date for data fetching (YYYY-MM-DD), defaults to today
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.sp500_tickers = self._get_sp500_tickers()
        self.price_data = None
        self.returns_data = None
        
    def _get_sp500_tickers(self) -> List[str]:
        """
        Get S&P 500 constituent tickers.
        
        Returns:
            List of S&P 500 ticker symbols
        """
        # Common S&P 500 tickers for demonstration
        # In production, this would fetch from a live source
        return [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'META', 'NVDA', 'BRK-B',
            'UNH', 'JNJ', 'JPM', 'V', 'PG', 'HD', 'CVX', 'MA', 'BAC', 'ABBV',
            'PFE', 'AVGO', 'KO', 'MRK', 'COST', 'DIS', 'TMO', 'DHR', 'WMT',
            'VZ', 'ABT', 'ADBE', 'CRM', 'NFLX', 'XOM', 'NKE', 'ACN', 'INTC'
        ]
    
    def fetch_data(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch price data for specified tickers.
        
        Args:
            tickers: List of ticker symbols, defaults to S&P 500 subset
            
        Returns:
            DataFrame with adjusted close prices
        """
        if tickers is None:
            tickers = self.sp500_tickers[:20]  # Use subset for faster processing
            
        print(f"Fetching data for {len(tickers)} tickers from {self.start_date} to {self.end_date}")
        
        try:
            data = yf.download(tickers, start=self.start_date, end=self.end_date, 
                             progress=False, threads=True)
            
            if len(tickers) == 1:
                self.price_data = pd.DataFrame({tickers[0]: data['Adj Close']})
            else:
                self.price_data = data['Adj Close']
                
            # Remove tickers with insufficient data
            min_data_points = 252  # Approximately 1 year of trading days
            valid_tickers = self.price_data.columns[
                self.price_data.count() >= min_data_points
            ]
            self.price_data = self.price_data[valid_tickers]
            
            # Forward fill and backward fill missing values
            self.price_data = self.price_data.fillna(method='ffill').fillna(method='bfill')
            
            print(f"Successfully fetched data for {len(self.price_data.columns)} tickers")
            return self.price_data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def calculate_returns(self, method: str = 'log') -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            method: 'log' for log returns or 'simple' for simple returns
            
        Returns:
            DataFrame with returns
        """
        if self.price_data is None:
            raise ValueError("Price data not available. Call fetch_data() first.")
            
        if method == 'log':
            self.returns_data = np.log(self.price_data / self.price_data.shift(1))
        elif method == 'simple':
            self.returns_data = self.price_data.pct_change()
        else:
            raise ValueError("Method must be 'log' or 'simple'")
            
        # Remove first row with NaN values
        self.returns_data = self.returns_data.dropna()
        
        return self.returns_data
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix of returns.
        
        Returns:
            Correlation matrix
        """
        if self.returns_data is None:
            self.calculate_returns()
            
        return self.returns_data.corr()
    
    def get_sector_mapping(self) -> Dict[str, str]:
        """
        Get sector mapping for tickers (simplified version).
        
        Returns:
            Dictionary mapping tickers to sectors
        """
        # Simplified sector mapping for demonstration
        sector_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'META': 'Technology', 'NVDA': 'Technology', 'ADBE': 'Technology',
            'CRM': 'Technology', 'INTC': 'Technology', 'NFLX': 'Technology',
            'JPM': 'Financials', 'BAC': 'Financials', 'V': 'Financials',
            'MA': 'Financials', 'UNH': 'Healthcare', 'JNJ': 'Healthcare',
            'PFE': 'Healthcare', 'ABBV': 'Healthcare', 'ABT': 'Healthcare',
            'TMO': 'Healthcare', 'DHR': 'Healthcare', 'PG': 'Consumer Goods',
            'KO': 'Consumer Goods', 'WMT': 'Consumer Goods', 'COST': 'Consumer Goods',
            'HD': 'Consumer Goods', 'NKE': 'Consumer Goods', 'DIS': 'Consumer Goods'
        }
        
        return {ticker: sector_map.get(ticker, 'Other') 
                for ticker in self.price_data.columns}
    
    def filter_by_sector(self, sector: str) -> List[str]:
        """
        Filter tickers by sector.
        
        Args:
            sector: Sector name to filter by
            
        Returns:
            List of tickers in the specified sector
        """
        sector_mapping = self.get_sector_mapping()
        return [ticker for ticker, sec in sector_mapping.items() if sec == sector]
    
    def get_price_data(self) -> pd.DataFrame:
        """Get the stored price data."""
        return self.price_data
    
    def get_returns_data(self) -> pd.DataFrame:
        """Get the stored returns data."""
        return self.returns_data