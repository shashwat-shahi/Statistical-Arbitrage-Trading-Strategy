"""
Pairs Identification Module

Implements cointegration analysis for identifying trading pairs in statistical arbitrage.
Uses Engle-Granger and Johansen tests for cointegration detection.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class PairsIdentifier:
    """Identifies cointegrated pairs for statistical arbitrage trading."""
    
    def __init__(self, significance_level: float = 0.05, min_half_life: int = 1, 
                 max_half_life: int = 252):
        """
        Initialize PairsIdentifier.
        
        Args:
            significance_level: Statistical significance level for cointegration tests
            min_half_life: Minimum half-life in days for mean reversion
            max_half_life: Maximum half-life in days for mean reversion
        """
        self.significance_level = significance_level
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.cointegrated_pairs = []
        self.pairs_stats = {}
        
    def test_cointegration_engle_granger(self, price1: pd.Series, price2: pd.Series) -> Tuple[bool, float, float]:
        """
        Test cointegration using Engle-Granger method.
        
        Args:
            price1: Price series for first asset
            price2: Price series for second asset
            
        Returns:
            Tuple of (is_cointegrated, p_value, hedge_ratio)
        """
        try:
            # Ensure both series have the same index
            common_index = price1.index.intersection(price2.index)
            p1 = price1.loc[common_index].dropna()
            p2 = price2.loc[common_index].dropna()
            
            if len(p1) < 30 or len(p2) < 30:
                return False, 1.0, 0.0
            
            # Perform cointegration test
            score, p_value, _ = coint(p1, p2)
            
            # Calculate hedge ratio using OLS
            hedge_ratio = np.polyfit(p2, p1, 1)[0]
            
            is_cointegrated = p_value < self.significance_level
            
            return is_cointegrated, p_value, hedge_ratio
            
        except Exception:
            return False, 1.0, 0.0
    
    def test_cointegration_johansen(self, price1: pd.Series, price2: pd.Series) -> Tuple[bool, float]:
        """
        Test cointegration using Johansen method.
        
        Args:
            price1: Price series for first asset
            price2: Price series for second asset
            
        Returns:
            Tuple of (is_cointegrated, test_statistic)
        """
        try:
            # Ensure both series have the same index
            common_index = price1.index.intersection(price2.index)
            p1 = price1.loc[common_index].dropna()
            p2 = price2.loc[common_index].dropna()
            
            if len(p1) < 30 or len(p2) < 30:
                return False, 0.0
            
            # Create data matrix
            data = np.column_stack([p1, p2])
            
            # Perform Johansen test
            result = coint_johansen(data, det_order=0, k_ar_diff=1)
            
            # Check if trace statistic > critical value at significance level
            trace_stat = result.lr1[0]  # Test for at least one cointegrating relationship
            critical_value = result.cvt[0, 1]  # 5% critical value
            
            is_cointegrated = trace_stat > critical_value
            
            return is_cointegrated, trace_stat
            
        except Exception:
            return False, 0.0
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion for a spread.
        
        Args:
            spread: Price spread series
            
        Returns:
            Half-life in days
        """
        try:
            # Use AR(1) model to estimate half-life
            spread_lag = spread.shift(1).dropna()
            spread_diff = spread.diff().dropna()
            
            # Align the series
            common_index = spread_lag.index.intersection(spread_diff.index)
            spread_lag = spread_lag.loc[common_index]
            spread_diff = spread_diff.loc[common_index]
            
            if len(spread_lag) < 10:
                return np.inf
            
            # OLS regression: Δspread = α + β * spread_lag + ε
            X = np.column_stack([np.ones(len(spread_lag)), spread_lag])
            y = spread_diff.values
            
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            beta = coeffs[1]
            
            if beta >= 0:
                return np.inf
            
            # Half-life = -ln(2) / ln(1 + β)
            half_life = -np.log(2) / np.log(1 + beta)
            
            return half_life
            
        except Exception:
            return np.inf
    
    def calculate_spread_statistics(self, price1: pd.Series, price2: pd.Series, 
                                  hedge_ratio: float) -> Dict[str, float]:
        """
        Calculate spread statistics for a pair.
        
        Args:
            price1: Price series for first asset
            price2: Price series for second asset
            hedge_ratio: Hedge ratio for the pair
            
        Returns:
            Dictionary with spread statistics
        """
        try:
            # Calculate spread
            spread = price1 - hedge_ratio * price2
            
            # Calculate statistics
            stats_dict = {
                'mean': spread.mean(),
                'std': spread.std(),
                'min': spread.min(),
                'max': spread.max(),
                'half_life': self.calculate_half_life(spread),
                'adf_statistic': 0.0,
                'adf_p_value': 1.0
            }
            
            # Augmented Dickey-Fuller test for stationarity
            try:
                adf_result = adfuller(spread.dropna())
                stats_dict['adf_statistic'] = adf_result[0]
                stats_dict['adf_p_value'] = adf_result[1]
            except Exception:
                pass
            
            return stats_dict
            
        except Exception:
            return {}
    
    def find_pairs(self, price_data: pd.DataFrame, method: str = 'engle_granger') -> List[Dict]:
        """
        Find cointegrated pairs from price data.
        
        Args:
            price_data: DataFrame with price data for multiple assets
            method: Cointegration test method ('engle_granger' or 'johansen')
            
        Returns:
            List of dictionaries containing pair information
        """
        pairs = []
        tickers = price_data.columns.tolist()
        n_tickers = len(tickers)
        
        print(f"Testing {n_tickers * (n_tickers - 1) // 2} pairs for cointegration...")
        
        for i in range(n_tickers):
            for j in range(i + 1, n_tickers):
                ticker1, ticker2 = tickers[i], tickers[j]
                
                try:
                    price1 = price_data[ticker1]
                    price2 = price_data[ticker2]
                    
                    if method == 'engle_granger':
                        is_coint, p_value, hedge_ratio = self.test_cointegration_engle_granger(
                            price1, price2
                        )
                        
                        if is_coint:
                            # Calculate spread statistics
                            spread_stats = self.calculate_spread_statistics(
                                price1, price2, hedge_ratio
                            )
                            
                            # Check half-life constraint
                            half_life = spread_stats.get('half_life', np.inf)
                            
                            if self.min_half_life <= half_life <= self.max_half_life:
                                pair_info = {
                                    'ticker1': ticker1,
                                    'ticker2': ticker2,
                                    'hedge_ratio': hedge_ratio,
                                    'p_value': p_value,
                                    'half_life': half_life,
                                    'spread_mean': spread_stats.get('mean', 0),
                                    'spread_std': spread_stats.get('spread_std', 0),
                                    'adf_p_value': spread_stats.get('adf_p_value', 1)
                                }
                                pairs.append(pair_info)
                    
                    elif method == 'johansen':
                        is_coint, test_stat = self.test_cointegration_johansen(price1, price2)
                        
                        if is_coint:
                            # For Johansen, estimate hedge ratio using OLS
                            hedge_ratio = np.polyfit(price2.dropna(), price1.dropna(), 1)[0]
                            
                            spread_stats = self.calculate_spread_statistics(
                                price1, price2, hedge_ratio
                            )
                            
                            half_life = spread_stats.get('half_life', np.inf)
                            
                            if self.min_half_life <= half_life <= self.max_half_life:
                                pair_info = {
                                    'ticker1': ticker1,
                                    'ticker2': ticker2,
                                    'hedge_ratio': hedge_ratio,
                                    'test_statistic': test_stat,
                                    'half_life': half_life,
                                    'spread_mean': spread_stats.get('mean', 0),
                                    'spread_std': spread_stats.get('spread_std', 0),
                                    'adf_p_value': spread_stats.get('adf_p_value', 1)
                                }
                                pairs.append(pair_info)
                
                except Exception as e:
                    continue
        
        # Sort pairs by half-life (shorter is better for trading)
        pairs.sort(key=lambda x: x['half_life'])
        
        self.cointegrated_pairs = pairs
        print(f"Found {len(pairs)} cointegrated pairs")
        
        return pairs
    
    def get_top_pairs(self, n: int = 10) -> List[Dict]:
        """
        Get top N pairs by quality metrics.
        
        Args:
            n: Number of top pairs to return
            
        Returns:
            List of top pairs
        """
        if not self.cointegrated_pairs:
            return []
        
        # Sort by a combination of half-life and statistical significance
        scored_pairs = []
        for pair in self.cointegrated_pairs:
            # Score based on half-life (lower is better) and p-value (lower is better)
            half_life_score = 1 / max(pair['half_life'], 1)  # Avoid division by zero
            p_value_score = 1 - pair.get('p_value', pair.get('adf_p_value', 1))
            
            combined_score = half_life_score * p_value_score
            scored_pairs.append((combined_score, pair))
        
        # Sort by combined score (higher is better)
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        
        return [pair for _, pair in scored_pairs[:n]]
    
    def filter_pairs_by_correlation(self, price_data: pd.DataFrame, 
                                  max_correlation: float = 0.95) -> List[Dict]:
        """
        Filter pairs to avoid highly correlated assets.
        
        Args:
            price_data: DataFrame with price data
            max_correlation: Maximum allowed correlation
            
        Returns:
            Filtered list of pairs
        """
        if not self.cointegrated_pairs:
            return []
        
        # Calculate correlation matrix
        returns = price_data.pct_change().dropna()
        corr_matrix = returns.corr()
        
        filtered_pairs = []
        for pair in self.cointegrated_pairs:
            ticker1, ticker2 = pair['ticker1'], pair['ticker2']
            
            if ticker1 in corr_matrix.columns and ticker2 in corr_matrix.columns:
                correlation = abs(corr_matrix.loc[ticker1, ticker2])
                
                if correlation <= max_correlation:
                    pair['correlation'] = correlation
                    filtered_pairs.append(pair)
        
        return filtered_pairs