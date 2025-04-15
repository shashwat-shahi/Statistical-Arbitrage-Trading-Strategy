"""
Mean Reversion Strategy Module

Implements mean reversion trading strategies for pairs trading.
Generates trading signals based on spread deviation from mean.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class MeanReversionStrategy:
    """Implements mean reversion trading strategy for pairs trading."""
    
    def __init__(self, entry_threshold: float = 2.0, exit_threshold: float = 0.5,
                 stop_loss_threshold: float = 3.0, lookback_window: int = 60):
        """
        Initialize mean reversion strategy.
        
        Args:
            entry_threshold: Z-score threshold for entering positions
            exit_threshold: Z-score threshold for exiting positions
            stop_loss_threshold: Z-score threshold for stop-loss
            lookback_window: Window for calculating rolling statistics
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.lookback_window = lookback_window
        self.positions = None
        self.signals = None
        
    def calculate_spread(self, price1: pd.Series, price2: pd.Series, 
                        hedge_ratio: float) -> pd.Series:
        """
        Calculate spread between two assets.
        
        Args:
            price1: Price series for asset 1
            price2: Price series for asset 2
            hedge_ratio: Hedge ratio for the pair
            
        Returns:
            Spread series
        """
        return price1 - hedge_ratio * price2
    
    def calculate_z_score(self, spread: pd.Series, window: int = None) -> pd.Series:
        """
        Calculate rolling z-score of the spread.
        
        Args:
            spread: Spread series
            window: Rolling window size, defaults to lookback_window
            
        Returns:
            Z-score series
        """
        if window is None:
            window = self.lookback_window
            
        rolling_mean = spread.rolling(window=window, min_periods=int(window*0.7)).mean()
        rolling_std = spread.rolling(window=window, min_periods=int(window*0.7)).std()
        
        z_score = (spread - rolling_mean) / rolling_std
        return z_score
    
    def generate_signals(self, price1: pd.Series, price2: pd.Series, 
                        hedge_ratio: pd.Series = None) -> pd.DataFrame:
        """
        Generate trading signals based on mean reversion.
        
        Args:
            price1: Price series for asset 1
            price2: Price series for asset 2
            hedge_ratio: Dynamic hedge ratio series, if None uses constant ratio
            
        Returns:
            DataFrame with trading signals
        """
        # Ensure same index
        common_index = price1.index.intersection(price2.index)
        p1 = price1.loc[common_index]
        p2 = price2.loc[common_index]
        
        if hedge_ratio is None:
            # Use constant hedge ratio from OLS
            hr = np.polyfit(p2, p1, 1)[0]
            hedge_ratio = pd.Series([hr] * len(p1), index=common_index)
        elif isinstance(hedge_ratio, (int, float)):
            hedge_ratio = pd.Series([hedge_ratio] * len(p1), index=common_index)
        else:
            hedge_ratio = hedge_ratio.loc[common_index]
        
        # Calculate spread and z-score
        spread = p1 - hedge_ratio * p2
        z_score = self.calculate_z_score(spread)
        
        # Initialize signals
        signals = pd.DataFrame(index=common_index)
        signals['price1'] = p1
        signals['price2'] = p2
        signals['hedge_ratio'] = hedge_ratio
        signals['spread'] = spread
        signals['z_score'] = z_score
        signals['signal'] = 0  # 0: no position, 1: long spread, -1: short spread
        signals['position1'] = 0  # Position in asset 1
        signals['position2'] = 0  # Position in asset 2
        
        # Generate signals
        current_position = 0
        
        for i, (date, row) in enumerate(signals.iterrows()):
            z = row['z_score']
            
            if np.isnan(z):
                signals.loc[date, 'signal'] = current_position
                continue
            
            # Entry signals
            if current_position == 0:
                if z > self.entry_threshold:
                    # Spread is too high, short the spread (short asset1, long asset2)
                    current_position = -1
                elif z < -self.entry_threshold:
                    # Spread is too low, long the spread (long asset1, short asset2)
                    current_position = 1
            
            # Exit signals
            elif current_position != 0:
                # Stop loss
                if abs(z) > self.stop_loss_threshold:
                    current_position = 0
                # Mean reversion exit
                elif (current_position == 1 and z > -self.exit_threshold) or \
                     (current_position == -1 and z < self.exit_threshold):
                    current_position = 0
            
            signals.loc[date, 'signal'] = current_position
        
        # Calculate positions
        signals['position1'] = signals['signal']  # Position in asset 1
        signals['position2'] = -signals['signal'] * signals['hedge_ratio']  # Position in asset 2
        
        self.signals = signals
        return signals
    
    def calculate_returns(self, signals: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate strategy returns.
        
        Args:
            signals: Signals DataFrame, uses stored signals if None
            
        Returns:
            DataFrame with returns
        """
        if signals is None:
            if self.signals is None:
                raise ValueError("No signals available. Call generate_signals() first.")
            signals = self.signals
        
        # Calculate returns
        returns = signals.copy()
        
        # Asset returns
        returns['return1'] = signals['price1'].pct_change()
        returns['return2'] = signals['price2'].pct_change()
        
        # Position returns (lag positions by 1 to avoid look-ahead bias)
        position1_lag = signals['position1'].shift(1)
        position2_lag = signals['position2'].shift(1)
        
        # Strategy returns
        returns['strategy_return1'] = position1_lag * returns['return1']
        returns['strategy_return2'] = position2_lag * returns['return2']
        returns['total_strategy_return'] = (
            returns['strategy_return1'] + returns['strategy_return2']
        )
        
        # Cumulative returns
        returns['cumulative_return'] = (1 + returns['total_strategy_return']).cumprod()
        
        return returns
    
    def optimize_thresholds(self, price1: pd.Series, price2: pd.Series,
                           hedge_ratio: pd.Series = None,
                           threshold_range: Tuple[float, float] = (1.0, 3.0),
                           n_steps: int = 20) -> Dict[str, float]:
        """
        Optimize entry and exit thresholds using grid search.
        
        Args:
            price1: Price series for asset 1
            price2: Price series for asset 2
            hedge_ratio: Dynamic hedge ratio series
            threshold_range: Range for threshold optimization
            n_steps: Number of steps in grid search
            
        Returns:
            Dictionary with optimal thresholds and performance metrics
        """
        best_sharpe = -np.inf
        best_params = {}
        
        entry_thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)
        exit_thresholds = np.linspace(0.1, 1.0, n_steps // 2)
        
        results = []
        
        for entry_thresh in entry_thresholds:
            for exit_thresh in exit_thresholds:
                if exit_thresh >= entry_thresh:
                    continue
                
                # Temporarily set thresholds
                original_entry = self.entry_threshold
                original_exit = self.exit_threshold
                
                self.entry_threshold = entry_thresh
                self.exit_threshold = exit_thresh
                
                try:
                    # Generate signals and calculate returns
                    signals = self.generate_signals(price1, price2, hedge_ratio)
                    returns = self.calculate_returns(signals)
                    
                    # Calculate performance metrics
                    strategy_returns = returns['total_strategy_return'].dropna()
                    
                    if len(strategy_returns) > 0 and strategy_returns.std() > 0:
                        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                        total_return = returns['cumulative_return'].iloc[-1] - 1
                        
                        results.append({
                            'entry_threshold': entry_thresh,
                            'exit_threshold': exit_thresh,
                            'sharpe_ratio': sharpe_ratio,
                            'total_return': total_return,
                            'n_trades': (signals['signal'].diff() != 0).sum()
                        })
                        
                        if sharpe_ratio > best_sharpe:
                            best_sharpe = sharpe_ratio
                            best_params = {
                                'entry_threshold': entry_thresh,
                                'exit_threshold': exit_thresh,
                                'sharpe_ratio': sharpe_ratio,
                                'total_return': total_return
                            }
                
                except Exception:
                    continue
                
                # Restore original thresholds
                self.entry_threshold = original_entry
                self.exit_threshold = original_exit
        
        # Set optimal thresholds
        if best_params:
            self.entry_threshold = best_params['entry_threshold']
            self.exit_threshold = best_params['exit_threshold']
        
        return best_params
    
    def calculate_half_life_adaptive_threshold(self, spread: pd.Series) -> float:
        """
        Calculate adaptive threshold based on spread half-life.
        
        Args:
            spread: Spread series
            
        Returns:
            Adaptive threshold
        """
        try:
            # Calculate half-life
            spread_lag = spread.shift(1).dropna()
            spread_diff = spread.diff().dropna()
            
            # Align series
            common_index = spread_lag.index.intersection(spread_diff.index)
            spread_lag = spread_lag.loc[common_index]
            spread_diff = spread_diff.loc[common_index]
            
            if len(spread_lag) < 10:
                return self.entry_threshold
            
            # OLS regression for half-life
            X = np.column_stack([np.ones(len(spread_lag)), spread_lag])
            y = spread_diff.values
            
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            beta = coeffs[1]
            
            if beta >= 0:
                return self.entry_threshold
            
            half_life = -np.log(2) / np.log(1 + beta)
            
            # Adjust threshold based on half-life
            # Shorter half-life -> lower threshold (more sensitive)
            # Longer half-life -> higher threshold (less sensitive)
            if half_life < 5:
                return max(1.5, self.entry_threshold * 0.75)
            elif half_life > 20:
                return min(3.0, self.entry_threshold * 1.25)
            else:
                return self.entry_threshold
                
        except Exception:
            return self.entry_threshold


class BollingerBandStrategy(MeanReversionStrategy):
    """
    Bollinger Band based mean reversion strategy.
    Uses Bollinger Bands instead of z-score for signal generation.
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        """
        Initialize Bollinger Band strategy.
        
        Args:
            window: Rolling window for Bollinger Bands
            num_std: Number of standard deviations for bands
        """
        super().__init__()
        self.bb_window = window
        self.num_std = num_std
    
    def calculate_bollinger_bands(self, spread: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands for spread.
        
        Args:
            spread: Spread series
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle_band = spread.rolling(window=self.bb_window).mean()
        std = spread.rolling(window=self.bb_window).std()
        
        upper_band = middle_band + (std * self.num_std)
        lower_band = middle_band - (std * self.num_std)
        
        return upper_band, middle_band, lower_band
    
    def generate_bollinger_signals(self, price1: pd.Series, price2: pd.Series,
                                 hedge_ratio: pd.Series = None) -> pd.DataFrame:
        """
        Generate signals using Bollinger Bands.
        
        Args:
            price1: Price series for asset 1
            price2: Price series for asset 2
            hedge_ratio: Dynamic hedge ratio series
            
        Returns:
            DataFrame with Bollinger Band signals
        """
        # Ensure same index
        common_index = price1.index.intersection(price2.index)
        p1 = price1.loc[common_index]
        p2 = price2.loc[common_index]
        
        if hedge_ratio is None:
            hr = np.polyfit(p2, p1, 1)[0]
            hedge_ratio = pd.Series([hr] * len(p1), index=common_index)
        elif isinstance(hedge_ratio, (int, float)):
            hedge_ratio = pd.Series([hedge_ratio] * len(p1), index=common_index)
        else:
            hedge_ratio = hedge_ratio.loc[common_index]
        
        # Calculate spread
        spread = p1 - hedge_ratio * p2
        
        # Calculate Bollinger Bands
        upper_band, middle_band, lower_band = self.calculate_bollinger_bands(spread)
        
        # Initialize signals
        signals = pd.DataFrame(index=common_index)
        signals['price1'] = p1
        signals['price2'] = p2
        signals['hedge_ratio'] = hedge_ratio
        signals['spread'] = spread
        signals['upper_band'] = upper_band
        signals['middle_band'] = middle_band
        signals['lower_band'] = lower_band
        signals['signal'] = 0
        
        # Generate signals
        current_position = 0
        
        for i, (date, row) in enumerate(signals.iterrows()):
            spread_val = row['spread']
            upper = row['upper_band']
            lower = row['lower_band']
            middle = row['middle_band']
            
            if np.isnan(spread_val) or np.isnan(upper) or np.isnan(lower):
                signals.loc[date, 'signal'] = current_position
                continue
            
            # Entry signals
            if current_position == 0:
                if spread_val > upper:
                    # Spread above upper band, short the spread
                    current_position = -1
                elif spread_val < lower:
                    # Spread below lower band, long the spread
                    current_position = 1
            
            # Exit signals
            elif current_position != 0:
                if (current_position == 1 and spread_val >= middle) or \
                   (current_position == -1 and spread_val <= middle):
                    current_position = 0
            
            signals.loc[date, 'signal'] = current_position
        
        # Calculate positions
        signals['position1'] = signals['signal']
        signals['position2'] = -signals['signal'] * signals['hedge_ratio']
        
        return signals