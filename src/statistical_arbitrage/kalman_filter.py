"""
Kalman Filter Module

Implements Kalman filter for dynamic hedge ratio calculation in pairs trading.
Uses state-space models to adaptively estimate the relationship between assets.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pykalman import KalmanFilter
import warnings

warnings.filterwarnings('ignore')


class KalmanHedgeRatio:
    """Kalman filter implementation for dynamic hedge ratio estimation."""
    
    def __init__(self, delta: float = 1e-4, ve: float = 1e-3, vw: float = 1e-3):
        """
        Initialize Kalman filter for hedge ratio estimation.
        
        Args:
            delta: Transition covariance parameter
            ve: Observation noise variance
            vw: State transition noise variance
        """
        self.delta = delta
        self.ve = ve
        self.vw = vw
        self.hedge_ratios = None
        self.state_means = None
        self.state_covariances = None
        
    def fit(self, price1: pd.Series, price2: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit Kalman filter to estimate dynamic hedge ratio.
        
        Args:
            price1: Price series for asset 1 (dependent variable)
            price2: Price series for asset 2 (independent variable)
            
        Returns:
            Tuple of (hedge_ratios, state_covariances)
        """
        # Ensure both series have the same index
        common_index = price1.index.intersection(price2.index)
        y = price1.loc[common_index].values
        x = price2.loc[common_index].values
        
        n = len(y)
        
        # Set up state space representation
        # State: [beta, alpha] where y_t = alpha + beta * x_t + e_t
        
        # State transition matrix (random walk for parameters)
        transition_matrices = np.eye(2)
        
        # Observation matrix [x_t, 1]
        observation_matrices = np.column_stack([x, np.ones(n)])
        
        # Initial state (OLS estimates)
        initial_state_mean = np.array([np.polyfit(x, y, 1)[0], np.polyfit(x, y, 1)[1]])
        
        # Covariance matrices
        transition_covariance = self.delta * np.eye(2)
        observation_covariance = self.ve
        initial_state_covariance = np.eye(2)
        
        # Create and fit Kalman filter
        kf = KalmanFilter(
            transition_matrices=transition_matrices,
            observation_matrices=observation_matrices,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance
        )
        
        try:
            # Fit the model
            state_means, state_covariances = kf.em(y, n_iter=50).smooth()[0:2]
            
            # Extract hedge ratios (beta coefficients)
            hedge_ratios = state_means[:, 0]
            
            self.hedge_ratios = hedge_ratios
            self.state_means = state_means
            self.state_covariances = state_covariances
            
            return hedge_ratios, state_covariances
            
        except Exception as e:
            print(f"Kalman filter fitting failed: {e}")
            # Fallback to constant OLS estimate
            constant_ratio = np.full(n, initial_state_mean[0])
            constant_cov = np.tile(initial_state_covariance, (n, 1, 1))
            
            self.hedge_ratios = constant_ratio
            self.state_covariances = constant_cov
            
            return constant_ratio, constant_cov
    
    def predict_next_ratio(self) -> float:
        """
        Predict the next hedge ratio.
        
        Returns:
            Predicted hedge ratio
        """
        if self.state_means is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Return the last estimated ratio (random walk assumption)
        return self.state_means[-1, 0]
    
    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence intervals for hedge ratios.
        
        Args:
            confidence: Confidence level (default 0.95)
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        if self.hedge_ratios is None or self.state_covariances is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        from scipy.stats import norm
        
        # Calculate standard errors
        std_errors = np.sqrt(self.state_covariances[:, 0, 0])
        
        # Calculate z-score for confidence level
        alpha = 1 - confidence
        z_score = norm.ppf(1 - alpha / 2)
        
        # Calculate bounds
        lower_bounds = self.hedge_ratios - z_score * std_errors
        upper_bounds = self.hedge_ratios + z_score * std_errors
        
        return lower_bounds, upper_bounds


class StateSpaceModel:
    """
    State-space model for pairs trading with regime switching.
    Implements a more sophisticated model that can handle regime changes.
    """
    
    def __init__(self, n_regimes: int = 2):
        """
        Initialize state-space model.
        
        Args:
            n_regimes: Number of regimes for regime-switching model
        """
        self.n_regimes = n_regimes
        self.regime_probs = None
        self.regime_params = None
        
    def fit_regime_switching_model(self, price1: pd.Series, price2: pd.Series) -> dict:
        """
        Fit regime-switching model for dynamic relationship estimation.
        
        Args:
            price1: Price series for asset 1
            price2: Price series for asset 2
            
        Returns:
            Dictionary with model parameters and regime probabilities
        """
        # Simplified regime-switching implementation
        # In practice, this would use more sophisticated methods like Hamilton filter
        
        # Calculate spread using rolling OLS
        window = 60  # 60-day rolling window
        spreads = []
        hedge_ratios = []
        
        common_index = price1.index.intersection(price2.index)
        p1 = price1.loc[common_index]
        p2 = price2.loc[common_index]
        
        for i in range(window, len(p1)):
            # Rolling regression
            y_window = p1.iloc[i-window:i]
            x_window = p2.iloc[i-window:i]
            
            try:
                beta, alpha = np.polyfit(x_window, y_window, 1)
                spread = y_window.iloc[-1] - beta * x_window.iloc[-1]
                
                hedge_ratios.append(beta)
                spreads.append(spread)
            except:
                hedge_ratios.append(np.nan)
                spreads.append(np.nan)
        
        hedge_ratios = pd.Series(hedge_ratios, index=p1.index[window:])
        spreads = pd.Series(spreads, index=p1.index[window:])
        
        # Simple regime identification based on spread volatility
        rolling_vol = spreads.rolling(30).std()
        high_vol_regime = rolling_vol > rolling_vol.median()
        
        regime_probs = pd.DataFrame({
            'regime_0': ~high_vol_regime,
            'regime_1': high_vol_regime
        })
        
        self.regime_probs = regime_probs
        self.hedge_ratios = hedge_ratios
        
        return {
            'hedge_ratios': hedge_ratios,
            'spreads': spreads,
            'regime_probabilities': regime_probs,
            'rolling_volatility': rolling_vol
        }
    
    def get_current_regime(self) -> int:
        """
        Get the most likely current regime.
        
        Returns:
            Current regime index
        """
        if self.regime_probs is None:
            return 0
        
        # Return regime with highest probability
        latest_probs = self.regime_probs.iloc[-1]
        return latest_probs.idxmax().split('_')[1]


class AdaptiveKalmanFilter(KalmanHedgeRatio):
    """
    Adaptive Kalman filter that adjusts parameters based on market conditions.
    """
    
    def __init__(self, initial_delta: float = 1e-4, min_delta: float = 1e-6, 
                 max_delta: float = 1e-2):
        """
        Initialize adaptive Kalman filter.
        
        Args:
            initial_delta: Initial transition covariance parameter
            min_delta: Minimum allowed delta
            max_delta: Maximum allowed delta
        """
        super().__init__(delta=initial_delta)
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.adaptive_deltas = None
        
    def adapt_parameters(self, residuals: np.ndarray, window: int = 30) -> np.ndarray:
        """
        Adapt filter parameters based on recent performance.
        
        Args:
            residuals: Model residuals
            window: Window for adaptation
            
        Returns:
            Array of adaptive delta parameters
        """
        n = len(residuals)
        adaptive_deltas = np.full(n, self.delta)
        
        for i in range(window, n):
            # Calculate recent residual variance
            recent_var = np.var(residuals[i-window:i])
            
            # Adjust delta based on recent performance
            if recent_var > np.var(residuals[:i]):
                # Increase adaptability in volatile periods
                new_delta = min(self.delta * 2, self.max_delta)
            else:
                # Decrease adaptability in stable periods
                new_delta = max(self.delta * 0.5, self.min_delta)
            
            adaptive_deltas[i] = new_delta
        
        self.adaptive_deltas = adaptive_deltas
        return adaptive_deltas
    
    def fit_adaptive(self, price1: pd.Series, price2: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit adaptive Kalman filter with time-varying parameters.
        
        Args:
            price1: Price series for asset 1
            price2: Price series for asset 2
            
        Returns:
            Tuple of (hedge_ratios, state_covariances)
        """
        # First, fit standard Kalman filter
        hedge_ratios, state_covariances = self.fit(price1, price2)
        
        # Calculate residuals
        common_index = price1.index.intersection(price2.index)
        y = price1.loc[common_index].values
        x = price2.loc[common_index].values
        
        predicted_y = hedge_ratios * x + self.state_means[:, 1]
        residuals = y - predicted_y
        
        # Adapt parameters
        self.adapt_parameters(residuals)
        
        # Refit with adaptive parameters (simplified implementation)
        # In practice, this would involve running the filter with time-varying covariances
        
        return hedge_ratios, state_covariances