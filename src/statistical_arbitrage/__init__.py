"""
Statistical Arbitrage Trading Strategy

This package implements a pairs trading algorithm using cointegration analysis
and mean reversion strategies with Kalman filters for dynamic hedge ratio calculation.
"""

__version__ = "1.0.0"
__author__ = "Shashwat Shahi"

# Import main classes for easy access
try:
    from .data_manager import DataManager
    from .pairs_identification import PairsIdentifier
    from .mean_reversion import MeanReversionStrategy
    from .kalman_filter import KalmanHedgeRatio, AdaptiveKalmanFilter
    from .backtesting import BacktestEngine
    from .risk_management import RiskManager
    from .performance_metrics import PerformanceAnalyzer
except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings
    warnings.warn(f"Some dependencies missing: {e}. Please install requirements.txt")