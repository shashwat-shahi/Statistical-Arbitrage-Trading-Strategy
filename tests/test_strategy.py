"""
Test suite for Statistical Arbitrage Trading Strategy

Basic tests to verify functionality of core components.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    from statistical_arbitrage.data_manager import DataManager
    from statistical_arbitrage.pairs_identification import PairsIdentifier
    from statistical_arbitrage.kalman_filter import KalmanHedgeRatio
    from statistical_arbitrage.mean_reversion import MeanReversionStrategy
    from statistical_arbitrage.backtesting import BacktestEngine
    from statistical_arbitrage.risk_management import RiskManager
    from statistical_arbitrage.performance_metrics import PerformanceAnalyzer
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    IMPORTS_AVAILABLE = False


class TestDataManager(unittest.TestCase):
    """Test DataManager functionality."""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        self.data_manager = DataManager(start_date="2023-01-01", end_date="2023-03-31")
    
    def test_sp500_tickers(self):
        """Test that S&P 500 tickers are loaded."""
        tickers = self.data_manager._get_sp500_tickers()
        self.assertIsInstance(tickers, list)
        self.assertGreater(len(tickers), 0)
        self.assertIn('AAPL', tickers)
        self.assertIn('MSFT', tickers)
    
    def test_sector_mapping(self):
        """Test sector mapping functionality."""
        # Create dummy price data
        dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')
        dummy_data = pd.DataFrame({
            'AAPL': np.random.randn(len(dates)).cumsum() + 100,
            'MSFT': np.random.randn(len(dates)).cumsum() + 200
        }, index=dates)
        self.data_manager.price_data = dummy_data
        
        sector_map = self.data_manager.get_sector_mapping()
        self.assertIsInstance(sector_map, dict)
        self.assertIn('AAPL', sector_map)


class TestPairsIdentification(unittest.TestCase):
    """Test pairs identification functionality."""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        self.pairs_identifier = PairsIdentifier(significance_level=0.05)
        
        # Create synthetic cointegrated data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        n = len(dates)
        
        # Generate cointegrated series
        noise1 = np.random.normal(0, 0.01, n)
        noise2 = np.random.normal(0, 0.01, n)
        
        price1 = 100 + np.cumsum(noise1)
        price2 = 50 + 0.5 * price1 + np.cumsum(noise2 * 0.1)  # Cointegrated with price1
        price3 = 75 + np.cumsum(np.random.normal(0, 0.02, n))  # Independent
        
        self.price_data = pd.DataFrame({
            'STOCK_A': price1,
            'STOCK_B': price2,
            'STOCK_C': price3
        }, index=dates)
    
    def test_cointegration_test(self):
        """Test cointegration testing."""
        price1 = self.price_data['STOCK_A']
        price2 = self.price_data['STOCK_B']
        
        is_coint, p_value, hedge_ratio = self.pairs_identifier.test_cointegration_engle_granger(
            price1, price2
        )
        
        self.assertIsInstance(is_coint, bool)
        self.assertIsInstance(p_value, float)
        self.assertIsInstance(hedge_ratio, float)
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)
    
    def test_half_life_calculation(self):
        """Test half-life calculation."""
        # Create mean-reverting series
        spread = np.random.normal(0, 1, 100)
        spread = pd.Series(spread).cumsum() * 0.1  # Add some persistence
        
        half_life = self.pairs_identifier.calculate_half_life(spread)
        self.assertIsInstance(half_life, float)
        self.assertGreater(half_life, 0)


class TestMeanReversionStrategy(unittest.TestCase):
    """Test mean reversion strategy."""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        self.strategy = MeanReversionStrategy(
            entry_threshold=2.0,
            exit_threshold=0.5,
            lookback_window=30
        )
        
        # Create synthetic price data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
        n = len(dates)
        
        price1 = 100 + np.cumsum(np.random.normal(0, 0.01, n))
        price2 = 50 + 0.5 * price1 + np.random.normal(0, 0.5, n)
        
        self.price1 = pd.Series(price1, index=dates)
        self.price2 = pd.Series(price2, index=dates)
    
    def test_spread_calculation(self):
        """Test spread calculation."""
        hedge_ratio = 0.5
        spread = self.strategy.calculate_spread(self.price1, self.price2, hedge_ratio)
        
        self.assertIsInstance(spread, pd.Series)
        self.assertEqual(len(spread), len(self.price1))
    
    def test_signal_generation(self):
        """Test signal generation."""
        signals = self.strategy.generate_signals(self.price1, self.price2)
        
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('signal', signals.columns)
        self.assertIn('position1', signals.columns)
        self.assertIn('position2', signals.columns)
        
        # Check signal values are valid
        unique_signals = signals['signal'].unique()
        for signal in unique_signals:
            self.assertIn(signal, [-1, 0, 1])


class TestBacktestEngine(unittest.TestCase):
    """Test backtesting engine."""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        self.backtest_engine = BacktestEngine(
            initial_capital=100000,
            transaction_cost=0.001
        )
        
        # Create sample signals data
        dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')
        n = len(dates)
        
        self.signals = pd.DataFrame({
            'price1': 100 + np.random.randn(n).cumsum() * 0.5,
            'price2': 50 + np.random.randn(n).cumsum() * 0.3,
            'hedge_ratio': [0.5] * n,
            'signal': np.random.choice([-1, 0, 1], n),
            'spread': np.random.randn(n),
            'z_score': np.random.randn(n)
        }, index=dates)
    
    def test_position_sizing(self):
        """Test position sizing calculation."""
        shares1, shares2 = self.backtest_engine.calculate_position_size(
            signal=1, price1=100, price2=50, hedge_ratio=0.5, current_capital=100000
        )
        
        self.assertIsInstance(shares1, int)
        self.assertIsInstance(shares2, int)
    
    def test_transaction_costs(self):
        """Test transaction cost calculation."""
        costs = self.backtest_engine.calculate_transaction_costs(
            shares_traded1=100, shares_traded2=200, price1=100, price2=50
        )
        
        self.assertIsInstance(costs, float)
        self.assertGreaterEqual(costs, 0)


class TestPerformanceAnalyzer(unittest.TestCase):
    """Test performance analysis."""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        self.analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
        
        # Create sample returns
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns
        self.portfolio_values = (1 + self.returns).cumprod() * 100000
    
    def test_basic_metrics(self):
        """Test basic performance metrics calculation."""
        metrics = self.analyzer.calculate_basic_metrics(self.returns)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('Sharpe Ratio', metrics)
        self.assertIn('Total Return', metrics)
        self.assertIn('Volatility', metrics)
        
        # Check metric ranges
        self.assertIsInstance(metrics['Sharpe Ratio'], float)
        self.assertGreaterEqual(metrics['Win Rate'], 0)
        self.assertLessEqual(metrics['Win Rate'], 1)
    
    def test_drawdown_metrics(self):
        """Test drawdown calculation."""
        metrics = self.analyzer.calculate_drawdown_metrics(self.portfolio_values)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('Max Drawdown', metrics)
        self.assertIn('Calmar Ratio', metrics)
        
        # Drawdown should be negative or zero
        self.assertLessEqual(metrics['Max Drawdown'], 0)


def run_integration_test():
    """Run a simple integration test of the complete strategy."""
    if not IMPORTS_AVAILABLE:
        print("Skipping integration test - imports not available")
        return
    
    print("Running integration test...")
    
    try:
        # Create synthetic data
        dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')
        n = len(dates)
        
        np.random.seed(42)
        price_data = pd.DataFrame({
            'AAPL': 150 + np.cumsum(np.random.normal(0, 0.02, n)),
            'MSFT': 250 + np.cumsum(np.random.normal(0, 0.02, n))
        }, index=dates)
        
        # Test pairs identification
        pairs_finder = PairsIdentifier()
        pairs = pairs_finder.find_pairs(price_data)
        
        if pairs:
            print(f"✓ Found {len(pairs)} pairs")
            
            # Test strategy
            strategy = MeanReversionStrategy()
            signals = strategy.generate_signals(
                price_data['AAPL'], price_data['MSFT']
            )
            
            print(f"✓ Generated {len(signals)} signals")
            
            # Test backtest
            backtest = BacktestEngine(initial_capital=100000)
            results = backtest.run_backtest(signals)
            
            print(f"✓ Backtest completed")
            
            # Test performance analysis
            analyzer = PerformanceAnalyzer()
            portfolio_values = results['portfolio']['portfolio_value']
            portfolio_returns = results['portfolio']['portfolio_return'].dropna()
            
            if len(portfolio_returns) > 0:
                metrics = analyzer.calculate_basic_metrics(portfolio_returns)
                print(f"✓ Calculated performance metrics")
                print(f"  Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}")
                print(f"  Total Return: {metrics.get('Total Return', 0):.2%}")
            
            print("✅ Integration test passed!")
        else:
            print("⚠ No pairs found in synthetic data")
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")


if __name__ == '__main__':
    # Run unit tests
    if IMPORTS_AVAILABLE:
        unittest.main(argv=[''], exit=False, verbosity=2)
        
        # Run integration test
        print("\n" + "="*50)
        run_integration_test()
    else:
        print("Cannot run tests - missing dependencies")
        print("Please install requirements: pip install -r requirements.txt")