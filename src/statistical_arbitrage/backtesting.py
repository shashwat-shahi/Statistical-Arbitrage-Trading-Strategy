"""
Backtesting Framework Module

Comprehensive backtesting framework with transaction cost modeling
and risk management constraints for statistical arbitrage strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import warnings

warnings.filterwarnings('ignore')


class BacktestEngine:
    """Comprehensive backtesting engine for statistical arbitrage strategies."""
    
    def __init__(self, initial_capital: float = 100000, 
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005,
                 max_position_size: float = 0.25,
                 max_leverage: float = 2.0,
                 commission_per_share: float = 0.005):
        """
        Initialize backtesting engine.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction of trade value
            slippage: Market impact/slippage as fraction of trade value
            max_position_size: Maximum position size as fraction of portfolio
            max_leverage: Maximum allowed leverage
            commission_per_share: Fixed commission per share
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.commission_per_share = commission_per_share
        
        # Results storage
        self.portfolio_values = None
        self.positions = None
        self.trades = None
        self.performance_metrics = None
        
    def calculate_position_size(self, signal: float, price1: float, price2: float,
                               hedge_ratio: float, current_capital: float) -> Tuple[int, int]:
        """
        Calculate position sizes based on signal and risk constraints.
        
        Args:
            signal: Trading signal (-1, 0, 1)
            price1: Current price of asset 1
            price2: Current price of asset 2
            hedge_ratio: Current hedge ratio
            current_capital: Available capital
            
        Returns:
            Tuple of (shares_asset1, shares_asset2)
        """
        if signal == 0:
            return 0, 0
        
        # Calculate dollar position size (max 25% of capital per position)
        max_dollar_position = current_capital * self.max_position_size
        
        # For pairs trading, position size is limited by the more expensive leg
        total_cost_per_unit = price1 + abs(hedge_ratio) * price2
        
        # Maximum units we can trade
        max_units = max_dollar_position / total_cost_per_unit
        
        # Apply leverage constraint
        leverage_adjusted_units = min(max_units, 
                                    current_capital * self.max_leverage / total_cost_per_unit)
        
        # Calculate actual shares
        shares_asset1 = int(signal * leverage_adjusted_units)
        shares_asset2 = int(-signal * hedge_ratio * leverage_adjusted_units)
        
        return shares_asset1, shares_asset2
    
    def calculate_transaction_costs(self, shares_traded1: int, shares_traded2: int,
                                  price1: float, price2: float) -> float:
        """
        Calculate total transaction costs for a trade.
        
        Args:
            shares_traded1: Shares traded in asset 1
            shares_traded2: Shares traded in asset 2
            price1: Price of asset 1
            price2: Price of asset 2
            
        Returns:
            Total transaction costs
        """
        # Dollar value of trades
        trade_value1 = abs(shares_traded1) * price1
        trade_value2 = abs(shares_traded2) * price2
        
        # Transaction costs as percentage of trade value
        percentage_costs = (trade_value1 + trade_value2) * (self.transaction_cost + self.slippage)
        
        # Fixed commission costs
        commission_costs = (abs(shares_traded1) + abs(shares_traded2)) * self.commission_per_share
        
        return percentage_costs + commission_costs
    
    def run_backtest(self, signals: pd.DataFrame, 
                    risk_manager: Optional[object] = None) -> Dict[str, pd.DataFrame]:
        """
        Run comprehensive backtest with transaction costs and risk management.
        
        Args:
            signals: DataFrame with trading signals and prices
            risk_manager: Optional risk manager object
            
        Returns:
            Dictionary with backtest results
        """
        # Initialize tracking variables
        dates = signals.index
        n_periods = len(dates)
        
        # Portfolio tracking
        portfolio_values = np.full(n_periods, self.initial_capital)
        cash = np.full(n_periods, self.initial_capital)
        positions1 = np.zeros(n_periods)
        positions2 = np.zeros(n_periods)
        
        # Trade tracking
        trades_log = []
        transaction_costs_log = []
        
        # Current positions
        current_pos1 = 0
        current_pos2 = 0
        current_cash = self.initial_capital
        
        for i, (date, row) in enumerate(signals.iterrows()):
            # Current market data
            price1 = row['price1']
            price2 = row['price2']
            hedge_ratio = row['hedge_ratio']
            signal = row['signal']
            
            # Skip if any price is NaN
            if np.isnan(price1) or np.isnan(price2) or np.isnan(hedge_ratio):
                if i > 0:
                    portfolio_values[i] = portfolio_values[i-1]
                    cash[i] = cash[i-1]
                    positions1[i] = positions1[i-1]
                    positions2[i] = positions2[i-1]
                continue
            
            # Calculate target positions
            target_pos1, target_pos2 = self.calculate_position_size(
                signal, price1, price2, hedge_ratio, 
                portfolio_values[i-1] if i > 0 else self.initial_capital
            )
            
            # Apply risk management if provided
            if risk_manager is not None:
                portfolio_value = (current_cash + 
                                 current_pos1 * price1 + 
                                 current_pos2 * price2)
                target_pos1, target_pos2 = risk_manager.adjust_positions(
                    target_pos1, target_pos2, portfolio_value, date
                )
            
            # Calculate trades needed
            trade1 = target_pos1 - current_pos1
            trade2 = target_pos2 - current_pos2
            
            # Execute trades if there are any
            transaction_costs = 0
            if trade1 != 0 or trade2 != 0:
                # Calculate transaction costs
                transaction_costs = self.calculate_transaction_costs(
                    trade1, trade2, price1, price2
                )
                
                # Check if we have enough cash for the trade
                trade_cost = (trade1 * price1 + trade2 * price2 + transaction_costs)
                
                if current_cash >= trade_cost:
                    # Execute trade
                    current_pos1 = target_pos1
                    current_pos2 = target_pos2
                    current_cash -= trade_cost
                    
                    # Log trade
                    trades_log.append({
                        'date': date,
                        'asset1_shares': trade1,
                        'asset2_shares': trade2,
                        'asset1_price': price1,
                        'asset2_price': price2,
                        'transaction_costs': transaction_costs,
                        'signal': signal
                    })
                else:
                    # Insufficient cash, maintain current positions
                    target_pos1 = current_pos1
                    target_pos2 = current_pos2
            
            # Update portfolio tracking
            portfolio_value = (current_cash + 
                             current_pos1 * price1 + 
                             current_pos2 * price2)
            
            portfolio_values[i] = portfolio_value
            cash[i] = current_cash
            positions1[i] = current_pos1
            positions2[i] = current_pos2
            transaction_costs_log.append(transaction_costs)
        
        # Create results DataFrames
        portfolio_df = pd.DataFrame({
            'portfolio_value': portfolio_values,
            'cash': cash,
            'position1': positions1,
            'position2': positions2,
            'price1': signals['price1'],
            'price2': signals['price2'],
            'signal': signals['signal'],
            'transaction_costs': transaction_costs_log
        }, index=dates)
        
        # Calculate returns
        portfolio_df['portfolio_return'] = portfolio_df['portfolio_value'].pct_change()
        portfolio_df['cumulative_return'] = (
            portfolio_df['portfolio_value'] / self.initial_capital - 1
        )
        
        trades_df = pd.DataFrame(trades_log)
        
        # Store results
        self.portfolio_values = portfolio_df
        self.trades = trades_df
        
        return {
            'portfolio': portfolio_df,
            'trades': trades_df,
            'signals': signals
        }
    
    def calculate_performance_metrics(self, portfolio_df: pd.DataFrame = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            portfolio_df: Portfolio DataFrame, uses stored if None
            
        Returns:
            Dictionary with performance metrics
        """
        if portfolio_df is None:
            if self.portfolio_values is None:
                raise ValueError("No backtest results available. Run backtest first.")
            portfolio_df = self.portfolio_values
        
        returns = portfolio_df['portfolio_return'].dropna()
        
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        total_return = portfolio_df['cumulative_return'].iloc[-1]
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative_values = portfolio_df['portfolio_value']
        running_max = cumulative_values.expanding().max()
        drawdown = (cumulative_values - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win/Loss statistics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Trading statistics
        total_trades = len(self.trades) if self.trades is not None else 0
        total_transaction_costs = portfolio_df['transaction_costs'].sum()
        
        # Information ratio (assuming benchmark return is 0 for market-neutral strategy)
        information_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annual_return / downside_volatility if downside_volatility > 0 else 0
        
        metrics = {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Information Ratio': information_ratio,
            'Calmar Ratio': calmar_ratio,
            'Maximum Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'Profit Factor': profit_factor,
            'Total Trades': total_trades,
            'Total Transaction Costs': total_transaction_costs,
            'Transaction Cost Ratio': total_transaction_costs / self.initial_capital
        }
        
        self.performance_metrics = metrics
        return metrics
    
    def monte_carlo_simulation(self, signals: pd.DataFrame, n_simulations: int = 1000,
                              return_distribution: str = 'bootstrap') -> Dict[str, np.ndarray]:
        """
        Run Monte Carlo simulation for strategy robustness testing.
        
        Args:
            signals: Trading signals DataFrame
            n_simulations: Number of simulations to run
            return_distribution: Method for generating returns ('bootstrap' or 'normal')
            
        Returns:
            Dictionary with simulation results
        """
        returns = signals['total_strategy_return'].dropna() if 'total_strategy_return' in signals.columns else []
        
        if len(returns) == 0:
            return {}
        
        simulation_results = {
            'final_values': [],
            'sharpe_ratios': [],
            'max_drawdowns': []
        }
        
        for _ in range(n_simulations):
            if return_distribution == 'bootstrap':
                # Bootstrap sampling
                simulated_returns = np.random.choice(returns, size=len(returns), replace=True)
            else:
                # Normal distribution
                mu = returns.mean()
                sigma = returns.std()
                simulated_returns = np.random.normal(mu, sigma, len(returns))
            
            # Calculate performance metrics for this simulation
            cumulative_value = self.initial_capital * (1 + pd.Series(simulated_returns)).cumprod()
            final_value = cumulative_value.iloc[-1]
            
            # Sharpe ratio
            sharpe = simulated_returns.mean() / simulated_returns.std() * np.sqrt(252) if simulated_returns.std() > 0 else 0
            
            # Maximum drawdown
            running_max = cumulative_value.expanding().max()
            drawdown = (cumulative_value - running_max) / running_max
            max_dd = drawdown.min()
            
            simulation_results['final_values'].append(final_value)
            simulation_results['sharpe_ratios'].append(sharpe)
            simulation_results['max_drawdowns'].append(max_dd)
        
        # Convert to numpy arrays
        for key in simulation_results:
            simulation_results[key] = np.array(simulation_results[key])
        
        return simulation_results
    
    def walk_forward_analysis(self, price_data: pd.DataFrame, strategy_func: Callable,
                             train_window: int = 252, test_window: int = 63) -> pd.DataFrame:
        """
        Perform walk-forward analysis for out-of-sample testing.
        
        Args:
            price_data: Price data DataFrame
            strategy_func: Function that generates signals given price data
            train_window: Training window size in days
            test_window: Testing window size in days
            
        Returns:
            DataFrame with walk-forward results
        """
        results = []
        n_periods = len(price_data)
        
        for start_idx in range(train_window, n_periods - test_window, test_window):
            # Training period
            train_start = start_idx - train_window
            train_end = start_idx
            train_data = price_data.iloc[train_start:train_end]
            
            # Test period
            test_start = start_idx
            test_end = start_idx + test_window
            test_data = price_data.iloc[test_start:test_end]
            
            try:
                # Generate strategy on training data and apply to test data
                strategy_signals = strategy_func(train_data, test_data)
                
                # Run backtest on test period
                backtest_results = self.run_backtest(strategy_signals)
                performance = self.calculate_performance_metrics(
                    backtest_results['portfolio']
                )
                
                result = {
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1],
                    **performance
                }
                results.append(result)
                
            except Exception as e:
                print(f"Walk-forward analysis failed for period {start_idx}: {e}")
                continue
        
        return pd.DataFrame(results)