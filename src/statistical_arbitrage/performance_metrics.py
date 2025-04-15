"""
Performance Metrics Module

Comprehensive performance analysis for statistical arbitrage strategies.
Calculates various risk-adjusted returns and provides detailed analytics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class PerformanceAnalyzer:
    """Comprehensive performance analysis for trading strategies."""
    
    def __init__(self, benchmark_return: float = 0.0, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer.
        
        Args:
            benchmark_return: Benchmark return for comparison (annualized)
            risk_free_rate: Risk-free rate for Sharpe ratio calculation (annualized)
        """
        self.benchmark_return = benchmark_return
        self.risk_free_rate = risk_free_rate
        
    def calculate_basic_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate basic performance metrics.
        
        Args:
            returns: Series of strategy returns
            
        Returns:
            Dictionary with basic metrics
        """
        if len(returns) == 0:
            return {}
        
        # Remove any infinite or NaN values
        clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(clean_returns) == 0:
            return {}
        
        # Basic statistics
        total_return = (1 + clean_returns).prod() - 1
        n_periods = len(clean_returns)
        annualized_return = (1 + total_return) ** (252 / n_periods) - 1
        
        # Risk metrics
        volatility = clean_returns.std() * np.sqrt(252)
        downside_returns = clean_returns[clean_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Ratios
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # Win/Loss statistics
        positive_returns = clean_returns[clean_returns > 0]
        negative_returns = clean_returns[clean_returns < 0]
        
        win_rate = len(positive_returns) / len(clean_returns)
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        
        # Profit factor
        total_wins = positive_returns.sum() if len(positive_returns) > 0 else 0
        total_losses = abs(negative_returns.sum()) if len(negative_returns) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        
        return {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Volatility': volatility,
            'Downside Volatility': downside_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Win Rate': win_rate,
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'Profit Factor': profit_factor,
            'Number of Periods': n_periods
        }
    
    def calculate_drawdown_metrics(self, portfolio_values: pd.Series) -> Dict[str, float]:
        """
        Calculate drawdown-related metrics.
        
        Args:
            portfolio_values: Series of portfolio values
            
        Returns:
            Dictionary with drawdown metrics
        """
        if len(portfolio_values) == 0:
            return {}
        
        # Calculate running maximum
        running_max = portfolio_values.expanding().max()
        
        # Calculate drawdown
        drawdown = (portfolio_values - running_max) / running_max
        
        # Key drawdown metrics
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1]
        
        # Drawdown duration analysis
        drawdown_periods = (drawdown < 0).astype(int)
        drawdown_starts = drawdown_periods.diff() > 0
        drawdown_ends = drawdown_periods.diff() < 0
        
        durations = []
        start_idx = None
        
        for i, (is_start, is_end) in enumerate(zip(drawdown_starts, drawdown_ends)):
            if is_start:
                start_idx = i
            elif is_end and start_idx is not None:
                durations.append(i - start_idx)
                start_idx = None
        
        # Handle case where drawdown period extends to the end
        if start_idx is not None:
            durations.append(len(drawdown) - start_idx)
        
        avg_drawdown_duration = np.mean(durations) if durations else 0
        max_drawdown_duration = max(durations) if durations else 0
        
        # Recovery analysis
        recovery_times = []
        for i, dd in enumerate(drawdown):
            if dd == 0 and i > 0:  # Recovery point
                # Find the start of this drawdown period
                for j in range(i-1, -1, -1):
                    if drawdown.iloc[j] == 0:
                        recovery_times.append(i - j)
                        break
        
        avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
        
        # Calmar ratio
        calmar_ratio = abs(portfolio_values.pct_change().mean() * 252 / max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'Max Drawdown': max_drawdown,
            'Current Drawdown': current_drawdown,
            'Average Drawdown Duration': avg_drawdown_duration,
            'Max Drawdown Duration': max_drawdown_duration,
            'Average Recovery Time': avg_recovery_time,
            'Calmar Ratio': calmar_ratio,
            'Number of Drawdown Periods': len(durations)
        }
    
    def calculate_risk_metrics(self, returns: pd.Series, confidence_levels: List[float] = [0.05, 0.01]) -> Dict[str, float]:
        """
        Calculate risk metrics including VaR and CVaR.
        
        Args:
            returns: Series of strategy returns
            confidence_levels: List of confidence levels for VaR calculation
            
        Returns:
            Dictionary with risk metrics
        """
        if len(returns) == 0:
            return {}
        
        clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(clean_returns) == 0:
            return {}
        
        risk_metrics = {}
        
        # Value at Risk (VaR) for different confidence levels
        for confidence in confidence_levels:
            var_percentile = np.percentile(clean_returns, confidence * 100)
            cvar = clean_returns[clean_returns <= var_percentile].mean()
            
            risk_metrics[f'VaR_{int(confidence*100)}%'] = var_percentile
            risk_metrics[f'CVaR_{int(confidence*100)}%'] = cvar
        
        # Skewness and Kurtosis
        risk_metrics['Skewness'] = stats.skew(clean_returns)
        risk_metrics['Kurtosis'] = stats.kurtosis(clean_returns)
        
        # Tail ratio
        percentile_95 = np.percentile(clean_returns, 95)
        percentile_5 = np.percentile(clean_returns, 5)
        tail_ratio = percentile_95 / abs(percentile_5) if percentile_5 != 0 else np.inf
        risk_metrics['Tail Ratio'] = tail_ratio
        
        # Maximum daily loss
        risk_metrics['Max Daily Loss'] = clean_returns.min()
        risk_metrics['Max Daily Gain'] = clean_returns.max()
        
        return risk_metrics
    
    def calculate_information_ratio(self, strategy_returns: pd.Series, 
                                  benchmark_returns: pd.Series = None) -> float:
        """
        Calculate information ratio vs benchmark.
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns (optional)
            
        Returns:
            Information ratio
        """
        if benchmark_returns is None:
            # Use constant benchmark return
            benchmark_returns = pd.Series([self.benchmark_return / 252] * len(strategy_returns),
                                        index=strategy_returns.index)
        
        # Calculate excess returns
        excess_returns = strategy_returns - benchmark_returns
        
        if excess_returns.std() == 0:
            return 0
        
        # Information ratio
        ir = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        return ir
    
    def calculate_treynor_ratio(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate Treynor ratio.
        
        Args:
            returns: Strategy returns
            market_returns: Market returns for beta calculation
            
        Returns:
            Treynor ratio
        """
        if len(returns) == 0 or len(market_returns) == 0:
            return 0
        
        # Align returns
        common_index = returns.index.intersection(market_returns.index)
        strategy_ret = returns.loc[common_index]
        market_ret = market_returns.loc[common_index]
        
        if len(strategy_ret) == 0 or market_ret.var() == 0:
            return 0
        
        # Calculate beta
        beta = np.cov(strategy_ret, market_ret)[0, 1] / market_ret.var()
        
        if beta == 0:
            return 0
        
        # Treynor ratio
        annualized_return = strategy_ret.mean() * 252
        treynor_ratio = (annualized_return - self.risk_free_rate) / beta
        
        return treynor_ratio
    
    def calculate_tracking_error(self, strategy_returns: pd.Series, 
                               benchmark_returns: pd.Series) -> float:
        """
        Calculate tracking error vs benchmark.
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Annualized tracking error
        """
        # Align returns
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_ret = strategy_returns.loc[common_index]
        benchmark_ret = benchmark_returns.loc[common_index]
        
        if len(strategy_ret) == 0:
            return 0
        
        # Calculate tracking error
        excess_returns = strategy_ret - benchmark_ret
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        return tracking_error
    
    def rolling_performance_analysis(self, returns: pd.Series, 
                                   window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            returns: Strategy returns
            window: Rolling window size
            
        Returns:
            DataFrame with rolling metrics
        """
        if len(returns) < window:
            return pd.DataFrame()
        
        rolling_metrics = pd.DataFrame(index=returns.index[window-1:])
        
        # Rolling returns and volatility
        rolling_metrics['Rolling Return'] = returns.rolling(window).apply(
            lambda x: (1 + x).prod() - 1
        ) * (252 / window)
        
        rolling_metrics['Rolling Volatility'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Rolling Sharpe ratio
        rolling_metrics['Rolling Sharpe'] = (
            (rolling_metrics['Rolling Return'] - self.risk_free_rate) / 
            rolling_metrics['Rolling Volatility']
        )
        
        # Rolling maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.rolling(window).max()
        rolling_drawdown = (cumulative_returns - rolling_max) / rolling_max
        rolling_metrics['Rolling Max Drawdown'] = rolling_drawdown.rolling(window).min()
        
        # Rolling win rate
        rolling_metrics['Rolling Win Rate'] = returns.rolling(window).apply(
            lambda x: (x > 0).mean()
        )
        
        return rolling_metrics
    
    def generate_performance_report(self, portfolio_values: pd.Series, 
                                  returns: pd.Series,
                                  benchmark_returns: pd.Series = None) -> Dict[str, any]:
        """
        Generate comprehensive performance report.
        
        Args:
            portfolio_values: Portfolio value series
            returns: Strategy returns
            benchmark_returns: Optional benchmark returns
            
        Returns:
            Comprehensive performance report
        """
        report = {}
        
        # Basic metrics
        report['Basic Metrics'] = self.calculate_basic_metrics(returns)
        
        # Drawdown metrics
        report['Drawdown Metrics'] = self.calculate_drawdown_metrics(portfolio_values)
        
        # Risk metrics
        report['Risk Metrics'] = self.calculate_risk_metrics(returns)
        
        # Information ratio
        if benchmark_returns is not None:
            report['Information Ratio'] = self.calculate_information_ratio(returns, benchmark_returns)
            report['Tracking Error'] = self.calculate_tracking_error(returns, benchmark_returns)
        
        # Rolling analysis
        if len(returns) >= 252:
            report['Rolling Analysis'] = self.rolling_performance_analysis(returns)
        
        return report
    
    def plot_performance_charts(self, portfolio_values: pd.Series, 
                              returns: pd.Series,
                              benchmark_values: pd.Series = None,
                              save_path: str = None) -> None:
        """
        Generate performance visualization charts.
        
        Args:
            portfolio_values: Portfolio value series
            returns: Strategy returns
            benchmark_values: Optional benchmark values
            save_path: Path to save the charts
        """
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Cumulative returns
        axes[0, 0].plot(portfolio_values.index, portfolio_values, label='Strategy', linewidth=2)
        if benchmark_values is not None:
            axes[0, 0].plot(benchmark_values.index, benchmark_values, 
                           label='Benchmark', linewidth=2, alpha=0.7)
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_ylabel('Portfolio Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        axes[0, 1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].plot(drawdown.index, drawdown, color='red', linewidth=1)
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Return distribution
        clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        axes[1, 0].hist(clean_returns, bins=50, alpha=0.7, density=True)
        axes[1, 0].axvline(clean_returns.mean(), color='red', linestyle='--', 
                          label=f'Mean: {clean_returns.mean():.4f}')
        axes[1, 0].set_title('Return Distribution')
        axes[1, 0].set_xlabel('Daily Returns')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Rolling Sharpe ratio
        if len(returns) >= 60:
            rolling_sharpe = (returns.rolling(60).mean() / returns.rolling(60).std()) * np.sqrt(252)
            axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe, linewidth=2)
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 1].set_title('Rolling 60-Day Sharpe Ratio')
            axes[1, 1].set_ylabel('Sharpe Ratio')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compare_strategies(self, strategy_returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Compare multiple strategies side by side.
        
        Args:
            strategy_returns: Dictionary of strategy returns {name: returns_series}
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_results = {}
        
        for strategy_name, returns in strategy_returns.items():
            metrics = self.calculate_basic_metrics(returns)
            
            # Add risk metrics
            risk_metrics = self.calculate_risk_metrics(returns)
            metrics.update(risk_metrics)
            
            comparison_results[strategy_name] = metrics
        
        comparison_df = pd.DataFrame(comparison_results).T
        
        # Add ranking columns
        ranking_metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Total Return']
        for metric in ranking_metrics:
            if metric in comparison_df.columns:
                comparison_df[f'{metric} Rank'] = comparison_df[metric].rank(ascending=False)
        
        return comparison_df