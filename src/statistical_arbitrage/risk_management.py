"""
Risk Management Module

Implements comprehensive risk management constraints for statistical arbitrage trading.
Includes position sizing, exposure limits, drawdown controls, and correlation monitoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class RiskManager:
    """Comprehensive risk management for statistical arbitrage strategies."""
    
    def __init__(self, max_portfolio_risk: float = 0.02,
                 max_pair_allocation: float = 0.1,
                 max_sector_exposure: float = 0.3,
                 max_drawdown_limit: float = 0.15,
                 correlation_threshold: float = 0.8,
                 var_confidence: float = 0.05,
                 lookback_window: int = 252):
        """
        Initialize risk manager.
        
        Args:
            max_portfolio_risk: Maximum portfolio risk as fraction of capital
            max_pair_allocation: Maximum allocation per pair as fraction of capital
            max_sector_exposure: Maximum sector exposure as fraction of capital
            max_drawdown_limit: Maximum allowed drawdown before position reduction
            correlation_threshold: Maximum correlation between pairs
            var_confidence: Confidence level for VaR calculation
            lookback_window: Lookback window for risk calculations
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_pair_allocation = max_pair_allocation
        self.max_sector_exposure = max_sector_exposure
        self.max_drawdown_limit = max_drawdown_limit
        self.correlation_threshold = correlation_threshold
        self.var_confidence = var_confidence
        self.lookback_window = lookback_window
        
        # Risk tracking
        self.current_positions = {}
        self.portfolio_history = []
        self.risk_metrics_history = []
        
    def calculate_position_risk(self, position1: float, position2: float,
                               price1: float, price2: float,
                               returns1: pd.Series, returns2: pd.Series,
                               correlation: float) -> Dict[str, float]:
        """
        Calculate risk metrics for a pairs position.
        
        Args:
            position1: Position size in asset 1 (dollar value)
            position2: Position size in asset 2 (dollar value)
            price1: Current price of asset 1
            price2: Current price of asset 2
            returns1: Historical returns for asset 1
            returns2: Historical returns for asset 2
            correlation: Correlation between the assets
            
        Returns:
            Dictionary with risk metrics
        """
        if len(returns1) == 0 or len(returns2) == 0:
            return {'var': 0, 'volatility': 0, 'beta': 0}
        
        # Portfolio volatility calculation
        var1 = returns1.var()
        var2 = returns2.var()
        
        # Position weights (normalized)
        total_position = abs(position1) + abs(position2)
        if total_position == 0:
            return {'var': 0, 'volatility': 0, 'beta': 0}
        
        w1 = position1 / total_position
        w2 = position2 / total_position
        
        # Portfolio variance
        portfolio_var = (w1**2 * var1 + w2**2 * var2 + 
                        2 * w1 * w2 * correlation * np.sqrt(var1 * var2))
        
        portfolio_vol = np.sqrt(portfolio_var) * np.sqrt(252)  # Annualized
        
        # Value at Risk (VaR)
        var_multiplier = np.percentile(returns1 + returns2, self.var_confidence * 100)
        portfolio_var_risk = total_position * var_multiplier
        
        # Beta to market (simplified as average beta)
        market_returns = (returns1 + returns2) / 2  # Simplified market proxy
        if market_returns.var() > 0:
            beta = np.cov(returns1 + returns2, market_returns)[0, 1] / market_returns.var()
        else:
            beta = 0
        
        return {
            'var': portfolio_var_risk,
            'volatility': portfolio_vol,
            'beta': beta,
            'position_value': total_position
        }
    
    def calculate_portfolio_risk(self, positions: Dict[str, Tuple[float, float]],
                                prices: Dict[str, Tuple[float, float]],
                                returns_data: Dict[str, Tuple[pd.Series, pd.Series]],
                                correlations: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate portfolio-level risk metrics.
        
        Args:
            positions: Dictionary of positions {pair_id: (pos1, pos2)}
            prices: Dictionary of prices {pair_id: (price1, price2)}
            returns_data: Dictionary of returns {pair_id: (returns1, returns2)}
            correlations: Correlation matrix between all assets
            
        Returns:
            Dictionary with portfolio risk metrics
        """
        if not positions:
            return {'total_var': 0, 'portfolio_volatility': 0, 'concentration_risk': 0}
        
        # Calculate individual position risks
        position_risks = {}
        total_portfolio_value = 0
        
        for pair_id, (pos1, pos2) in positions.items():
            if pair_id not in prices or pair_id not in returns_data:
                continue
                
            price1, price2 = prices[pair_id]
            returns1, returns2 = returns_data[pair_id]
            
            # Get correlation between assets in this pair
            correlation = 0.5  # Default correlation if not available
            try:
                asset1_name = f"{pair_id}_1"
                asset2_name = f"{pair_id}_2"
                if asset1_name in correlations.columns and asset2_name in correlations.columns:
                    correlation = correlations.loc[asset1_name, asset2_name]
            except:
                pass
            
            risk_metrics = self.calculate_position_risk(
                pos1, pos2, price1, price2, returns1, returns2, correlation
            )
            position_risks[pair_id] = risk_metrics
            total_portfolio_value += risk_metrics['position_value']
        
        # Portfolio-level aggregation
        total_var = sum(metrics['var'] for metrics in position_risks.values())
        
        # Concentration risk (Herfindahl index)
        if total_portfolio_value > 0:
            weights = [metrics['position_value'] / total_portfolio_value 
                      for metrics in position_risks.values()]
            concentration_risk = sum(w**2 for w in weights)
        else:
            concentration_risk = 0
        
        # Portfolio volatility (simplified)
        avg_volatility = np.mean([metrics['volatility'] for metrics in position_risks.values()])
        
        return {
            'total_var': total_var,
            'portfolio_volatility': avg_volatility,
            'concentration_risk': concentration_risk,
            'total_portfolio_value': total_portfolio_value,
            'position_risks': position_risks
        }
    
    def check_position_limits(self, pair_id: str, target_pos1: float, target_pos2: float,
                             current_portfolio_value: float) -> Tuple[float, float]:
        """
        Check and adjust positions based on risk limits.
        
        Args:
            pair_id: Pair identifier
            target_pos1: Target position in asset 1
            target_pos2: Target position in asset 2
            current_portfolio_value: Current portfolio value
            
        Returns:
            Adjusted positions
        """
        # Calculate position value
        position_value = abs(target_pos1) + abs(target_pos2)
        
        # Check pair allocation limit
        max_pair_value = current_portfolio_value * self.max_pair_allocation
        if position_value > max_pair_value:
            # Scale down positions
            scale_factor = max_pair_value / position_value
            target_pos1 *= scale_factor
            target_pos2 *= scale_factor
        
        return target_pos1, target_pos2
    
    def check_sector_exposure(self, positions: Dict[str, Tuple[float, float]],
                             sector_mapping: Dict[str, str],
                             current_portfolio_value: float) -> Dict[str, Tuple[float, float]]:
        """
        Check and adjust positions based on sector exposure limits.
        
        Args:
            positions: Current positions
            sector_mapping: Mapping of pairs to sectors
            current_portfolio_value: Current portfolio value
            
        Returns:
            Adjusted positions
        """
        # Calculate sector exposures
        sector_exposures = {}
        for pair_id, (pos1, pos2) in positions.items():
            sector = sector_mapping.get(pair_id, 'Unknown')
            position_value = abs(pos1) + abs(pos2)
            
            if sector not in sector_exposures:
                sector_exposures[sector] = 0
            sector_exposures[sector] += position_value
        
        # Check sector limits and scale down if needed
        adjusted_positions = positions.copy()
        max_sector_value = current_portfolio_value * self.max_sector_exposure
        
        for sector, exposure in sector_exposures.items():
            if exposure > max_sector_value:
                scale_factor = max_sector_value / exposure
                
                # Scale down all positions in this sector
                for pair_id, (pos1, pos2) in positions.items():
                    if sector_mapping.get(pair_id) == sector:
                        adjusted_positions[pair_id] = (pos1 * scale_factor, pos2 * scale_factor)
        
        return adjusted_positions
    
    def check_correlation_risk(self, new_pair: str, existing_positions: Dict[str, Tuple[float, float]],
                              correlations: pd.DataFrame) -> bool:
        """
        Check if adding a new pair violates correlation limits.
        
        Args:
            new_pair: New pair to add
            existing_positions: Current positions
            correlations: Correlation matrix
            
        Returns:
            True if pair can be added, False otherwise
        """
        if not existing_positions:
            return True
        
        # Check correlation with existing pairs
        for existing_pair in existing_positions.keys():
            try:
                # Simplified correlation check
                if new_pair in correlations.index and existing_pair in correlations.columns:
                    correlation = abs(correlations.loc[new_pair, existing_pair])
                    if correlation > self.correlation_threshold:
                        return False
            except:
                continue
        
        return True
    
    def calculate_drawdown(self, portfolio_values: pd.Series) -> Tuple[float, float]:
        """
        Calculate current and maximum drawdown.
        
        Args:
            portfolio_values: Series of portfolio values
            
        Returns:
            Tuple of (current_drawdown, max_drawdown)
        """
        if len(portfolio_values) == 0:
            return 0.0, 0.0
        
        # Calculate running maximum
        running_max = portfolio_values.expanding().max()
        
        # Calculate drawdown
        drawdown = (portfolio_values - running_max) / running_max
        
        current_drawdown = drawdown.iloc[-1]
        max_drawdown = drawdown.min()
        
        return current_drawdown, max_drawdown
    
    def adjust_positions(self, target_pos1: float, target_pos2: float,
                        portfolio_value: float, date: pd.Timestamp) -> Tuple[float, float]:
        """
        Main method to adjust positions based on all risk constraints.
        
        Args:
            target_pos1: Target position in asset 1
            target_pos2: Target position in asset 2
            portfolio_value: Current portfolio value
            date: Current date
            
        Returns:
            Risk-adjusted positions
        """
        # Update portfolio history
        self.portfolio_history.append({
            'date': date,
            'portfolio_value': portfolio_value
        })
        
        # Check drawdown constraint
        if len(self.portfolio_history) > 1:
            portfolio_series = pd.Series([h['portfolio_value'] for h in self.portfolio_history])
            current_dd, max_dd = self.calculate_drawdown(portfolio_series)
            
            if abs(current_dd) > self.max_drawdown_limit:
                # Reduce position sizes during high drawdown periods
                reduction_factor = 1 - (abs(current_dd) - self.max_drawdown_limit) / self.max_drawdown_limit
                reduction_factor = max(0.5, reduction_factor)  # Minimum 50% reduction
                
                target_pos1 *= reduction_factor
                target_pos2 *= reduction_factor
        
        # Apply position limits
        target_pos1, target_pos2 = self.check_position_limits(
            'current_pair', target_pos1, target_pos2, portfolio_value
        )
        
        return target_pos1, target_pos2
    
    def calculate_portfolio_var(self, positions: Dict[str, Tuple[float, float]],
                               returns_data: Dict[str, Tuple[pd.Series, pd.Series]],
                               confidence_level: float = 0.05) -> float:
        """
        Calculate portfolio Value at Risk (VaR).
        
        Args:
            positions: Current positions
            returns_data: Historical returns data
            confidence_level: VaR confidence level
            
        Returns:
            Portfolio VaR
        """
        if not positions or not returns_data:
            return 0.0
        
        # Collect all returns for portfolio
        portfolio_returns = []
        
        for pair_id, (pos1, pos2) in positions.items():
            if pair_id not in returns_data:
                continue
                
            returns1, returns2 = returns_data[pair_id]
            
            # Calculate position returns (simplified)
            common_dates = returns1.index.intersection(returns2.index)
            if len(common_dates) > 0:
                r1 = returns1.loc[common_dates]
                r2 = returns2.loc[common_dates]
                
                # Position-weighted returns
                total_pos = abs(pos1) + abs(pos2)
                if total_pos > 0:
                    w1 = pos1 / total_pos
                    w2 = pos2 / total_pos
                    pair_returns = w1 * r1 + w2 * r2
                    portfolio_returns.append(pair_returns)
        
        if not portfolio_returns:
            return 0.0
        
        # Aggregate portfolio returns
        portfolio_return_series = pd.concat(portfolio_returns, axis=1).sum(axis=1)
        
        # Calculate VaR
        var_percentile = np.percentile(portfolio_return_series.dropna(), confidence_level * 100)
        
        return var_percentile
    
    def generate_risk_report(self, positions: Dict[str, Tuple[float, float]],
                           portfolio_value: float) -> Dict[str, any]:
        """
        Generate comprehensive risk report.
        
        Args:
            positions: Current positions
            portfolio_value: Current portfolio value
            
        Returns:
            Risk report dictionary
        """
        if len(self.portfolio_history) == 0:
            return {}
        
        # Portfolio history analysis
        portfolio_series = pd.Series([h['portfolio_value'] for h in self.portfolio_history])
        current_dd, max_dd = self.calculate_drawdown(portfolio_series)
        
        # Position concentration
        position_values = [abs(pos1) + abs(pos2) for pos1, pos2 in positions.values()]
        total_invested = sum(position_values)
        
        if total_invested > 0:
            max_position_pct = max(position_values) / total_invested
            concentration_ratio = sum(pv**2 for pv in position_values) / total_invested**2
        else:
            max_position_pct = 0
            concentration_ratio = 0
        
        # Risk utilization
        risk_utilization = total_invested / portfolio_value if portfolio_value > 0 else 0
        
        report = {
            'current_drawdown': current_dd,
            'max_drawdown': max_dd,
            'risk_utilization': risk_utilization,
            'max_position_percentage': max_position_pct,
            'concentration_ratio': concentration_ratio,
            'total_positions': len(positions),
            'total_invested_capital': total_invested,
            'available_capital': portfolio_value - total_invested,
            'drawdown_limit_breach': abs(current_dd) > self.max_drawdown_limit,
            'risk_budget_usage': risk_utilization / self.max_portfolio_risk if self.max_portfolio_risk > 0 else 0
        }
        
        return report


class VolatilityTargeting:
    """
    Volatility targeting risk management for dynamic position sizing.
    """
    
    def __init__(self, target_volatility: float = 0.15, lookback_window: int = 60):
        """
        Initialize volatility targeting.
        
        Args:
            target_volatility: Target portfolio volatility (annualized)
            lookback_window: Lookback window for volatility estimation
        """
        self.target_volatility = target_volatility
        self.lookback_window = lookback_window
    
    def calculate_position_multiplier(self, returns: pd.Series) -> float:
        """
        Calculate position size multiplier based on recent volatility.
        
        Args:
            returns: Recent portfolio returns
            
        Returns:
            Position size multiplier
        """
        if len(returns) < self.lookback_window // 2:
            return 1.0
        
        # Calculate recent volatility
        recent_vol = returns.tail(self.lookback_window).std() * np.sqrt(252)
        
        if recent_vol <= 0:
            return 1.0
        
        # Calculate multiplier to achieve target volatility
        multiplier = self.target_volatility / recent_vol
        
        # Limit multiplier to reasonable range
        multiplier = np.clip(multiplier, 0.25, 4.0)
        
        return multiplier