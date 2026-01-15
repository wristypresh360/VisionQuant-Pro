"""
高级Transaction Cost模型
Advanced Transaction Cost Model

完整Transaction Cost = Commission + Slippage + Market Impact + Opportunity Cost

基于：
- Almgren & Chriss (2000): Optimal execution of portfolio transactions
- Kissell & Glantz (2003): Optimal Trading Strategies
- 业界标准实践

Author: VisionQuant Team
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class TransactionCostConfig:
    """Transaction Cost配置"""
    commission_rate: float = 0.001      # 手续费率 0.1%
    slippage_rate: float = 0.001        # 滑点率 0.1%
    market_impact_coef: float = 0.0001  # 市场冲击系数
    opportunity_cost_rate: float = 0.0005  # 机会成本率
    min_commission: float = 5.0         # 最小手续费（元）
    max_commission_pct: float = 0.003   # 最大手续费率 0.3%


class AdvancedTransactionCost:
    """
    高级Transaction Cost计算器
    
    完整模型：
    Total Cost = Commission + Slippage + Market Impact + Opportunity Cost
    """
    
    def __init__(self, config: TransactionCostConfig = None):
        """
        初始化Transaction Cost计算器
        
        Args:
            config: 配置参数，如果为None则使用默认配置
        """
        self.config = config or TransactionCostConfig()
        
    def calculate_cost(
        self,
        trade_size: float,
        price: float,
        volume: float,
        volatility: float = None,
        is_buy: bool = True
    ) -> Dict:
        """
        计算完整Transaction Cost
        
        Args:
            trade_size: 交易金额（元）
            price: 交易价格
            volume: 当日成交量（股）
            volatility: 波动率（可选，用于计算市场冲击）
            is_buy: 是否买入（True=买入, False=卖出）
            
        Returns:
            成本明细字典
        """
        shares = trade_size / price if price > 0 else 0
        
        # 1. Commission（手续费）
        commission = self._calculate_commission(trade_size)
        
        # 2. Slippage（滑点）
        slippage = self._calculate_slippage(trade_size, volatility)
        
        # 3. Market Impact（市场冲击）- 增强版
        # I = c * sigma * sqrt(Q / V)
        # Q: trade size (shares), V: daily volume
        market_impact = self._calculate_market_impact_enhanced(
            shares, price, volume, volatility
        )
        
        # 4. Opportunity Cost（机会成本）
        opportunity_cost = self._calculate_opportunity_cost(trade_size, volatility)
        
        # 总成本
        total_cost = commission + slippage + market_impact + opportunity_cost
        total_cost_pct = (total_cost / trade_size) * 100 if trade_size > 0 else 0
        
        return {
            'total_cost': round(total_cost, 2),
            'total_cost_pct': round(total_cost_pct, 4),
            'commission': round(commission, 2),
            'commission_pct': round((commission / trade_size) * 100, 4) if trade_size > 0 else 0,
            'slippage': round(slippage, 2),
            'slippage_pct': round((slippage / trade_size) * 100, 4) if trade_size > 0 else 0,
            'market_impact': round(market_impact, 2),
            'market_impact_pct': round((market_impact / trade_size) * 100, 4) if trade_size > 0 else 0,
            'opportunity_cost': round(opportunity_cost, 2),
            'opportunity_cost_pct': round((opportunity_cost / trade_size) * 100, 4) if trade_size > 0 else 0,
            'trade_size': round(trade_size, 2),
            'shares': round(shares, 0)
        }
    
    def _calculate_commission(self, trade_size: float) -> float:
        """计算手续费"""
        commission = trade_size * self.config.commission_rate
        commission = max(self.config.min_commission, commission)
        commission = min(commission, trade_size * self.config.max_commission_pct)
        return commission
    
    def _calculate_slippage(self, trade_size: float, volatility: float = None) -> float:
        """计算滑点"""
        rate = self.config.slippage_rate
        if volatility is not None and volatility > 0:
            rate *= (1 + volatility * 10)  # 波动率越高，滑点越大
        return trade_size * rate
    
    def _calculate_market_impact(self, shares, price, volume, volatility, is_buy):
        """(Legacy) 简单冲击"""
        return self._calculate_market_impact_enhanced(shares, price, volume, volatility)

    def _calculate_market_impact_enhanced(self, shares: float, price: float, 
                                         daily_volume: float, volatility: float = None) -> float:
        """
        计算市场冲击成本 (Square-root law approximation)
        Cost = 0.5 * Spread + c * sigma * sqrt(Size / Volume)
        这里简化 Spread 部分，着重冲击部分
        """
        if daily_volume <= 0 or shares <= 0:
            return 0.0
            
        participation_rate = shares / daily_volume
        
        # 冲击系数 (c)
        c = 0.5
        if volatility:
            c += volatility * 5  # 波动大，冲击大
            
        # 根号法则：Impact ~ sqrt(participation)
        impact_bps = c * np.sqrt(participation_rate)
        
        # 限制最大冲击 (防止小盘股溢出)
        impact_bps = min(impact_bps, 0.05) # Max 5%
        
        return shares * price * impact_bps
    
    def _calculate_opportunity_cost(self, trade_size: float, volatility: float = None) -> float:
        """计算机会成本（未成交风险）"""
        rate = self.config.opportunity_cost_rate
        if volatility is not None and volatility > 0:
            rate *= (1 + volatility * 5)
        return trade_size * rate
