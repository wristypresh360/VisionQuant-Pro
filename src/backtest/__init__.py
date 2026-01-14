"""
分层回测模块
Stratified Backtesting Module

包含：
- 股票分层逻辑
- 分层回测引擎
- 结果汇总
- 可视化
"""

from .stock_stratifier import StockStratifier
from .stratified_backtester import StratifiedBacktester
from .result_aggregator import ResultAggregator

__all__ = [
    'StockStratifier',
    'StratifiedBacktester',
    'ResultAggregator'
]
