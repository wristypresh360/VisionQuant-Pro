"""
分层回测引擎
Stratified Backtester

对每个分层进行回测，并支持Walk-Forward验证

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from tqdm import tqdm

from src.backtest.stock_stratifier import StockStratifier
from src.strategies.backtest_engine import VisionQuantBacktester, BacktestResult
from src.utils.walk_forward import WalkForwardValidator


class StratifiedBacktester:
    """
    分层回测引擎
    
    功能：
    1. 对每个分层进行回测
    2. 支持Walk-Forward验证
    3. 汇总各层结果
    """
    
    def __init__(
        self,
        use_walk_forward: bool = True,
        walk_forward_config: Dict = None
    ):
        """
        初始化分层回测引擎
        
        Args:
            use_walk_forward: 是否使用Walk-Forward验证
            walk_forward_config: Walk-Forward配置
        """
        self.use_walk_forward = use_walk_forward
        self.walk_forward_config = walk_forward_config or {
            'train_period': 252,
            'val_period': 63,
            'test_period': 63,
            'step_size': 63
        }
        
        self.stratifier = StockStratifier()
        self.backtester = VisionQuantBacktester(
            use_walk_forward=use_walk_forward
        )
    
    def run_stratified_backtest(
        self,
        stratified_stocks: pd.DataFrame,
        signal_generator: Callable,
        data_loader = None
    ) -> Dict[str, BacktestResult]:
        """
        运行分层回测
        
        Args:
            stratified_stocks: 已分层的股票DataFrame
            signal_generator: 信号生成函数 (df, symbol) -> (signals, scores, win_rates)
            data_loader: 数据加载器
            
        Returns:
            各分层的回测结果字典
        """
        # 获取所有分层
        strata = self.stratifier.get_stratum_list(stratified_stocks)
        
        results = {}
        
        for stratum in tqdm(strata, desc="分层回测"):
            # 获取该分层的股票
            stratum_stocks = self.stratifier.get_stratum_stocks(stratified_stocks, stratum)
            
            if len(stratum_stocks) < 10:  # 至少10只股票
                continue
            
            # 对该分层进行回测
            try:
                result = self._backtest_stratum(
                    stratum_stocks,
                    signal_generator,
                    data_loader
                )
                results[stratum] = result
            except Exception as e:
                print(f"⚠️ 分层 {stratum} 回测失败: {e}")
                continue
        
        return results
    
    def _backtest_stratum(
        self,
        stratum_stocks: pd.DataFrame,
        signal_generator: Callable,
        data_loader
    ) -> BacktestResult:
        """
        对单个分层进行回测
        
        Args:
            stratum_stocks: 该分层的股票
            signal_generator: 信号生成函数
            data_loader: 数据加载器
            
        Returns:
            回测结果
        """
        # 如果使用Walk-Forward，需要特殊处理
        if self.use_walk_forward:
            return self._backtest_stratum_walk_forward(
                stratum_stocks,
                signal_generator,
                data_loader
            )
        else:
            # 单次回测（简化：对每只股票单独回测，然后汇总）
            all_results = []
            
            for _, stock_row in stratum_stocks.iterrows():
                symbol = stock_row.get('symbol') or stock_row.name
                
                if data_loader:
                    df = data_loader.get_stock_data(symbol)
                    if df.empty:
                        continue
                    
                    # 生成信号
                    signals, scores, win_rates = signal_generator(df, symbol)
                    
                    # 回测
                    result = self.backtester.run_backtest(df, signals, scores, win_rates)
                    all_results.append(result)
            
            # 汇总结果
            return self._aggregate_results(all_results)
    
    def _backtest_stratum_walk_forward(
        self,
        stratum_stocks: pd.DataFrame,
        signal_generator: Callable,
        data_loader
    ) -> BacktestResult:
        """
        使用Walk-Forward对分层进行回测
        
        方法：对每只股票使用Walk-Forward，然后汇总
        """
        all_fold_results = []
        
        for _, stock_row in stratum_stocks.iterrows():
            symbol = stock_row.get('symbol') or stock_row.name
            
            if data_loader:
                df = data_loader.get_stock_data(symbol)
                if df.empty or len(df) < 500:  # 需要足够的数据
                    continue
                
                # 使用Walk-Forward回测
                result = self.backtester.run_walk_forward_backtest(
                    df=df,
                    signal_generator=lambda train_df: signal_generator(train_df, symbol),
                    **self.walk_forward_config
                )
                
                all_fold_results.append(result)
        
        # 汇总所有fold的结果
        return self._aggregate_walk_forward_results(all_fold_results)
    
    def _aggregate_results(self, results: List[BacktestResult]) -> BacktestResult:
        """
        汇总多个回测结果
        
        Args:
            results: 回测结果列表
            
        Returns:
            汇总后的结果
        """
        if not results:
            return BacktestResult(
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                avg_holding_period=0.0,
                equity_curve=pd.Series(),
                trade_log=[],
                fold_results=[]
            )
        
        # 计算平均指标
        total_return = np.mean([r.total_return for r in results])
        sharpe_ratio = np.mean([r.sharpe_ratio for r in results])
        max_drawdown = np.mean([r.max_drawdown for r in results])
        win_rate = np.mean([r.win_rate for r in results])
        profit_factor = np.mean([r.profit_factor for r in results])
        total_trades = sum([r.total_trades for r in results])
        
        # 合并交易日志
        all_trades = []
        for r in results:
            all_trades.extend(r.trade_log)
        
        # 合并净值曲线（简化：取平均）
        all_equity = []
        if results[0].equity_curve is not None and len(results[0].equity_curve) > 0:
            # 对齐所有净值曲线
            max_len = max([len(r.equity_curve) for r in results if r.equity_curve is not None])
            for i in range(max_len):
                equity_values = []
                for r in results:
                    if r.equity_curve is not None and i < len(r.equity_curve):
                        equity_values.append(r.equity_curve.iloc[i])
                if equity_values:
                    all_equity.append(np.mean(equity_values))
        
        equity_series = pd.Series(all_equity) if all_equity else pd.Series()
        
        # 年化收益（简化计算）
        if len(equity_series) > 0:
            annualized_return = (1 + total_return) ** (252 / len(equity_series)) - 1
        else:
            annualized_return = total_return
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_holding_period=0.0,  # TODO: 计算平均持仓天数
            equity_curve=equity_series,
            trade_log=all_trades,
            fold_results=[]
        )
    
    def _aggregate_walk_forward_results(self, results: List[BacktestResult]) -> BacktestResult:
        """
        汇总Walk-Forward回测结果
        """
        # 合并所有fold的结果
        all_fold_results = []
        for r in results:
            if r.fold_results:
                all_fold_results.extend(r.fold_results)
        
        # 计算平均指标
        if all_fold_results:
            total_return = np.mean([f.get('ret', 0) for f in all_fold_results])
            sharpe_ratio = np.mean([f.get('sharpe', 0) for f in all_fold_results])
            total_trades = sum([f.get('trades', 0) for f in all_fold_results])
        else:
            total_return = 0.0
            sharpe_ratio = 0.0
            total_trades = 0
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=total_return,  # 简化
            sharpe_ratio=sharpe_ratio,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=total_trades,
            avg_holding_period=0.0,
            equity_curve=pd.Series(),
            trade_log=[],
            fold_results=all_fold_results
        )


if __name__ == "__main__":
    print("=== 分层回测引擎测试 ===")
    
    # 模拟数据
    np.random.seed(42)
    stocks = pd.DataFrame({
        'symbol': [f'600{i:03d}' for i in range(50)],
        'market_cap': np.random.uniform(10, 1000, 50),
        'industry': np.random.choice(['银行', '地产', '科技'], 50)
    })
    
    stratifier = StockStratifier()
    stratified = stratifier.stratify_combined(stocks)
    
    backtester = StratifiedBacktester(use_walk_forward=False)
    
    # 模拟信号生成函数
    def dummy_signal_generator(df, symbol):
        signals = pd.Series(np.random.choice([-1, 0, 1], len(df)), index=df.index)
        scores = pd.Series(np.random.uniform(0, 10, len(df)), index=df.index)
        win_rates = pd.Series(np.random.uniform(40, 70, len(df)), index=df.index)
        return signals, scores, win_rates
    
    print("ℹ️ 需要提供真实的数据加载器和信号生成函数进行完整测试")
