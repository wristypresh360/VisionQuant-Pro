"""
分层结果汇总
Stratified Result Aggregator

汇总各分层的回测结果，生成对比分析

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from src.strategies.backtest_engine import BacktestResult


class ResultAggregator:
    """
    结果汇总器
    
    功能：
    1. 汇总各分层回测结果
    2. 生成对比表格
    3. 计算分层差异统计
    """
    
    def __init__(self):
        """初始化结果汇总器"""
        pass
    
    def aggregate_stratified_results(
        self,
        stratified_results: Dict[str, BacktestResult]
    ) -> pd.DataFrame:
        """
        汇总分层回测结果
        
        Args:
            stratified_results: {stratum: BacktestResult} 字典
            
        Returns:
            汇总表格DataFrame
        """
        rows = []
        
        for stratum, result in stratified_results.items():
            rows.append({
                'Stratum': stratum,
                'Total Return': f"{result.total_return*100:.2f}%",
                'Annualized Return': f"{result.annualized_return*100:.2f}%",
                'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
                'Max Drawdown': f"{result.max_drawdown*100:.2f}%",
                'Win Rate': f"{result.win_rate*100:.1f}%",
                'Profit Factor': f"{result.profit_factor:.2f}",
                'Total Trades': result.total_trades
            })
        
        df = pd.DataFrame(rows)
        
        # 按Sharpe Ratio排序
        df['Sharpe_Value'] = df['Sharpe Ratio'].str.replace('%', '').astype(float)
        df = df.sort_values('Sharpe_Value', ascending=False)
        df = df.drop(columns=['Sharpe_Value'])
        
        return df
    
    def compare_by_market_cap(
        self,
        stratified_results: Dict[str, BacktestResult]
    ) -> pd.DataFrame:
        """
        按市值分层对比
        
        Args:
            stratified_results: 分层回测结果
            
        Returns:
            市值对比表格
        """
        market_cap_results = {
            'small': [],
            'medium': [],
            'large': []
        }
        
        for stratum, result in stratified_results.items():
            if 'small' in stratum:
                market_cap_results['small'].append(result)
            elif 'medium' in stratum:
                market_cap_results['medium'].append(result)
            elif 'large' in stratum:
                market_cap_results['large'].append(result)
        
        rows = []
        for market_cap, results in market_cap_results.items():
            if results:
                avg_return = np.mean([r.total_return for r in results])
                avg_sharpe = np.mean([r.sharpe_ratio for r in results])
                avg_drawdown = np.mean([r.max_drawdown for r in results])
                
                rows.append({
                    'Market Cap': market_cap,
                    'Avg Return': f"{avg_return*100:.2f}%",
                    'Avg Sharpe': f"{avg_sharpe:.2f}",
                    'Avg Drawdown': f"{avg_drawdown*100:.2f}%",
                    'Num Strata': len(results)
                })
        
        return pd.DataFrame(rows)
    
    def compare_by_industry(
        self,
        stratified_results: Dict[str, BacktestResult]
    ) -> pd.DataFrame:
        """
        按行业分层对比
        
        Args:
            stratified_results: 分层回测结果
            
        Returns:
            行业对比表格
        """
        industry_results = {}
        
        for stratum, result in stratified_results.items():
            # 解析stratum获取行业
            parts = stratum.split('_', 1)
            if len(parts) == 2:
                industry = parts[1]
                if industry not in industry_results:
                    industry_results[industry] = []
                industry_results[industry].append(result)
        
        rows = []
        for industry, results in industry_results.items():
            if results:
                avg_return = np.mean([r.total_return for r in results])
                avg_sharpe = np.mean([r.sharpe_ratio for r in results])
                avg_drawdown = np.mean([r.max_drawdown for r in results])
                
                rows.append({
                    'Industry': industry,
                    'Avg Return': f"{avg_return*100:.2f}%",
                    'Avg Sharpe': f"{avg_sharpe:.2f}",
                    'Avg Drawdown': f"{avg_drawdown*100:.2f}%",
                    'Num Strata': len(results)
                })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('Avg Sharpe', ascending=False)
        
        return df
    
    def generate_summary_statistics(
        self,
        stratified_results: Dict[str, BacktestResult]
    ) -> Dict:
        """
        生成汇总统计
        
        Args:
            stratified_results: 分层回测结果
            
        Returns:
            统计信息字典
        """
        if not stratified_results:
            return {}
        
        all_returns = [r.total_return for r in stratified_results.values()]
        all_sharpes = [r.sharpe_ratio for r in stratified_results.values()]
        all_drawdowns = [r.max_drawdown for r in stratified_results.values()]
        all_win_rates = [r.win_rate for r in stratified_results.values()]
        
        return {
            'num_strata': len(stratified_results),
            'return_stats': {
                'mean': np.mean(all_returns),
                'std': np.std(all_returns),
                'min': np.min(all_returns),
                'max': np.max(all_returns),
                'median': np.median(all_returns)
            },
            'sharpe_stats': {
                'mean': np.mean(all_sharpes),
                'std': np.std(all_sharpes),
                'min': np.min(all_sharpes),
                'max': np.max(all_sharpes),
                'median': np.median(all_sharpes)
            },
            'drawdown_stats': {
                'mean': np.mean(all_drawdowns),
                'std': np.std(all_drawdowns),
                'max': np.max(all_drawdowns)
            },
            'win_rate_stats': {
                'mean': np.mean(all_win_rates),
                'std': np.std(all_win_rates)
            }
        }


if __name__ == "__main__":
    print("=== 结果汇总器测试 ===")
    
    from src.strategies.backtest_engine import BacktestResult
    
    # 模拟结果
    results = {
        'small_银行': BacktestResult(
            total_return=0.15, sharpe_ratio=1.2, max_drawdown=0.10,
            win_rate=0.55, profit_factor=1.5, total_trades=100,
            annualized_return=0.15, avg_holding_period=5.0,
            equity_curve=pd.Series(), trade_log=[], fold_results=[]
        ),
        'large_科技': BacktestResult(
            total_return=0.25, sharpe_ratio=1.8, max_drawdown=0.12,
            win_rate=0.60, profit_factor=1.8, total_trades=80,
            annualized_return=0.25, avg_holding_period=6.0,
            equity_curve=pd.Series(), trade_log=[], fold_results=[]
        )
    }
    
    aggregator = ResultAggregator()
    summary_df = aggregator.aggregate_stratified_results(results)
    print("\n汇总表格:")
    print(summary_df)
