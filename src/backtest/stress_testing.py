"""
Stress Testing模块
Stress Testing Module

在极端市场条件下测试策略

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from src.strategies.backtest_engine import VisionQuantBacktester, BacktestResult


@dataclass
class StressScenario:
    """Stress测试场景"""
    name: str
    start_date: str
    end_date: str
    description: str


class StressTester:
    """
    Stress测试器
    
    功能：
    1. 定义多种stress场景
    2. 在极端条件下回测
    3. 评估策略鲁棒性
    """
    
    def __init__(self):
        """初始化Stress测试器"""
        # 预定义stress场景
        self.scenarios = {
            'financial_crisis_2008': StressScenario(
                name='2008金融危机',
                start_date='2008-09-15',
                end_date='2009-03-09',
                description='2008年金融危机期间，市场大幅下跌'
            ),
            'covid_crash_2020': StressScenario(
                name='2020疫情崩盘',
                start_date='2020-02-20',
                end_date='2020-03-23',
                description='2020年COVID-19疫情引发的市场崩盘'
            ),
            'market_crash_2015': StressScenario(
                name='2015股灾',
                start_date='2015-06-15',
                end_date='2015-08-26',
                description='2015年中国股市异常波动'
            ),
            'factor_decay': StressScenario(
                name='因子衰减场景',
                start_date=None,  # 动态检测
                end_date=None,
                description='因子IC持续低于阈值时的表现'
            )
        }
    
    def run_stress_test(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        scores: pd.Series = None,
        win_rates: pd.Series = None,
        scenario_name: str = None
    ) -> Dict[str, BacktestResult]:
        """
        运行Stress测试
        
        Args:
            df: OHLCV数据
            signals: 交易信号
            scores: 评分序列
            win_rates: 胜率序列
            scenario_name: 场景名称（如果为None，测试所有场景）
            
        Returns:
            {scenario_name: BacktestResult} 字典
        """
        results = {}
        
        scenarios_to_test = [scenario_name] if scenario_name else list(self.scenarios.keys())
        
        for scenario_key in scenarios_to_test:
            if scenario_key not in self.scenarios:
                continue
            
            scenario = self.scenarios[scenario_key]
            
            # 提取stress期间的数据
            if scenario.start_date and scenario.end_date:
                stress_df = self._extract_stress_period(df, scenario.start_date, scenario.end_date)
            else:
                # 动态场景（如因子衰减），使用全部数据
                stress_df = df.copy()
            
            if stress_df.empty or len(stress_df) < 50:
                print(f"⚠️ 场景 {scenario.name} 数据不足，跳过")
                continue
            
            # 对齐信号
            stress_signals = signals.reindex(stress_df.index).fillna(0)
            stress_scores = scores.reindex(stress_df.index) if scores is not None else None
            stress_win_rates = win_rates.reindex(stress_df.index) if win_rates is not None else None
            
            # 回测
            backtester = VisionQuantBacktester()
            result = backtester.run_backtest(
                stress_df,
                stress_signals,
                stress_scores,
                stress_win_rates
            )
            
            results[scenario.name] = result
        
        return results
    
    def _extract_stress_period(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        提取stress期间的数据
        
        Args:
            df: 完整数据
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            stress期间的数据
        """
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        mask = (df.index >= start) & (df.index <= end)
        return df.loc[mask].copy()
    
    def run_adversarial_test(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        scores: pd.Series = None,
        win_rates: pd.Series = None,
        min_drawdown: float = 0.15
    ) -> Dict:
        """
        对抗性测试：主动寻找策略失效的场景
        
        Args:
            df: OHLCV数据
            signals: 交易信号
            scores: 评分序列
            win_rates: 胜率序列
            min_drawdown: 最小回撤阈值
            
        Returns:
            失效场景信息
        """
        # 滑动窗口检测
        window_size = 60  # 2个月
        failure_periods = []
        
        for i in range(window_size, len(df)):
            window_df = df.iloc[i-window_size:i]
            window_signals = signals.iloc[i-window_size:i] if i < len(signals) else signals.iloc[:i]
            
            # 回测该窗口
            backtester = VisionQuantBacktester()
            result = backtester.run_backtest(
                window_df,
                window_signals,
                scores.iloc[i-window_size:i] if scores is not None else None,
                win_rates.iloc[i-window_size:i] if win_rates is not None else None
            )
            
            # 检查是否失效
            if result.max_drawdown >= min_drawdown or result.total_return < -0.10:
                failure_periods.append({
                    'start': window_df.index[0],
                    'end': window_df.index[-1],
                    'return': result.total_return,
                    'drawdown': result.max_drawdown,
                    'sharpe': result.sharpe_ratio
                })
        
        return {
            'failure_periods': failure_periods,
            'num_failures': len(failure_periods),
            'failure_rate': len(failure_periods) / (len(df) - window_size) if len(df) > window_size else 0.0
        }
    
    def generate_stress_report(
        self,
        stress_results: Dict[str, BacktestResult]
    ) -> str:
        """
        生成Stress测试报告
        
        Args:
            stress_results: Stress测试结果
            
        Returns:
            报告文本
        """
        report = "=== Stress Testing Report ===\n\n"
        
        for scenario_name, result in stress_results.items():
            report += f"场景: {scenario_name}\n"
            report += f"  总收益: {result.total_return*100:.2f}%\n"
            report += f"  Sharpe比率: {result.sharpe_ratio:.2f}\n"
            report += f"  最大回撤: {result.max_drawdown*100:.2f}%\n"
            report += f"  胜率: {result.win_rate*100:.1f}%\n"
            report += f"  交易次数: {result.total_trades}\n"
            report += "\n"
        
        return report


if __name__ == "__main__":
    print("=== Stress测试器测试 ===")
    
    tester = StressTester()
    
    print(f"预定义场景数: {len(tester.scenarios)}")
    for name, scenario in tester.scenarios.items():
        print(f"  - {scenario.name}: {scenario.description}")
