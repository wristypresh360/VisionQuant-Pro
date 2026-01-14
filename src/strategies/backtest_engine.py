"""
回测引擎 - 集成Walk-Forward验证
Backtest Engine with Walk-Forward Integration

整合了:
1. Walk-Forward时序验证
2. Triple Barrier标签预测
3. 凯利公式仓位管理
4. 统计显著性检验

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings

# 导入项目模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.walk_forward import WalkForwardValidator, WalkForwardSplit
from src.data.triple_barrier import TripleBarrierLabeler, calculate_win_loss_ratio
from src.strategies.kelly_position import KellyPositionCalculator, PositionManager
from src.strategies.transaction_cost import AdvancedTransactionCost


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_holding_period: float
    equity_curve: pd.Series
    trade_log: List[Dict]
    fold_results: List[Dict]  # Walk-Forward各折的结果


class VisionQuantBacktester:
    """
    VisionQuant回测引擎
    
    特点:
    1. 基于视觉相似度的信号生成
    2. Walk-Forward防止过拟合
    3. 凯利公式动态仓位
    4. Triple Barrier风控
    """
    
    def __init__(
        self,
        initial_capital: float = 1000000,
        commission: float = 0.001,      # 手续费0.1%（已废弃，使用AdvancedTransactionCost）
        slippage: float = 0.001,        # 滑点0.1%（已废弃，使用AdvancedTransactionCost）
        stop_loss: float = 0.08,        # 止损8%
        take_profit: float = 0.15,      # 止盈15%
        max_position: float = 0.25,     # 最大仓位25%
        use_kelly: bool = True,
        use_walk_forward: bool = True,
        use_advanced_cost: bool = True,  # 是否使用高级Transaction Cost模型
        max_daily_turnover: float = 0.20,  # 单日最大turnover 20%
        max_weekly_turnover: float = 0.50  # 单周最大turnover 50%
    ):
        self.initial_capital = initial_capital
        self.commission = commission  # 保留用于向后兼容
        self.slippage = slippage  # 保留用于向后兼容
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_position = max_position
        self.use_kelly = use_kelly
        self.use_walk_forward = use_walk_forward
        self.use_advanced_cost = use_advanced_cost
        self.max_daily_turnover = max_daily_turnover
        self.max_weekly_turnover = max_weekly_turnover
        
        # 组件初始化
        self.position_manager = PositionManager() if use_kelly else None
        self.labeler = TripleBarrierLabeler(
            upper_barrier=take_profit,
            lower_barrier=stop_loss
        )
        self.cost_calculator = AdvancedTransactionCost() if use_advanced_cost else None
        
        # Turnover跟踪
        self.daily_turnover = {}  # {date: turnover}
        self.weekly_turnover = {}  # {week: turnover}
        
    def run_backtest(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        scores: pd.Series = None,
        win_rates: pd.Series = None
    ) -> BacktestResult:
        """
        运行回测
        
        Args:
            df: OHLCV数据
            signals: 交易信号 (1=买入, 0=持有, -1=卖出)
            scores: V+F+Q评分序列
            win_rates: 历史胜率序列
            
        Returns:
            BacktestResult
        """
        # 初始化
        capital = self.initial_capital
        position = 0
        entry_price = 0
        equity = [capital]
        trade_log = []
        
        # 对齐索引
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        
        for i in range(1, len(df)):
            date = df.index[i]
            price = df['Close'].iloc[i]
            signal = signals.iloc[i] if i < len(signals) else 0
            
            # 持仓状态下检查止盈止损
            if position > 0:
                returns = (price - entry_price) / entry_price
                
                # 触发止损
                if returns <= -self.stop_loss:
                    sell_amount = position * price
                    sell_value, cost = self._execute_trade(
                        shares=position,
                        price=price,
                        volume=df.iloc[i].get('Volume', position * 10) if 'Volume' in df.columns else position * 10,
                        volatility=None,
                        is_buy=False
                    )
                    capital += sell_value
                    trade_log.append({
                        'date': date,
                        'action': 'STOP_LOSS',
                        'price': price,
                        'shares': position,
                        'return': returns,
                        'cost': cost
                    })
                    position = 0
                    
                # 触发止盈
                elif returns >= self.take_profit:
                    sell_amount = position * price
                    sell_value, cost = self._execute_trade(
                        shares=position,
                        price=price,
                        volume=df.iloc[i].get('Volume', position * 10) if 'Volume' in df.columns else position * 10,
                        volatility=None,
                        is_buy=False
                    )
                    capital += sell_value
                    trade_log.append({
                        'date': date,
                        'action': 'TAKE_PROFIT',
                        'price': price,
                        'shares': position,
                        'return': returns,
                        'cost': cost
                    })
                    position = 0
            
            # 处理交易信号
            if signal == 1 and position == 0:  # 买入信号
                # 检查Turnover约束
                if not self._check_turnover_constraint(date, capital, 'BUY'):
                    continue  # 超过Turnover限制，跳过交易
                
                # 计算仓位
                if self.use_kelly and scores is not None and win_rates is not None:
                    score = scores.iloc[i] if i < len(scores) else 5
                    win_rate = win_rates.iloc[i] if i < len(win_rates) else 0.5
                    
                    result = self.position_manager.get_position(
                        win_rate=win_rate,
                        win_loss_ratio=1.5,  # 默认盈亏比
                        score=score
                    )
                    position_pct = result['final_position']
                else:
                    position_pct = self.max_position
                
                # 执行买入（考虑Transaction Cost）
                buy_amount = capital * position_pct
                # 先计算成本，再确定实际可买股数
                if self.use_advanced_cost and self.cost_calculator:
                    cost_detail = self.cost_calculator.calculate_cost(
                        trade_size=buy_amount,
                        price=price,
                        volume=df.iloc[i].get('Volume', buy_amount / price * 10) if 'Volume' in df.columns else buy_amount / price * 10,
                        volatility=None,
                        is_buy=True
                    )
                    # 考虑成本后，实际可买金额减少
                    net_buy_amount = buy_amount - cost_detail['total_cost']
                    shares = int(net_buy_amount / price)
                    cost = cost_detail['total_cost']
                else:
                    # 简单模型
                    shares = int(buy_amount / (price * (1 + self.commission + self.slippage)))
                    cost = buy_amount - shares * price
                
                # 再次检查Turnover（基于实际交易金额）
                actual_trade_amount = shares * price
                if not self._check_turnover_with_amount(date, capital, actual_trade_amount):
                    continue
                
                if shares > 0:
                    capital -= (buy_amount + cost)
                    position = shares
                    entry_price = price
                    
                    # 更新Turnover
                    self._update_turnover(date, buy_amount, capital)
                    
                    trade_log.append({
                        'date': date,
                        'action': 'BUY',
                        'price': price,
                        'shares': shares,
                        'position_pct': position_pct,
                        'cost': cost
                    })
                    
            elif signal == -1 and position > 0:  # 卖出信号
                # 检查Turnover约束
                if not self._check_turnover_constraint(date, capital, 'SELL'):
                    continue
                
                returns = (price - entry_price) / entry_price
                sell_amount = position * price
                sell_value, cost = self._execute_trade(
                    shares=position,
                    price=price,
                    volume=df.iloc[i].get('Volume', position * 10) if 'Volume' in df.columns else position * 10,
                    volatility=None,
                    is_buy=False
                )
                capital += sell_value
                
                # 更新Turnover
                self._update_turnover(date, sell_amount, capital)
                
                trade_log.append({
                    'date': date,
                    'action': 'SELL',
                    'price': price,
                    'shares': position,
                    'return': returns,
                    'cost': cost
                })
                position = 0
            
            # 记录净值
            current_value = capital + position * price
            equity.append(current_value)
        
        # 计算统计指标
        equity_series = pd.Series(equity, index=df.index[:len(equity)])
        returns_series = equity_series.pct_change().dropna()
        
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital
        
        # 年化收益
        trading_days = len(df)
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1
        
        # 夏普比率
        if returns_series.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns_series.mean() / returns_series.std()
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # 胜率
        wins = [t for t in trade_log if t.get('return', 0) > 0]
        losses = [t for t in trade_log if t.get('return', 0) < 0]
        win_rate = len(wins) / len(trade_log) if trade_log else 0
        
        # 盈亏比
        avg_win = np.mean([t['return'] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t['return'] for t in losses])) if losses else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trade_log),
            avg_holding_period=0,  # TODO: 计算平均持仓天数
            equity_curve=equity_series,
            trade_log=trade_log,
            fold_results=[]
        )
    
    def run_walk_forward_backtest(
        self,
        df: pd.DataFrame,
        signal_generator,
        train_period: int = 252,
        test_period: int = 63,
        step_size: int = 63
    ) -> BacktestResult:
        """
        运行Walk-Forward回测
        
        Args:
            df: OHLCV数据
            signal_generator: 信号生成函数 (train_df) -> (signals, scores, win_rates)
            train_period: 训练期长度
            test_period: 测试期长度
            step_size: 滚动步长
            
        Returns:
            BacktestResult
        """
        validator = WalkForwardValidator(
            train_period=train_period,
            val_period=0,  # 简化版不使用验证集
            test_period=test_period,
            step_size=step_size
        )
        
        all_equity = [self.initial_capital]
        all_trades = []
        fold_results = []
        capital = self.initial_capital
        
        for split in validator.split(df):
            train_df = df.iloc[split.train_indices]
            test_df = df.iloc[split.test_indices]
            
            if len(test_df) < 10:
                continue
            
            # 生成信号
            try:
                signals, scores, win_rates = signal_generator(train_df)
                
                # 对齐到测试集
                test_signals = signals.reindex(test_df.index).fillna(0)
                test_scores = scores.reindex(test_df.index).fillna(5) if scores is not None else None
                test_win_rates = win_rates.reindex(test_df.index).fillna(0.5) if win_rates is not None else None
                
            except Exception as e:
                print(f"Fold {split.fold_id} 信号生成失败: {e}")
                continue
            
            # 运行单折回测
            fold_result = self.run_backtest(
                df=test_df,
                signals=test_signals,
                scores=test_scores,
                win_rates=test_win_rates
            )
            
            # 累积结果
            fold_results.append({
                'fold_id': split.fold_id,
                'test_start': split.test_start,
                'test_end': split.test_end,
                'return': fold_result.total_return,
                'sharpe': fold_result.sharpe_ratio,
                'trades': fold_result.total_trades
            })
            
            all_trades.extend(fold_result.trade_log)
            
            # 更新资金
            capital = capital * (1 + fold_result.total_return)
            all_equity.extend(fold_result.equity_curve.tolist()[1:])
        
        # 汇总结果
        equity_series = pd.Series(all_equity)
        total_return = (capital - self.initial_capital) / self.initial_capital
        
        returns_series = equity_series.pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns_series.mean() / returns_series.std() if returns_series.std() > 0 else 0
        
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        wins = [t for t in all_trades if t.get('return', 0) > 0]
        win_rate = len(wins) / len(all_trades) if all_trades else 0
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=0,  # TODO
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=0,  # TODO
            total_trades=len(all_trades),
            avg_holding_period=0,
            equity_curve=equity_series,
            trade_log=all_trades,
            fold_results=fold_results
        )


    def _execute_trade(
        self,
        trade_size: float = None,
        shares: float = None,
        price: float = None,
        volume: float = None,
        volatility: float = None,
        is_buy: bool = True
    ) -> Tuple[float, float]:
        """
        执行交易，计算实际成交金额和成本
        
        Returns:
            (实际成交金额, 总成本)
        """
        if shares is None and trade_size is not None:
            shares = trade_size / price
        
        if shares is None or price is None:
            return 0.0, 0.0
        
        trade_amount = shares * price
        
        if self.use_advanced_cost and self.cost_calculator:
            # 使用高级Transaction Cost模型
            cost_detail = self.cost_calculator.calculate_cost(
                trade_size=trade_amount,
                price=price,
                volume=volume or (shares * 10),
                volatility=volatility,
                is_buy=is_buy
            )
            total_cost = cost_detail['total_cost']
        else:
            # 使用简单模型（向后兼容）
            total_cost = trade_amount * (self.commission + self.slippage)
        
        if is_buy:
            # 买入：实际支付 = 交易金额 + 成本
            actual_amount = trade_amount + total_cost
            return shares, total_cost
        else:
            # 卖出：实际收到 = 交易金额 - 成本
            actual_amount = trade_amount - total_cost
            return actual_amount, total_cost
    
    def _check_turnover_constraint(
        self,
        date: pd.Timestamp,
        capital: float,
        action: str
    ) -> bool:
        """
        检查Turnover约束
        
        Args:
            date: 交易日期
            capital: 当前资金
            action: 交易动作 ('BUY' or 'SELL')
            
        Returns:
            True if 允许交易, False if 超过限制
        """
        date_str = date.strftime('%Y-%m-%d')
        
        # 计算当日已用Turnover
        daily_turnover = self.daily_turnover.get(date_str, 0.0)
        
        # 计算本次交易将产生的Turnover
        # Turnover = 交易金额 / 总资产
        # 这里需要知道交易金额，但在这个阶段还不知道，所以先检查当日累计
        # 实际限制在_execute_trade中检查
        
        # 检查单日Turnover限制
        if daily_turnover >= self.max_daily_turnover:
            return False
        
        # 检查单周Turnover限制
        week_key = date.strftime('%Y-W%U')
        weekly_turnover = self.weekly_turnover.get(week_key, 0.0)
        if weekly_turnover >= self.max_weekly_turnover:
            return False
        
        return True
    
    def _check_turnover_with_amount(
        self,
        date: pd.Timestamp,
        capital: float,
        trade_amount: float
    ) -> bool:
        """
        基于实际交易金额检查Turnover约束
        
        Args:
            date: 交易日期
            capital: 当前资金
            trade_amount: 交易金额
            
        Returns:
            True if 允许交易
        """
        if capital <= 0:
            return False
        
        turnover = trade_amount / capital
        
        date_str = date.strftime('%Y-%m-%d')
        daily_turnover = self.daily_turnover.get(date_str, 0.0)
        
        # 检查单日限制
        if daily_turnover + turnover > self.max_daily_turnover:
            return False
        
        # 检查单周限制
        week_key = date.strftime('%Y-W%U')
        weekly_turnover = self.weekly_turnover.get(week_key, 0.0)
        if weekly_turnover + turnover > self.max_weekly_turnover:
            return False
        
        return True
    
    def _update_turnover(
        self,
        date: pd.Timestamp,
        trade_amount: float,
        total_capital: float
    ):
        """
        更新Turnover记录
        
        Args:
            date: 交易日期
            trade_amount: 交易金额
            total_capital: 总资产
        """
        if total_capital <= 0:
            return
        
        turnover = trade_amount / total_capital
        
        # 更新单日Turnover
        date_str = date.strftime('%Y-%m-%d')
        self.daily_turnover[date_str] = self.daily_turnover.get(date_str, 0.0) + turnover
        
        # 更新单周Turnover
        week_key = date.strftime('%Y-W%U')
        self.weekly_turnover[week_key] = self.weekly_turnover.get(week_key, 0.0) + turnover


def statistical_test(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> Dict:
    """
    统计显著性检验
    
    Args:
        strategy_returns: 策略收益率序列
        benchmark_returns: 基准收益率序列
        
    Returns:
        检验结果
    """
    from scipy import stats
    
    # 配对t检验
    excess_returns = strategy_returns - benchmark_returns
    t_stat, p_value = stats.ttest_1samp(excess_returns.dropna(), 0)
    
    # Wilcoxon符号秩检验（非参数检验）
    try:
        w_stat, w_pvalue = stats.wilcoxon(excess_returns.dropna())
    except:
        w_stat, w_pvalue = 0, 1
    
    # 效应量（Cohen's d）
    cohens_d = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
    
    return {
        't_statistic': t_stat,
        't_pvalue': p_value,
        'significant_5pct': p_value < 0.05,
        'significant_1pct': p_value < 0.01,
        'wilcoxon_stat': w_stat,
        'wilcoxon_pvalue': w_pvalue,
        'cohens_d': cohens_d,
        'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
    }


    def validate_dynamic_weights(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        scores: pd.Series,
        win_rates: pd.Series,
        regime_manager = None
    ) -> Dict:
        """
        验证动态权重的有效性
        
        方法：比较固定权重和动态权重的回测结果
        
        Args:
            df: OHLCV数据
            signals: 交易信号
            scores: 评分序列
            win_rates: 胜率序列
            regime_manager: Regime管理器
            
        Returns:
            验证结果字典
        """
        if regime_manager is None:
            from src.strategies.regime_manager import RegimeManager
            regime_manager = RegimeManager()
        
        # 1. 固定权重回测（传统方式）
        fixed_result = self.run_backtest(df, signals, scores, win_rates)
        
        # 2. 动态权重回测（需要重新计算评分）
        # 这里简化处理：假设动态权重只影响评分计算
        # 实际应该重新计算每个时间点的评分
        
        # 计算动态权重的时间序列
        returns = df['Close'].pct_change().dropna()
        dynamic_weights_history = []
        
        for i in range(len(df)):
            if i < 60:  # 需要足够的历史数据
                dynamic_weights_history.append({
                    'kline_factor': 0.5,
                    'fundamental': 0.3,
                    'technical': 0.2
                })
                continue
            
            # 获取当前regime
            current_returns = returns.iloc[:i] if i < len(returns) else returns
            current_regime = regime_manager.get_current_regime(current_returns)
            
            # 获取权重
            weights = regime_manager.get_regime_weights(current_regime)
            dynamic_weights_history.append(weights)
        
        # 3. 比较结果
        comparison = {
            'fixed_weights': {
                'total_return': fixed_result.total_return,
                'sharpe_ratio': fixed_result.sharpe_ratio,
                'max_drawdown': fixed_result.max_drawdown,
                'win_rate': fixed_result.win_rate
            },
            'dynamic_weights': {
                'weights_history': dynamic_weights_history,
                'note': '动态权重回测需要重新计算评分，这里仅提供权重历史'
            },
            'improvement': {
                'note': '需要完整动态权重回测结果才能计算改进幅度'
            }
        }
        
        return comparison


if __name__ == "__main__":
    print("=== 回测引擎测试 ===")
    
    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    df = pd.DataFrame({
        'Open': 100 + np.random.randn(500).cumsum(),
        'High': 100 + np.random.randn(500).cumsum() + 2,
        'Low': 100 + np.random.randn(500).cumsum() - 2,
        'Close': 100 + np.random.randn(500).cumsum(),
        'Volume': np.random.randint(1000000, 10000000, 500)
    }, index=dates)
    
    # 创建简单信号
    signals = pd.Series(0, index=dates)
    signals.iloc[::20] = 1   # 每20天买入
    signals.iloc[10::20] = -1  # 10天后卖出
    
    # 运行回测
    backtester = VisionQuantBacktester(use_kelly=False)
    result = backtester.run_backtest(df, signals)
    
    print(f"\n回测结果:")
    print(f"  总收益: {result.total_return*100:.2f}%")
    print(f"  夏普比率: {result.sharpe_ratio:.2f}")
    print(f"  最大回撤: {result.max_drawdown*100:.2f}%")
    print(f"  胜率: {result.win_rate*100:.1f}%")
    print(f"  交易次数: {result.total_trades}")
    
    print("\n测试完成！")
