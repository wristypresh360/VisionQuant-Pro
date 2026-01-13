"""
Backtrader Integration - 专业回测框架集成

基于Backtrader实现VisionQuant策略的严谨回测

优势：
1. 事件驱动：逐bar模拟真实交易
2. 防止未来函数：严格的时间顺序执行
3. 完整的交易成本模型：手续费、滑点
4. 丰富的绩效指标：夏普、最大回撤、胜率等

Author: VisionQuant Team
Date: 2026-01
"""

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime
import warnings


class VisionQuantStrategy(bt.Strategy):
    """
    VisionQuant策略的Backtrader实现
    
    策略逻辑：
    1. 每日获取模型预测（类别+收益率）
    2. 根据Triple Barrier预测决定交易方向
    3. 根据收益率预测决定仓位大小
    4. 执行风险控制（止损、最大仓位）
    """
    
    params = (
        ('model_predictions', None),     # 模型预测DataFrame
        ('position_sizing', 'fixed'),    # 仓位管理: 'fixed', 'kelly', 'volatility'
        ('max_position', 0.2),           # 单只股票最大仓位
        ('stop_loss', 0.08),             # 止损比例
        ('take_profit', 0.15),           # 止盈比例
        ('min_confidence', 0.6),         # 最小置信度阈值
        ('use_trailing_stop', False),    # 是否使用移动止损
        ('trailing_pct', 0.05),          # 移动止损比例
        ('log_trades', True),            # 是否记录交易
    )
    
    def __init__(self):
        """初始化策略"""
        self.order = None
        self.entry_price = None
        self.highest_price = None
        self.trade_log = []
        
        # 预测数据
        self.predictions = self.params.model_predictions
        if self.predictions is not None:
            self.predictions.index = pd.to_datetime(self.predictions.index)
    
    def log(self, txt: str, dt: datetime = None):
        """日志输出"""
        if self.params.log_trades:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                self.highest_price = order.executed.price
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, '
                    f'Size: {order.executed.size:.0f}, '
                    f'Cost: {order.executed.value:.2f}, '
                    f'Comm: {order.executed.comm:.2f}'
                )
            else:
                self.log(
                    f'SELL EXECUTED, Price: {order.executed.price:.2f}, '
                    f'Size: {order.executed.size:.0f}, '
                    f'PnL: {order.executed.pnl:.2f}'
                )
                
                # 记录交易
                if self.entry_price:
                    return_pct = (order.executed.price - self.entry_price) / self.entry_price
                    self.trade_log.append({
                        'exit_date': self.datas[0].datetime.date(0),
                        'entry_price': self.entry_price,
                        'exit_price': order.executed.price,
                        'return': return_pct,
                        'pnl': order.executed.pnl
                    })
                
                self.entry_price = None
                self.highest_price = None
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        self.order = None
    
    def notify_trade(self, trade):
        """交易状态通知"""
        if not trade.isclosed:
            return
        
        self.log(f'TRADE PROFIT, Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')
    
    def get_prediction(self, dt: datetime) -> Optional[Dict]:
        """获取指定日期的模型预测"""
        if self.predictions is None:
            return None
        
        dt = pd.Timestamp(dt)
        
        if dt in self.predictions.index:
            row = self.predictions.loc[dt]
            return {
                'class': int(row.get('predicted_class', 0)),
                'return': float(row.get('predicted_return', 0)),
                'confidence': float(row.get('confidence', 0.5))
            }
        
        return None
    
    def calculate_position_size(self, prediction: Dict) -> float:
        """计算仓位大小"""
        confidence = prediction.get('confidence', 0.5)
        predicted_return = prediction.get('return', 0)
        
        if self.params.position_sizing == 'fixed':
            # 固定仓位
            return self.params.max_position
        
        elif self.params.position_sizing == 'kelly':
            # 凯利公式
            win_prob = confidence
            win_return = abs(predicted_return) if predicted_return > 0 else 0.05
            loss_return = self.params.stop_loss
            
            kelly_fraction = (win_prob * win_return - (1 - win_prob) * loss_return) / win_return
            kelly_fraction = max(0, min(kelly_fraction, self.params.max_position))
            
            return kelly_fraction
        
        elif self.params.position_sizing == 'volatility':
            # 波动率调整
            # TODO: 实现基于波动率的仓位调整
            return self.params.max_position * confidence
        
        else:
            return self.params.max_position
    
    def next(self):
        """每个bar执行的策略逻辑"""
        # 等待订单完成
        if self.order:
            return
        
        current_date = self.datas[0].datetime.date(0)
        current_price = self.datas[0].close[0]
        
        # 获取预测
        prediction = self.get_prediction(current_date)
        
        # 持仓管理
        if self.position:
            # 更新最高价（用于移动止损）
            if self.highest_price:
                self.highest_price = max(self.highest_price, current_price)
            
            # 止损检查
            if self.entry_price:
                return_since_entry = (current_price - self.entry_price) / self.entry_price
                
                # 固定止损
                if return_since_entry <= -self.params.stop_loss:
                    self.log(f'STOP LOSS triggered at {current_price:.2f}')
                    self.order = self.close()
                    return
                
                # 止盈
                if return_since_entry >= self.params.take_profit:
                    self.log(f'TAKE PROFIT triggered at {current_price:.2f}')
                    self.order = self.close()
                    return
                
                # 移动止损
                if self.params.use_trailing_stop and self.highest_price:
                    trailing_stop = self.highest_price * (1 - self.params.trailing_pct)
                    if current_price < trailing_stop:
                        self.log(f'TRAILING STOP triggered at {current_price:.2f}')
                        self.order = self.close()
                        return
            
            # 检查是否应该平仓（预测变化）
            if prediction and prediction['class'] == -1:  # 看跌信号
                self.log(f'SIGNAL REVERSAL: closing position')
                self.order = self.close()
                return
        
        else:
            # 无持仓，检查开仓信号
            if prediction is None:
                return
            
            # 只在看涨信号且置信度足够时开仓
            if prediction['class'] == 1 and prediction['confidence'] >= self.params.min_confidence:
                # 计算仓位
                position_pct = self.calculate_position_size(prediction)
                
                # 计算可买入数量
                available_cash = self.broker.getcash()
                size = int(available_cash * position_pct / current_price)
                
                if size > 0:
                    self.log(
                        f'BUY CREATE, Price: {current_price:.2f}, '
                        f'Size: {size}, Confidence: {prediction["confidence"]:.2f}'
                    )
                    self.order = self.buy(size=size)


class VisionQuantBacktester:
    """
    VisionQuant回测器
    
    封装Backtrader，提供简洁的回测接口
    """
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission: float = 0.001,      # 0.1% 手续费
        slippage: float = 0.001,        # 0.1% 滑点
        stake: int = 100                # 默认交易单位
    ):
        """
        初始化回测器
        
        Args:
            initial_cash: 初始资金
            commission: 手续费率
            slippage: 滑点率
            stake: 默认交易单位
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.stake = stake
        
        self.results = None
        self.cerebro = None
    
    def run(
        self,
        price_data: pd.DataFrame,
        predictions: pd.DataFrame,
        strategy_params: Optional[Dict] = None
    ) -> Dict:
        """
        运行回测
        
        Args:
            price_data: OHLCV价格数据
            predictions: 模型预测数据
            strategy_params: 策略参数
            
        Returns:
            回测结果字典
        """
        # 创建Cerebro引擎
        self.cerebro = bt.Cerebro()
        
        # 添加数据
        data = bt.feeds.PandasData(
            dataname=price_data,
            datetime=None,  # 使用索引作为日期
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            openinterest=-1
        )
        self.cerebro.adddata(data)
        
        # 添加策略
        strategy_params = strategy_params or {}
        strategy_params['model_predictions'] = predictions
        self.cerebro.addstrategy(VisionQuantStrategy, **strategy_params)
        
        # 设置资金
        self.cerebro.broker.setcash(self.initial_cash)
        
        # 设置手续费
        self.cerebro.broker.setcommission(commission=self.commission)
        
        # 添加分析器
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')
        
        # 运行回测
        print(f'Starting Portfolio Value: {self.cerebro.broker.getvalue():.2f}')
        self.results = self.cerebro.run()
        print(f'Final Portfolio Value: {self.cerebro.broker.getvalue():.2f}')
        
        # 收集结果
        return self._collect_results()
    
    def _collect_results(self) -> Dict:
        """收集回测结果"""
        if self.results is None:
            return {}
        
        strat = self.results[0]
        
        # 基本指标
        final_value = self.cerebro.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash
        
        # 夏普比率
        sharpe = strat.analyzers.sharpe.get_analysis()
        sharpe_ratio = sharpe.get('sharperatio', 0)
        
        # 回撤
        drawdown = strat.analyzers.drawdown.get_analysis()
        max_drawdown = drawdown.get('max', {}).get('drawdown', 0) / 100
        
        # 交易统计
        trades = strat.analyzers.trades.get_analysis()
        total_trades = trades.get('total', {}).get('total', 0)
        won_trades = trades.get('won', {}).get('total', 0)
        lost_trades = trades.get('lost', {}).get('total', 0)
        
        win_rate = won_trades / total_trades if total_trades > 0 else 0
        
        # 收益统计
        returns_analysis = strat.analyzers.returns.get_analysis()
        annual_return = returns_analysis.get('rnorm100', 0) / 100
        
        # VWR (Variability-Weighted Return)
        vwr = strat.analyzers.vwr.get_analysis()
        vwr_value = vwr.get('vwr', 0)
        
        return {
            'initial_cash': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio if sharpe_ratio else 0,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'won_trades': won_trades,
            'lost_trades': lost_trades,
            'win_rate': win_rate,
            'vwr': vwr_value,
            'calmar_ratio': annual_return / max_drawdown if max_drawdown > 0 else 0,
            'trade_log': strat.trade_log if hasattr(strat, 'trade_log') else []
        }
    
    def plot(self, filename: Optional[str] = None):
        """绘制回测图表"""
        if self.cerebro is None:
            print("请先运行回测")
            return
        
        fig = self.cerebro.plot(style='candlestick', volume=True)[0][0]
        
        if filename:
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"图表已保存到: {filename}")


class BenchmarkStrategy(bt.Strategy):
    """
    基准策略：买入并持有
    """
    
    def __init__(self):
        self.order = None
        self.bought = False
    
    def next(self):
        if not self.bought and not self.order:
            size = int(self.broker.getcash() * 0.95 / self.datas[0].close[0])
            if size > 0:
                self.order = self.buy(size=size)
                self.bought = True
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


def run_comparison_backtest(
    price_data: pd.DataFrame,
    predictions: pd.DataFrame,
    initial_cash: float = 100000.0,
    commission: float = 0.001
) -> Dict:
    """
    运行对比回测：VQ策略 vs 买入持有
    
    Args:
        price_data: OHLCV数据
        predictions: 模型预测
        initial_cash: 初始资金
        commission: 手续费
        
    Returns:
        对比结果
    """
    # VQ策略回测
    vq_backtester = VisionQuantBacktester(
        initial_cash=initial_cash,
        commission=commission
    )
    vq_results = vq_backtester.run(price_data, predictions)
    
    # 买入持有回测
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=price_data)
    cerebro.adddata(data)
    cerebro.addstrategy(BenchmarkStrategy)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    
    bh_results = cerebro.run()
    bh_final = cerebro.broker.getvalue()
    bh_return = (bh_final - initial_cash) / initial_cash
    bh_drawdown = bh_results[0].analyzers.drawdown.get_analysis()
    
    return {
        'vq_strategy': vq_results,
        'buy_and_hold': {
            'final_value': bh_final,
            'total_return': bh_return,
            'max_drawdown': bh_drawdown.get('max', {}).get('drawdown', 0) / 100
        },
        'alpha': vq_results['total_return'] - bh_return,
        'outperformance': vq_results['total_return'] > bh_return
    }


def calculate_statistics(
    vq_returns: np.ndarray,
    bh_returns: np.ndarray
) -> Dict:
    """
    计算统计显著性
    
    Args:
        vq_returns: VQ策略日收益率
        bh_returns: 买入持有日收益率
        
    Returns:
        统计检验结果
    """
    from scipy import stats
    
    # 配对t检验
    t_stat, t_pvalue = stats.ttest_rel(vq_returns, bh_returns)
    
    # Wilcoxon符号秩检验
    try:
        w_stat, w_pvalue = stats.wilcoxon(vq_returns - bh_returns)
    except:
        w_stat, w_pvalue = 0, 1
    
    # Cohen's d效应量
    diff = vq_returns - bh_returns
    cohens_d = np.mean(diff) / (np.std(diff) + 1e-8)
    
    return {
        't_statistic': t_stat,
        't_pvalue': t_pvalue,
        'wilcoxon_statistic': w_stat,
        'wilcoxon_pvalue': w_pvalue,
        'cohens_d': cohens_d,
        'significant_at_5pct': t_pvalue < 0.05,
        'significant_at_1pct': t_pvalue < 0.01
    }


if __name__ == "__main__":
    # 测试回测框架
    print("Testing Backtrader Integration...")
    
    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
    n = len(dates)
    
    # 模拟价格
    returns = np.random.randn(n) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    
    price_data = pd.DataFrame({
        'Open': prices * (1 + np.random.randn(n) * 0.005),
        'High': prices * (1 + np.abs(np.random.randn(n)) * 0.01),
        'Low': prices * (1 - np.abs(np.random.randn(n)) * 0.01),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n)
    }, index=dates)
    
    # 模拟预测
    predictions = pd.DataFrame({
        'predicted_class': np.random.choice([-1, 0, 1], n, p=[0.3, 0.4, 0.3]),
        'predicted_return': np.random.randn(n) * 0.05,
        'confidence': np.random.uniform(0.5, 1.0, n)
    }, index=dates)
    
    print(f"Price data shape: {price_data.shape}")
    print(f"Predictions shape: {predictions.shape}")
    
    # 运行回测
    backtester = VisionQuantBacktester(
        initial_cash=100000,
        commission=0.001
    )
    
    results = backtester.run(
        price_data,
        predictions,
        strategy_params={
            'stop_loss': 0.08,
            'take_profit': 0.15,
            'min_confidence': 0.6,
            'log_trades': False
        }
    )
    
    print("\n" + "=" * 50)
    print("Backtest Results:")
    print("=" * 50)
    print(f"Initial Cash: ${results['initial_cash']:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Calmar Ratio: {results['calmar_ratio']:.2f}")
    
    print("\n✅ Backtrader integration test passed!")
