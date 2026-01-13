"""
Triple Barrier Method 标签生成器

理论基础：
- López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
- Chapter 3: Meta-Labeling

三重障碍法是金融机器学习中标准的标签定义方法，考虑：
1. 止盈障碍 (Profit-Taking): 价格上涨达到阈值
2. 止损障碍 (Stop-Loss): 价格下跌达到阈值
3. 时间障碍 (Max Holding Period): 达到最大持有期

优势：
- 直接对接交易策略（止盈/止损）
- 避免简单涨跌二分类的信息损失
- 考虑交易成本和滑点

Author: VisionQuant Team
Date: 2026-01
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List
from dataclasses import dataclass
from enum import IntEnum


class BarrierLabel(IntEnum):
    """Triple Barrier 标签枚举"""
    LOSS = -1      # 先触及止损
    NEUTRAL = 0    # 先触及时间边界（震荡）
    PROFIT = 1     # 先触及止盈


@dataclass
class BarrierEvent:
    """单个障碍事件的详细信息"""
    entry_date: pd.Timestamp        # 入场日期
    entry_price: float              # 入场价格
    exit_date: pd.Timestamp         # 出场日期
    exit_price: float               # 出场价格
    label: BarrierLabel             # 标签
    barrier_type: str               # 触发的障碍类型: 'profit', 'loss', 'time'
    return_pct: float               # 实际收益率
    holding_days: int               # 持有天数
    

class TripleBarrierLabeler:
    """
    Triple Barrier Method 标签生成器
    
    核心思想：
    - 在每个时间点t，设置三个障碍：
      1. 上障碍: entry_price * (1 + pt)
      2. 下障碍: entry_price * (1 - sl)
      3. 时间障碍: t + max_holding
    - 标签取决于哪个障碍先被触及
    
    使用场景：
    - 监督学习的标签定义
    - 策略信号的元标签（Meta-Labeling）
    """
    
    def __init__(
        self,
        profit_taking: float = 0.05,     # 止盈阈值 5%
        stop_loss: float = 0.03,          # 止损阈值 3%
        max_holding: int = 20,            # 最大持有期（交易日）
        use_volatility_scaling: bool = True,  # 是否使用波动率调整阈值
        volatility_window: int = 20,      # 波动率计算窗口
        min_samples: int = 5              # 最小样本数
    ):
        """
        初始化Triple Barrier标签器
        
        Args:
            profit_taking: 止盈阈值（相对收益率）
            stop_loss: 止损阈值（相对收益率）
            max_holding: 最大持有期（天数）
            use_volatility_scaling: 是否根据波动率动态调整阈值
            volatility_window: 波动率计算窗口
            min_samples: 窗口内最少数据点
        """
        self.pt = profit_taking
        self.sl = stop_loss
        self.max_holding = max_holding
        self.use_volatility_scaling = use_volatility_scaling
        self.volatility_window = volatility_window
        self.min_samples = min_samples
    
    def _compute_volatility(self, close: pd.Series) -> pd.Series:
        """
        计算滚动波动率（年化）
        
        使用对数收益率的标准差
        """
        log_returns = np.log(close / close.shift(1))
        volatility = log_returns.rolling(window=self.volatility_window).std()
        # 年化 (假设252个交易日)
        return volatility * np.sqrt(252)
    
    def _get_barriers(
        self,
        entry_price: float,
        volatility: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        计算止盈止损价格
        
        如果启用波动率调整，阈值会根据当前波动率缩放
        """
        pt = self.pt
        sl = self.sl
        
        if self.use_volatility_scaling and volatility is not None and volatility > 0:
            # 波动率调整：波动率越高，阈值越宽
            vol_factor = volatility / 0.2  # 以20%年化波动率为基准
            vol_factor = np.clip(vol_factor, 0.5, 2.0)  # 限制在0.5-2倍
            pt = pt * vol_factor
            sl = sl * vol_factor
        
        upper_barrier = entry_price * (1 + pt)
        lower_barrier = entry_price * (1 - sl)
        
        return upper_barrier, lower_barrier
    
    def get_label_single(
        self,
        close: pd.Series,
        entry_idx: int,
        volatility: Optional[float] = None
    ) -> BarrierEvent:
        """
        计算单个时间点的Triple Barrier标签
        
        Args:
            close: 收盘价序列
            entry_idx: 入场时间索引
            volatility: 当前波动率（可选）
            
        Returns:
            BarrierEvent对象，包含标签和详细信息
        """
        entry_price = close.iloc[entry_idx]
        entry_date = close.index[entry_idx]
        
        # 获取障碍价格
        upper, lower = self._get_barriers(entry_price, volatility)
        
        # 设置时间障碍
        exit_idx = min(entry_idx + self.max_holding, len(close) - 1)
        
        # 在持有期内搜索首先触及的障碍
        for i in range(entry_idx + 1, exit_idx + 1):
            current_price = close.iloc[i]
            
            # 检查是否触及上障碍（止盈）
            if current_price >= upper:
                return BarrierEvent(
                    entry_date=entry_date,
                    entry_price=entry_price,
                    exit_date=close.index[i],
                    exit_price=current_price,
                    label=BarrierLabel.PROFIT,
                    barrier_type='profit',
                    return_pct=(current_price - entry_price) / entry_price,
                    holding_days=i - entry_idx
                )
            
            # 检查是否触及下障碍（止损）
            if current_price <= lower:
                return BarrierEvent(
                    entry_date=entry_date,
                    entry_price=entry_price,
                    exit_date=close.index[i],
                    exit_price=current_price,
                    label=BarrierLabel.LOSS,
                    barrier_type='loss',
                    return_pct=(current_price - entry_price) / entry_price,
                    holding_days=i - entry_idx
                )
        
        # 触及时间障碍（震荡）
        exit_price = close.iloc[exit_idx]
        return BarrierEvent(
            entry_date=entry_date,
            entry_price=entry_price,
            exit_date=close.index[exit_idx],
            exit_price=exit_price,
            label=BarrierLabel.NEUTRAL,
            barrier_type='time',
            return_pct=(exit_price - entry_price) / entry_price,
            holding_days=exit_idx - entry_idx
        )
    
    def get_labels(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close'
    ) -> pd.DataFrame:
        """
        为整个数据集生成Triple Barrier标签
        
        Args:
            df: 包含价格数据的DataFrame
            price_col: 价格列名
            
        Returns:
            添加了标签列的DataFrame
        """
        close = df[price_col]
        n = len(close)
        
        # 计算波动率（如果需要）
        volatility = None
        if self.use_volatility_scaling:
            volatility = self._compute_volatility(close)
        
        # 生成标签
        labels = []
        returns = []
        holding_days = []
        barrier_types = []
        
        for i in range(n - self.min_samples):
            vol = volatility.iloc[i] if volatility is not None else None
            
            if pd.isna(vol) and self.use_volatility_scaling:
                # 波动率数据不足，使用默认阈值
                vol = None
            
            event = self.get_label_single(close, i, vol)
            labels.append(int(event.label))
            returns.append(event.return_pct)
            holding_days.append(event.holding_days)
            barrier_types.append(event.barrier_type)
        
        # 填充最后几行（没有足够的未来数据）
        for _ in range(self.min_samples):
            labels.append(np.nan)
            returns.append(np.nan)
            holding_days.append(np.nan)
            barrier_types.append(None)
        
        # 创建结果DataFrame
        result = df.copy()
        result['tb_label'] = labels
        result['tb_return'] = returns
        result['tb_holding_days'] = holding_days
        result['tb_barrier_type'] = barrier_types
        
        return result
    
    def get_label_distribution(self, labels: pd.Series) -> dict:
        """
        统计标签分布
        
        Args:
            labels: 标签序列
            
        Returns:
            分布统计字典
        """
        labels = labels.dropna()
        total = len(labels)
        
        if total == 0:
            return {'profit': 0, 'neutral': 0, 'loss': 0, 'total': 0}
        
        return {
            'profit': (labels == 1).sum() / total,
            'neutral': (labels == 0).sum() / total,
            'loss': (labels == -1).sum() / total,
            'total': total
        }


class MetaLabeler:
    """
    元标签器 (Meta-Labeling)
    
    用于增强现有交易信号：
    - 基础模型提供交易方向（买/卖）
    - 元标签器预测该交易是否会盈利
    
    优势：
    - 提高精确率（减少假阳性）
    - 保持召回率（不错过好机会）
    - 可以融合多个策略信号
    """
    
    def __init__(self, base_labeler: TripleBarrierLabeler):
        """
        初始化元标签器
        
        Args:
            base_labeler: 基础Triple Barrier标签器
        """
        self.base_labeler = base_labeler
    
    def get_meta_labels(
        self,
        df: pd.DataFrame,
        signal_col: str,
        price_col: str = 'Close'
    ) -> pd.DataFrame:
        """
        生成元标签
        
        对于基础信号指示的每个交易机会，判断是否应该执行
        
        Args:
            df: 数据DataFrame
            signal_col: 基础信号列（1=买入信号, -1=卖出信号, 0=无信号）
            price_col: 价格列
            
        Returns:
            添加元标签的DataFrame
        """
        result = df.copy()
        
        # 首先获取Triple Barrier标签
        tb_df = self.base_labeler.get_labels(df, price_col)
        
        # 元标签：当基础信号与实际结果一致时为1，否则为0
        meta_labels = []
        
        for i in range(len(df)):
            signal = df[signal_col].iloc[i]
            tb_label = tb_df['tb_label'].iloc[i]
            
            if pd.isna(tb_label) or signal == 0:
                meta_labels.append(np.nan)
            elif signal > 0:
                # 买入信号：如果实际上涨则元标签为1
                meta_labels.append(1 if tb_label == 1 else 0)
            else:
                # 卖出信号：如果实际下跌则元标签为1
                meta_labels.append(1 if tb_label == -1 else 0)
        
        result['meta_label'] = meta_labels
        result['tb_label'] = tb_df['tb_label']
        result['tb_return'] = tb_df['tb_return']
        
        return result


class QuantileLabeler:
    """
    分位数标签器
    
    将收益率按分位数分为多个类别，用于更细粒度的预测
    """
    
    def __init__(
        self,
        n_classes: int = 5,
        forward_period: int = 5,
        quantiles: Optional[List[float]] = None
    ):
        """
        初始化分位数标签器
        
        Args:
            n_classes: 类别数量
            forward_period: 前向收益计算周期
            quantiles: 自定义分位数（如果为None，则均匀分布）
        """
        self.n_classes = n_classes
        self.forward_period = forward_period
        
        if quantiles is None:
            self.quantiles = np.linspace(0, 1, n_classes + 1)[1:-1]
        else:
            self.quantiles = quantiles
    
    def get_labels(
        self,
        df: pd.DataFrame,
        price_col: str = 'Close'
    ) -> pd.DataFrame:
        """
        生成分位数标签
        
        Args:
            df: 数据DataFrame
            price_col: 价格列
            
        Returns:
            添加标签的DataFrame
        """
        result = df.copy()
        close = df[price_col]
        
        # 计算前向收益率
        forward_returns = close.shift(-self.forward_period) / close - 1
        
        # 计算滚动分位数边界（使用过去数据，避免未来函数）
        rolling_quantiles = forward_returns.rolling(
            window=252, min_periods=60
        ).quantile(self.quantiles[0])
        
        # 简化版本：使用全局分位数（用于训练集）
        boundaries = forward_returns.quantile(self.quantiles)
        
        # 生成标签
        labels = np.digitize(forward_returns, boundaries)
        
        result['forward_return'] = forward_returns
        result['quantile_label'] = labels
        
        return result


def generate_triple_barrier_dataset(
    data_loader,
    symbols: List[str],
    output_path: str,
    labeler: Optional[TripleBarrierLabeler] = None
) -> pd.DataFrame:
    """
    批量生成Triple Barrier标签数据集
    
    Args:
        data_loader: 数据加载器
        symbols: 股票代码列表
        output_path: 输出CSV路径
        labeler: 标签器实例（可选）
        
    Returns:
        合并的DataFrame
    """
    from tqdm import tqdm
    
    if labeler is None:
        labeler = TripleBarrierLabeler()
    
    all_data = []
    
    for symbol in tqdm(symbols, desc="Generating labels"):
        try:
            df = data_loader.get_stock_data(symbol)
            if df is None or len(df) < 100:
                continue
            
            # 生成标签
            labeled_df = labeler.get_labels(df)
            labeled_df['symbol'] = symbol
            
            all_data.append(labeled_df)
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue
    
    if not all_data:
        return pd.DataFrame()
    
    # 合并所有数据
    result = pd.concat(all_data, ignore_index=False)
    
    # 保存
    result.to_csv(output_path)
    print(f"Saved {len(result)} samples to {output_path}")
    
    return result


if __name__ == "__main__":
    # 测试Triple Barrier标签器
    import matplotlib.pyplot as plt
    
    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # 模拟股价（带趋势和波动）
    returns = np.random.randn(500) * 0.02
    returns[:100] += 0.001  # 上涨趋势
    returns[100:200] -= 0.001  # 下跌趋势
    returns[200:300] += 0.0005  # 温和上涨
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'Close': prices,
        'Open': prices * (1 + np.random.randn(500) * 0.005),
        'High': prices * (1 + np.abs(np.random.randn(500)) * 0.01),
        'Low': prices * (1 - np.abs(np.random.randn(500)) * 0.01),
        'Volume': np.random.randint(1000000, 10000000, 500)
    }, index=dates)
    
    # 测试标签器
    labeler = TripleBarrierLabeler(
        profit_taking=0.05,
        stop_loss=0.03,
        max_holding=20,
        use_volatility_scaling=True
    )
    
    labeled_df = labeler.get_labels(df)
    
    # 统计分布
    dist = labeler.get_label_distribution(labeled_df['tb_label'])
    print(f"\nLabel Distribution:")
    print(f"  Profit (1): {dist['profit']:.2%}")
    print(f"  Neutral (0): {dist['neutral']:.2%}")
    print(f"  Loss (-1): {dist['loss']:.2%}")
    print(f"  Total samples: {dist['total']}")
    
    # 可视化
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 价格走势
    axes[0].plot(labeled_df.index, labeled_df['Close'], label='Close Price')
    
    # 标记不同标签
    profit_mask = labeled_df['tb_label'] == 1
    loss_mask = labeled_df['tb_label'] == -1
    neutral_mask = labeled_df['tb_label'] == 0
    
    axes[0].scatter(
        labeled_df.index[profit_mask],
        labeled_df['Close'][profit_mask],
        c='green', s=10, alpha=0.5, label='Profit'
    )
    axes[0].scatter(
        labeled_df.index[loss_mask],
        labeled_df['Close'][loss_mask],
        c='red', s=10, alpha=0.5, label='Loss'
    )
    axes[0].legend()
    axes[0].set_title('Price with Triple Barrier Labels')
    
    # 收益分布
    axes[1].hist(
        labeled_df['tb_return'].dropna(),
        bins=50,
        edgecolor='black',
        alpha=0.7
    )
    axes[1].axvline(x=0, color='red', linestyle='--')
    axes[1].set_title('Return Distribution')
    axes[1].set_xlabel('Return')
    
    plt.tight_layout()
    plt.savefig('triple_barrier_test.png', dpi=150)
    print("\nTest image saved to triple_barrier_test.png")
