"""
Walk-Forward Validation Framework
滚动窗口验证框架

理论基础：
- 金融时序数据具有时间依赖性，不能使用标准的K-Fold交叉验证
- Walk-Forward模拟真实交易场景：只使用历史数据训练，在未来数据上测试
- 防止未来函数（Look-ahead Bias）

方法类型：
1. Anchored Walk-Forward: 训练集起点固定，随时间扩展
2. Rolling Walk-Forward: 训练集窗口滚动，大小固定
3. Purged K-Fold: 带时间间隔的交叉验证

Author: VisionQuant Team
Date: 2026-01
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Generator, Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings


@dataclass
class WalkForwardSplit:
    """单次Walk-Forward划分"""
    fold_id: int
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: datetime
    test_end: datetime
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


class WalkForwardValidator:
    """
    Walk-Forward 验证器
    
    实现滚动窗口的训练-验证-测试划分，模拟真实交易场景
    
    典型配置：
    - 训练集：3年历史数据
    - 验证集：6个月（用于超参数调优）
    - 测试集：6个月（最终评估）
    - 每次滚动：6个月
    """
    
    def __init__(
        self,
        train_period: int = 756,      # 训练期（交易日），约3年
        val_period: int = 126,        # 验证期，约6个月
        test_period: int = 126,       # 测试期，约6个月
        step_size: int = 126,         # 滚动步长，约6个月
        purge_gap: int = 5,           # 训练和测试之间的间隔（防止泄漏）
        embargo_period: int = 5,      # 测试后的禁止期
        anchored: bool = False        # 是否使用锚定模式（训练集起点固定）
    ):
        """
        初始化Walk-Forward验证器
        
        Args:
            train_period: 训练期长度（交易日）
            val_period: 验证期长度
            test_period: 测试期长度
            step_size: 每次滚动的步长
            purge_gap: 训练和验证之间的间隔
            embargo_period: 测试后的禁止期
            anchored: 是否锚定训练集起点
        """
        self.train_period = train_period
        self.val_period = val_period
        self.test_period = test_period
        self.step_size = step_size
        self.purge_gap = purge_gap
        self.embargo_period = embargo_period
        self.anchored = anchored
    
    def split(
        self,
        data: pd.DataFrame,
        date_col: Optional[str] = None
    ) -> Generator[WalkForwardSplit, None, None]:
        """
        生成Walk-Forward划分
        
        Args:
            data: 时序数据DataFrame，索引应为DatetimeIndex
            date_col: 日期列名（如果索引不是日期）
            
        Yields:
            WalkForwardSplit对象
        """
        # 获取日期索引
        if date_col is not None:
            dates = pd.to_datetime(data[date_col])
        elif isinstance(data.index, pd.DatetimeIndex):
            dates = data.index
        else:
            dates = pd.to_datetime(data.index)
        
        n = len(dates)
        min_required = self.train_period + self.val_period + self.test_period + self.purge_gap * 2
        
        if n < min_required:
            warnings.warn(f"数据量不足: {n} < {min_required}，无法进行Walk-Forward验证")
            return
        
        fold_id = 0
        
        if self.anchored:
            # 锚定模式：训练集从头开始，逐渐扩展
            train_start_idx = 0
        else:
            # 滚动模式：训练集起点随时间滚动
            train_start_idx = 0
        
        while True:
            # 计算各期边界
            if self.anchored:
                # 锚定模式：训练期从0开始，长度逐渐增加
                train_end_idx = self.train_period + fold_id * self.step_size
            else:
                # 滚动模式：训练期大小固定
                train_start_idx = fold_id * self.step_size
                train_end_idx = train_start_idx + self.train_period
            
            # 加入purge gap
            val_start_idx = train_end_idx + self.purge_gap
            val_end_idx = val_start_idx + self.val_period
            
            # 加入purge gap
            test_start_idx = val_end_idx + self.purge_gap
            test_end_idx = test_start_idx + self.test_period
            
            # 检查是否超出数据范围
            if test_end_idx > n:
                break
            
            # 生成划分
            yield WalkForwardSplit(
                fold_id=fold_id,
                train_start=dates[train_start_idx if not self.anchored else 0],
                train_end=dates[train_end_idx - 1],
                val_start=dates[val_start_idx],
                val_end=dates[val_end_idx - 1],
                test_start=dates[test_start_idx],
                test_end=dates[test_end_idx - 1],
                train_indices=np.arange(train_start_idx if not self.anchored else 0, train_end_idx),
                val_indices=np.arange(val_start_idx, val_end_idx),
                test_indices=np.arange(test_start_idx, test_end_idx)
            )
            
            fold_id += 1
    
    def get_n_splits(self, data: pd.DataFrame) -> int:
        """
        计算总划分数量
        
        Args:
            data: 数据DataFrame
            
        Returns:
            划分数量
        """
        return sum(1 for _ in self.split(data))
    
    def summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成划分摘要表
        
        Args:
            data: 数据DataFrame
            
        Returns:
            摘要DataFrame
        """
        summaries = []
        for split in self.split(data):
            summaries.append({
                'fold': split.fold_id,
                'train_start': split.train_start,
                'train_end': split.train_end,
                'train_size': len(split.train_indices),
                'val_start': split.val_start,
                'val_end': split.val_end,
                'val_size': len(split.val_indices),
                'test_start': split.test_start,
                'test_end': split.test_end,
                'test_size': len(split.test_indices)
            })
        
        return pd.DataFrame(summaries)


class PurgedKFold:
    """
    Purged K-Fold 交叉验证
    
    适用于金融时序的交叉验证方法：
    1. 在训练集和测试集之间添加间隔（purge）
    2. 在测试集后添加禁止期（embargo）
    3. 防止信息泄漏
    
    参考：de Prado, M. L. (2018). Advances in Financial Machine Learning.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 5,
        embargo_pct: float = 0.01
    ):
        """
        初始化Purged K-Fold
        
        Args:
            n_splits: 折数
            purge_gap: 训练和测试之间的间隔
            embargo_pct: 测试集后禁止期占总数据的比例
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        生成Purged K-Fold划分
        
        Args:
            X: 特征数组
            y: 标签数组（可选）
            groups: 分组数组（可选）
            
        Yields:
            (train_indices, test_indices) 元组
        """
        n = len(X)
        indices = np.arange(n)
        
        # 计算每折大小
        fold_size = n // self.n_splits
        embargo_size = int(n * self.embargo_pct)
        
        for fold in range(self.n_splits):
            # 测试集范围
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n
            
            test_indices = indices[test_start:test_end]
            
            # 训练集：排除测试集、purge gap 和 embargo
            train_mask = np.ones(n, dtype=bool)
            
            # 排除测试集
            train_mask[test_start:test_end] = False
            
            # 排除purge gap（测试集前）
            purge_start = max(0, test_start - self.purge_gap)
            train_mask[purge_start:test_start] = False
            
            # 排除embargo（测试集后）
            embargo_end = min(n, test_end + embargo_size)
            train_mask[test_end:embargo_end] = False
            
            train_indices = indices[train_mask]
            
            yield train_indices, test_indices
    
    def get_n_splits(self) -> int:
        """返回折数"""
        return self.n_splits


class TimeSeriesSplitter:
    """
    通用时序划分器
    
    支持多种划分策略
    """
    
    @staticmethod
    def train_test_split(
        data: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        gap: int = 5
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        简单的训练-验证-测试划分
        
        Args:
            data: 数据DataFrame
            test_size: 测试集比例
            val_size: 验证集比例
            gap: 各集之间的间隔
            
        Returns:
            (train_df, val_df, test_df)
        """
        n = len(data)
        
        test_start = int(n * (1 - test_size))
        val_start = int(n * (1 - test_size - val_size)) - gap
        
        train_df = data.iloc[:val_start]
        val_df = data.iloc[val_start + gap:test_start - gap]
        test_df = data.iloc[test_start:]
        
        return train_df, val_df, test_df
    
    @staticmethod
    def expanding_window(
        data: pd.DataFrame,
        initial_train_size: int,
        step_size: int,
        test_size: int,
        gap: int = 5
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        扩展窗口划分
        
        训练集逐渐扩大，测试集固定大小滚动
        
        Args:
            data: 数据DataFrame
            initial_train_size: 初始训练集大小
            step_size: 每次扩展的步长
            test_size: 测试集大小
            gap: 间隔
            
        Yields:
            (train_df, test_df)
        """
        n = len(data)
        train_end = initial_train_size
        
        while train_end + gap + test_size <= n:
            train_df = data.iloc[:train_end]
            test_start = train_end + gap
            test_df = data.iloc[test_start:test_start + test_size]
            
            yield train_df, test_df
            
            train_end += step_size
    
    @staticmethod
    def sliding_window(
        data: pd.DataFrame,
        train_size: int,
        test_size: int,
        step_size: int,
        gap: int = 5
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        滑动窗口划分
        
        训练集固定大小滚动
        
        Args:
            data: 数据DataFrame
            train_size: 训练集大小
            test_size: 测试集大小
            step_size: 滑动步长
            gap: 间隔
            
        Yields:
            (train_df, test_df)
        """
        n = len(data)
        train_start = 0
        
        while train_start + train_size + gap + test_size <= n:
            train_df = data.iloc[train_start:train_start + train_size]
            test_start = train_start + train_size + gap
            test_df = data.iloc[test_start:test_start + test_size]
            
            yield train_df, test_df
            
            train_start += step_size


class WalkForwardBacktester:
    """
    Walk-Forward 回测器
    
    集成Walk-Forward验证和回测，确保无未来函数
    """
    
    def __init__(
        self,
        validator: WalkForwardValidator,
        model_factory: Callable,
        strategy_factory: Callable
    ):
        """
        初始化回测器
        
        Args:
            validator: Walk-Forward验证器
            model_factory: 模型工厂函数，返回新的模型实例
            strategy_factory: 策略工厂函数，返回新的策略实例
        """
        self.validator = validator
        self.model_factory = model_factory
        self.strategy_factory = strategy_factory
        self.results = []
    
    def run(
        self,
        data: pd.DataFrame,
        X_cols: List[str],
        y_col: str,
        price_col: str = 'Close'
    ) -> pd.DataFrame:
        """
        运行Walk-Forward回测
        
        Args:
            data: 完整数据
            X_cols: 特征列名
            y_col: 标签列名
            price_col: 价格列名
            
        Returns:
            回测结果DataFrame
        """
        all_results = []
        
        for split in self.validator.split(data):
            print(f"Fold {split.fold_id}: "
                  f"Train {split.train_start.date()} - {split.train_end.date()}, "
                  f"Test {split.test_start.date()} - {split.test_end.date()}")
            
            # 获取训练和测试数据
            train_data = data.iloc[split.train_indices]
            val_data = data.iloc[split.val_indices]
            test_data = data.iloc[split.test_indices]
            
            # 训练模型
            model = self.model_factory()
            X_train = train_data[X_cols].values
            y_train = train_data[y_col].values
            
            # 过滤NaN
            mask = ~np.isnan(y_train)
            X_train = X_train[mask]
            y_train = y_train[mask]
            
            if len(X_train) < 10:
                print(f"  跳过: 训练样本不足 ({len(X_train)})")
                continue
            
            model.fit(X_train, y_train)
            
            # 在验证集上调优（可选）
            # ...
            
            # 在测试集上评估
            X_test = test_data[X_cols].values
            y_test = test_data[y_col].values
            prices = test_data[price_col].values
            
            predictions = model.predict(X_test)
            
            # 计算策略收益
            strategy = self.strategy_factory()
            returns = strategy.calculate_returns(predictions, prices)
            
            # 记录结果
            fold_result = {
                'fold': split.fold_id,
                'test_start': split.test_start,
                'test_end': split.test_end,
                'predictions': predictions,
                'actuals': y_test,
                'returns': returns,
                'total_return': np.prod(1 + returns) - 1,
                'sharpe': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            }
            all_results.append(fold_result)
        
        self.results = all_results
        return self._aggregate_results()
    
    def _aggregate_results(self) -> pd.DataFrame:
        """聚合所有折的结果"""
        if not self.results:
            return pd.DataFrame()
        
        summary = pd.DataFrame([
            {
                'fold': r['fold'],
                'test_start': r['test_start'],
                'test_end': r['test_end'],
                'total_return': r['total_return'],
                'sharpe': r['sharpe']
            }
            for r in self.results
        ])
        
        return summary


def calculate_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    returns: np.ndarray
) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        predictions: 预测值
        actuals: 实际值
        returns: 策略收益率序列
        
    Returns:
        指标字典
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # 分类指标
    pred_labels = (predictions > 0).astype(int)
    actual_labels = (actuals > 0).astype(int)
    
    # 过滤NaN
    mask = ~np.isnan(actuals)
    pred_labels = pred_labels[mask]
    actual_labels = actual_labels[mask]
    
    accuracy = accuracy_score(actual_labels, pred_labels)
    precision = precision_score(actual_labels, pred_labels, zero_division=0)
    recall = recall_score(actual_labels, pred_labels, zero_division=0)
    f1 = f1_score(actual_labels, pred_labels, zero_division=0)
    
    # 收益指标
    returns = returns[~np.isnan(returns)]
    total_return = np.prod(1 + returns) - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
    volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
    sharpe = annual_return / (volatility + 1e-8)
    
    # 最大回撤
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'calmar_ratio': annual_return / (abs(max_drawdown) + 1e-8)
    }


if __name__ == "__main__":
    # 测试Walk-Forward验证器
    import matplotlib.pyplot as plt
    
    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', '2024-12-31', freq='D')
    n = len(dates)
    
    df = pd.DataFrame({
        'Close': 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01)),
        'Volume': np.random.randint(1000000, 10000000, n)
    }, index=dates)
    
    # 测试Walk-Forward验证器
    validator = WalkForwardValidator(
        train_period=504,    # 2年
        val_period=126,      # 6个月
        test_period=126,     # 6个月
        step_size=126,       # 6个月步长
        purge_gap=5,
        anchored=False
    )
    
    print("Walk-Forward Splits:")
    print("=" * 80)
    
    summary = validator.summary(df)
    print(summary.to_string())
    
    print(f"\nTotal splits: {len(summary)}")
    
    # 可视化
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 绘制价格
    ax.plot(df.index, df['Close'], alpha=0.5, label='Price')
    
    # 绘制划分
    colors = plt.cm.Set3(np.linspace(0, 1, len(summary)))
    
    for idx, row in summary.iterrows():
        color = colors[idx]
        
        # 训练期
        ax.axvspan(row['train_start'], row['train_end'], 
                   alpha=0.2, color='blue', label='Train' if idx == 0 else '')
        
        # 验证期
        ax.axvspan(row['val_start'], row['val_end'],
                   alpha=0.2, color='orange', label='Val' if idx == 0 else '')
        
        # 测试期
        ax.axvspan(row['test_start'], row['test_end'],
                   alpha=0.2, color='green', label='Test' if idx == 0 else '')
    
    ax.legend(loc='upper left')
    ax.set_title('Walk-Forward Validation Splits')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    
    plt.tight_layout()
    plt.savefig('walk_forward_test.png', dpi=150)
    print("\nTest image saved to walk_forward_test.png")
