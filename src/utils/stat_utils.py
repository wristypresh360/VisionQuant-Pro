import numpy as np
import pandas as pd

def winsorize(series: pd.Series, limits=(0.05, 0.05)) -> pd.Series:
    """去极值处理 (Winsorization)"""
    if series.empty:
        return series
    q_low = series.quantile(limits[0])
    q_high = series.quantile(1.0 - limits[1])
    return series.clip(lower=q_low, upper=q_high)

def calculate_ic_robust(factors: pd.Series, returns: pd.Series) -> float:
    """稳健IC计算 (RankIC + Winsorize)"""
    if len(factors) < 5 or len(returns) < 5:
        return 0.0
    common = factors.index.intersection(returns.index)
    if len(common) < 5:
        return 0.0
    f = winsorize(factors.loc[common])
    r = winsorize(returns.loc[common])
    return f.corr(r, method='spearman')

def calculate_sharpe_robust(returns: pd.Series, risk_free=0.0) -> float:
    """稳健夏普比率 (基于中位数和MAD)"""
    if len(returns) < 5:
        return 0.0
    # 中位数收益
    med_ret = returns.median()
    # MAD (Median Absolute Deviation) 替代标准差
    mad = (returns - med_ret).abs().median()
    sigma = mad * 1.4826  # 正态分布下 MAD -> Sigma 的转换因子
    if sigma <= 1e-6:
        return 0.0
    return (med_ret - risk_free) / sigma * np.sqrt(252)
