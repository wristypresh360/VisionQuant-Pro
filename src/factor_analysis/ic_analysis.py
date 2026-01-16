"""
Rolling IC/Sharpe分析模块 (Enhanced with Robust Statistics)
Rolling Information Coefficient and Sharpe Ratio Analysis

IC (Information Coefficient): 因子值与未来收益率的相关系数
Sharpe Ratio: 因子多空组合的夏普比率

v3.0 Update: 引入稳健统计 (Winsorization, Huber Loss) 减少异常值干扰

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats
import warnings
from src.utils.stat_utils import winsorize, calculate_ic_robust, calculate_sharpe_robust


def calculate_rolling_ic(
    factor_values: pd.Series,
    returns: pd.Series,
    window: int = 252,
    method: str = 'pearson',
    use_robust: bool = True
) -> pd.Series:
    """
    计算滚动窗口的IC序列
    
    Args:
        factor_values: K线学习因子值（胜率或得分）
        returns: 未来收益率序列
        window: 滚动窗口大小（交易日，默认252天=1年）
        method: 相关系数计算方法 ('pearson' 或 'spearman')
        use_robust: 是否使用稳健统计 (Winsorization)
        
    Returns:
        IC序列（与factor_values对齐的索引）
    """
    if len(factor_values) != len(returns):
        raise ValueError("factor_values和returns长度必须一致")
    
    ic_series = []
    dates = []
    
    for i in range(window, len(factor_values)):
        factor_window = factor_values.iloc[i-window:i]
        return_window = returns.iloc[i-window:i]
        
        # 去除NaN
        valid_mask = ~(factor_window.isna() | return_window.isna())
        factor_clean = factor_window[valid_mask]
        return_clean = return_window[valid_mask]
        
        if len(factor_clean) < 10:  # 至少需要10个有效样本
            ic_series.append(np.nan)
        else:
            if use_robust:
                # 使用稳健统计（Winsorization）
                factor_clean = winsorize(factor_clean)
                return_clean = winsorize(return_clean)

            if method == 'pearson':
                ic, _ = stats.pearsonr(factor_clean, return_clean)
            elif method == 'spearman':
                ic, _ = stats.spearmanr(factor_clean, return_clean)
            else:
                raise ValueError(f"不支持的相关系数方法: {method}")
            
            ic_series.append(ic if not np.isnan(ic) else 0.0)
        
        dates.append(factor_values.index[i])
    
    return pd.Series(ic_series, index=dates)


def calculate_rolling_sharpe(
    factor_values: pd.Series,
    returns: pd.Series,
    window: int = 252,
    quantiles: int = 5,
    use_robust: bool = True
) -> pd.Series:
    """
    计算滚动窗口的Sharpe比率
    """
    if len(factor_values) != len(returns):
        raise ValueError("factor_values和returns长度必须一致")
    
    sharpe_series = []
    dates = []
    
    for i in range(window, len(factor_values)):
        factor_window = factor_values.iloc[i-window:i]
        return_window = returns.iloc[i-window:i]
        
        valid_mask = ~(factor_window.isna() | return_window.isna())
        factor_clean = factor_window[valid_mask]
        return_clean = return_window[valid_mask]
        
        if len(factor_clean) < 20:
            sharpe_series.append(np.nan)
        else:
            try:
                factor_quantiles = pd.qcut(factor_clean, q=quantiles, labels=False, duplicates='drop')
                top_mask = factor_quantiles == (quantiles - 1)
                top_returns = return_clean[top_mask]
                bottom_mask = factor_quantiles == 0
                bottom_returns = return_clean[bottom_mask]
                
                if len(top_returns) > 0 and len(bottom_returns) > 0:
                    long_short_return = top_returns.mean() - bottom_returns.mean()
                    
                    if use_robust:
                        # 稳健Sharpe: 基于中位数和MAD
                        all_ls_returns = pd.concat([top_returns, -bottom_returns])
                        sharpe = calculate_sharpe_robust(all_ls_returns)
                    else:
                        # 传统Sharpe
                        # 简化：假设每日多空收益的标准差为所有样本标准差（近似）
                        std = return_clean.std()
                        sharpe = (long_short_return / std) * np.sqrt(252) if std > 0 else 0
                        
                    sharpe_series.append(sharpe)
                else:
                    sharpe_series.append(0.0)
            except Exception:
                sharpe_series.append(0.0)
                
        dates.append(factor_values.index[i])
    
    return pd.Series(sharpe_series, index=dates)


class ICAnalyzer:
    """
    IC分析器
    """
    
    def __init__(self, window: int = 252, method: str = 'pearson'):
        self.window = window
        self.method = method
    
    def analyze(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        method: str = 'pearson'
    ) -> Dict:
        """
        全量分析
        """
        # 1. 滚动IC
        ic_series = calculate_rolling_ic(factor_values, returns, self.window, method)
        
        # 2. 滚动Sharpe
        sharpe_series = calculate_rolling_sharpe(factor_values, returns, self.window)
        
        # 3. 统计指标
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        
        # IC > 0 的比例
        ic_positive_ratio = (ic_series > 0).sum() / len(ic_series) if len(ic_series) > 0 else 0
        
        # t检验
        t_stat, p_value = stats.ttest_1samp(ic_series.dropna(), 0)
        is_significant = p_value < 0.05
        
        # Sharpe统计
        sharpe_mean = sharpe_series.mean()
        sharpe_std = sharpe_series.std()
        sharpe_positive_ratio = (sharpe_series > 0).sum() / len(sharpe_series) if len(sharpe_series) > 0 else 0

        half_life = self._ic_half_life(ic_series)
        stability_score = self._ic_stability_score(ic_series)

        return {
            'ic_series': ic_series,
            'sharpe_series': sharpe_series,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'ic_positive_ratio': ic_positive_ratio,
            'ic_t_stat': t_stat,
            'ic_p_value': p_value,
            'ic_significant': is_significant,
            'sharpe_mean': sharpe_mean,
            'sharpe_std': sharpe_std,
            'sharpe_positive_ratio': sharpe_positive_ratio,
            'half_life': half_life,
            'stability_score': stability_score,
            'summary': {
                'mean_ic': round(ic_mean, 4),
                'std_ic': round(ic_std, 4),
                'ir': round(ic_ir, 4),
                'positive_ratio': round(ic_positive_ratio, 2),
                'significant': is_significant,
                'mean_sharpe': round(sharpe_mean, 4),
                'half_life': None if half_life is None else round(float(half_life), 2),
                'stability_score': round(float(stability_score), 4)
            }
        }

    def analyze_multi_horizon(
        self,
        factor_values: pd.Series,
        returns_map: Dict[int, pd.Series],
        method: str = 'pearson'
    ) -> Dict:
        """
        多持有期IC分析（1/5/10/20天）
        """
        rows = []
        details = {}
        for horizon, ret_series in returns_map.items():
            if ret_series is None or ret_series.empty:
                continue
            aligned = ret_series.reindex(factor_values.index).dropna()
            aligned_f = factor_values.loc[aligned.index]
            if len(aligned_f) < 20:
                continue
            # 动态窗口
            window = min(self.window, max(20, len(aligned_f) // 2))
            window = min(window, max(2, len(aligned_f) - 1))
            analyzer = ICAnalyzer(window=window, method=self.method)
            res = analyzer.analyze(aligned_f, aligned, method=method)
            details[horizon] = res
            rows.append({
                "horizon": horizon,
                "ic_mean": round(res.get("ic_mean", 0), 4),
                "ic_ir": round(res.get("ic_ir", 0), 4),
                "half_life": res.get("half_life"),
                "positive_ratio": round(res.get("ic_positive_ratio", 0), 3)
            })
        matrix = pd.DataFrame(rows).sort_values("horizon") if rows else pd.DataFrame()
        return {"ic_matrix": matrix, "details": details}

    def _ic_half_life(self, ic_series: pd.Series) -> Optional[float]:
        """IC Half-Life（基于AR(1)近似）"""
        s = ic_series.dropna()
        if len(s) < 20:
            return None
        x = s.shift(1).dropna()
        y = s.loc[x.index]
        if len(x) < 10:
            return None
        # 估计AR(1)系数
        phi = np.corrcoef(x.values, y.values)[0, 1]
        if phi is None or phi <= 0 or phi >= 0.999:
            return None
        half_life = -np.log(2) / np.log(phi)
        return float(half_life)

    def _ic_stability_score(self, ic_series: pd.Series) -> float:
        """IC Stability Score（越高越稳定）"""
        s = ic_series.dropna()
        if len(s) == 0:
            return 0.0
        mean_ic = s.mean()
        std_ic = s.std()
        if std_ic <= 0:
            return 1.0
        # 更稳健的稳定性分数：避免均值略小导致直接归零
        ratio = std_ic / (abs(mean_ic) + 1e-6)
        score = 1.0 / (1.0 + ratio)
        return float(np.clip(score, 0.0, 1.0))
