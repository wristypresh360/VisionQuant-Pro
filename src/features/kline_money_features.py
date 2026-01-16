import numpy as np
import pandas as pd
from typing import Dict, Optional


class MoneyFeatureExtractor:
    """
    量价/资金复合特征提取器（基于可得数据）
    - 优先使用 AkShare 可得的 OHLCV
    - 若存在成交额/换手率等字段，则自动增强
    """

    def __init__(self, window: int = 20):
        self.window = window

    def extract(self, df: pd.DataFrame) -> Dict[str, Optional[float]]:
        if df is None or df.empty:
            return {}

        data = df.copy()
        cols = {c.lower(): c for c in data.columns}

        close_col = cols.get("close", "Close")
        high_col = cols.get("high", "High")
        low_col = cols.get("low", "Low")
        vol_col = cols.get("volume", "Volume")

        if close_col not in data or vol_col not in data:
            return {}

        close = data[close_col].astype(float)
        volume = data[vol_col].astype(float)
        high = data[high_col].astype(float) if high_col in data else close
        low = data[low_col].astype(float) if low_col in data else close

        # 成交额与换手率（可选）
        amount_col = self._find_col(data, ["amount", "成交额", "成交金额", "成交额(元)"])
        turnover_col = self._find_col(data, ["turnover", "换手", "换手率"])
        amount = data[amount_col].astype(float) if amount_col else None
        turnover = data[turnover_col].astype(float) if turnover_col else None

        # 量价变化
        ret = close.pct_change()
        vol_chg = volume.pct_change()
        vol_ma = volume.rolling(self.window).mean()
        vol_ratio = (volume / (vol_ma + 1e-8)).iloc[-1]

        # 价格-成交量相关性
        pv_corr = self._rolling_corr(ret, vol_chg, self.window)

        # OBV 与斜率
        direction = np.sign(ret.fillna(0.0))
        obv = (direction * volume).cumsum()
        obv_slope = self._safe_slope(obv, self.window)

        # CMF (Chaikin Money Flow)
        mfm = ((close - low) - (high - close)) / (high - low + 1e-8)
        mfv = mfm * volume
        cmf = (mfv.rolling(self.window).sum() / (volume.rolling(self.window).sum() + 1e-8)).iloc[-1]

        # MFI (Money Flow Index) - 14
        typical = (high + low + close) / 3.0
        raw_mf = typical * volume
        mf_pos = raw_mf.where(typical.diff() > 0, 0.0)
        mf_neg = raw_mf.where(typical.diff() < 0, 0.0)
        mfi = 100 - 100 / (1 + (mf_pos.rolling(14).sum() / (mf_neg.rolling(14).sum() + 1e-8)))
        mfi = float(mfi.iloc[-1]) if len(mfi) > 0 else None

        # 资金流向代理（若有成交额）
        vwap = None
        if amount is not None:
            vwap = (amount / (volume + 1e-8)).iloc[-1]

        return {
            "vol_ratio": float(vol_ratio) if np.isfinite(vol_ratio) else None,
            "pv_corr": float(pv_corr) if pv_corr is not None else None,
            "obv_slope": float(obv_slope) if obv_slope is not None else None,
            "cmf": float(cmf) if np.isfinite(cmf) else None,
            "mfi": float(mfi) if mfi is not None and np.isfinite(mfi) else None,
            "turnover": float(turnover.iloc[-1]) if turnover is not None else None,
            "amount": float(amount.iloc[-1]) if amount is not None else None,
            "vwap": float(vwap) if vwap is not None and np.isfinite(vwap) else None,
        }

    def score(self, features: Dict[str, Optional[float]]) -> float:
        """
        将量价特征映射为 0~1 的资金强度分数（稳健版）
        """
        if not features:
            return 0.5

        score = 0.5

        vol_ratio = features.get("vol_ratio")
        if vol_ratio is not None:
            if vol_ratio >= 1.8:
                score += 0.15
            elif vol_ratio >= 1.2:
                score += 0.08
            elif vol_ratio <= 0.6:
                score -= 0.08

        pv_corr = features.get("pv_corr")
        if pv_corr is not None:
            if pv_corr > 0.2:
                score += 0.06
            elif pv_corr < -0.2:
                score -= 0.06

        obv_slope = features.get("obv_slope")
        if obv_slope is not None:
            score += 0.05 if obv_slope > 0 else -0.05

        cmf = features.get("cmf")
        if cmf is not None:
            score += 0.05 if cmf > 0 else -0.05

        mfi = features.get("mfi")
        if mfi is not None:
            if mfi > 80:
                score -= 0.05  # 过热
            elif mfi < 20:
                score += 0.05  # 低位修复

        turnover = features.get("turnover")
        if turnover is not None:
            if turnover > 5:
                score += 0.04
            elif turnover < 0.5:
                score -= 0.04

        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> Optional[float]:
        try:
            s = pd.concat([a, b], axis=1).dropna()
            if len(s) < window:
                return None
            return float(s.iloc[-window:].corr().iloc[0, 1])
        except Exception:
            return None

    @staticmethod
    def _safe_slope(series: pd.Series, window: int) -> Optional[float]:
        try:
            s = series.dropna()
            if len(s) < window:
                return None
            y = s.iloc[-window:].values
            x = np.arange(len(y))
            coef = np.polyfit(x, y, 1)[0]
            denom = np.mean(np.abs(y)) + 1e-8
            return float(coef / denom)
        except Exception:
            return None

    @staticmethod
    def _find_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
        cols = list(df.columns)
        for c in cols:
            lc = str(c).lower()
            for cand in candidates:
                if cand.lower() in lc:
                    return c
        return None
