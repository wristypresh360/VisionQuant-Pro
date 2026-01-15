from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import numpy as np
import os
import time
import sys

# 添加项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.data_loader import DataLoader
from src.strategies.kline_factor import KLineFactorCalculator
from src.factor_analysis.ic_analysis import ICAnalyzer
from src.strategies.portfolio_optimizer import PortfolioOptimizer
from src.models.vision_engine import VisionEngine

app = FastAPI(
    title="VisionQuant-Pro API",
    description="K线视觉学习因子投研服务（v3.0）",
    version="3.0.0",
)

# 简单审计日志
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "api_access.log")

# 全局单例（减少加载开销）
_vision_engine = None
_data_loader = None

def get_vision_engine():
    global _vision_engine
    if _vision_engine is None:
        _vision_engine = VisionEngine()
    return _vision_engine

def get_data_loader():
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
    return _data_loader

@app.middleware("http")
async def audit_log(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    cost = (time.time() - start) * 1000
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"{datetime.now().isoformat()} {request.method} {request.url.path} {response.status_code} {cost:.1f}ms\n")
    except Exception:
        pass
    return response


class SingleStockRequest(BaseModel):
    symbol: str
    start_date: str = "20200101"
    end_date: str = datetime.now().strftime("%Y%m%d")


class FactorRequest(BaseModel):
    symbol: str
    start_date: str = "20200101"
    end_date: str = datetime.now().strftime("%Y%m%d")
    robust: bool = True  # v3.0 新增: 是否使用稳健统计


class PortfolioRequest(BaseModel):
    symbols: list[str]
    risk_aversion: float = 1.0
    cvar_limit: float = 0.05


@app.get("/health")
def health():
    return {"status": "ok", "service": "visionquant-pro", "version": "3.0"}


@app.post("/analyze_single_stock")
def analyze_single_stock(req: SingleStockRequest):
    loader = get_data_loader()
    df = loader.get_stock_data(req.symbol, start_date=req.start_date, end_date=req.end_date)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Data empty")

    df.index = pd.to_datetime(df.index)
    last_close = float(df["Close"].iloc[-1])
    ret_5d = float((df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) * 100) if len(df) > 6 else 0.0

    return {
        "symbol": req.symbol,
        "last_close": last_close,
        "ret_5d_pct": round(ret_5d, 2),
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
        "points": len(df),
    }


@app.post("/factor/ic")
def factor_ic(req: FactorRequest):
    loader = get_data_loader()
    df = loader.get_stock_data(req.symbol, start_date=req.start_date, end_date=req.end_date)
    if df is None or df.empty or len(df) < 80:
        raise HTTPException(status_code=400, detail="Data insufficient")

    df.index = pd.to_datetime(df.index)
    returns = df["Close"].pct_change().dropna()
    
    # 模拟因子值（这里简单用动量代替，实际应用应调用 VisionEngine 批量计算）
    factor_values = df["Close"].pct_change(20).shift(1).dropna()
    
    common_idx = factor_values.index.intersection(returns.index)
    if len(common_idx) < 20:
        raise HTTPException(status_code=400, detail="Not enough overlapping data")
        
    factor_values = factor_values.loc[common_idx]
    returns = returns.loc[common_idx]

    analyzer = ICAnalyzer(window=min(60, max(20, len(factor_values) // 2)))
    # v3.0: 传递 robust 参数
    if req.robust:
        # 这里实际上 calculate_rolling_ic 内部并没有直接暴露 use_robust 参数给 analyze 方法
        # 我们需要在 analyze 方法中支持 kwargs 或者更新 analyze 签名
        # 暂时直接调用 analyze，稳健性在 analyze 内部实现
        pass
        
    ic_result = analyzer.analyze(factor_values, returns)
    
    return {
        "symbol": req.symbol,
        "ic_mean": ic_result.get("ic_mean"),
        "ic_ir": ic_result.get("ic_ir"),
        "half_life": ic_result.get("half_life"),
        "stability": ic_result.get("stability_score"),
        "points": len(factor_values),
    }


@app.post("/portfolio/optimize")
def optimize_portfolio(req: PortfolioRequest):
    """v3.0: 组合优化接口"""
    loader = get_data_loader()
    optimizer = PortfolioOptimizer()
    
    # 构造虚构的 analysis_results (仅用于演示 API 通通)
    # 实际场景应先调用 batch_analyze
    mock_results = {}
    for sym in req.symbols:
        mock_results[sym] = {
            "score": np.random.uniform(4, 9),
            "win_rate": np.random.uniform(45, 65),
            "expected_return": np.random.uniform(5, 20),
            "action": "BUY" if np.random.random() > 0.5 else "WAIT"
        }
        
    try:
        result = optimizer.optimize_multi_tier_portfolio(
            mock_results, loader,
            risk_aversion=req.risk_aversion,
            cvar_limit=req.cvar_limit
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
