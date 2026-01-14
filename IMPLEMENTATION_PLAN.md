# VisionQuant-Pro v2.0: K线学习因子深化方案 - 最终确认版

## 用户最终选择汇总

### 核心功能选择
- 胜率计算: **B. 混合方法** (Triple Barrier + 传统胜率)
- 因子有效性分析: **全选** (Rolling IC/Sharpe, Regime, 衰减, 拥挤, 风险补偿, 行业分层)
- 数据源: **A. 聚宽/米筐**
- 学习精度提升: **全选** (维度提升, 多尺度, 双流优化, 对比学习)
- 其他因子处理: **C. 动态权重**
- 回测范围: **B. 分层回测**
- 因子研究维度: **全选** (行为偏差, 信息扩散, 相关性, 稳定性, 组合优化)
- UI方案: **D. 渐进式** (先Streamlit，后续Web应用)
- UI功能需求: **全选**

### 回测细节选择（已确认）
- **Transaction Cost模型**: **C. 高级模型** (Commission + Slippage + Market Impact + Opportunity Cost)
- **Turnover约束**: **B. 硬约束** (单日/单周最大turnover 20%)
- **Walk-Forward**: **B. 可选模式** (支持Walk-Forward和单次回测两种模式)
- **Stress Testing**: **C. 完整Stress Testing** (多种stress场景)

### 因子失效性选择（已确认）
- **因子失效检测**: **D. 多维度检测** (IC + Sharpe + 衰减曲线 + 拥挤度)
- **因子失效处理**: **C. 动态调整** (根据失效程度动态调整权重)

### 理论基础选择（已确认）
- **理论基础**: **D. 多理论融合** (行为金融 + 技术分析 + 市场微观结构 + 机器学习)
- **因子失效原因**: **C+B. 量化监测+详细分析** (量化指标监测 + 深入分析)

### 技术栈清理选择（已确认）
- **Grad-CAM**: **A. 保留** (作为可解释性工具)
- **Transformer/Attention**: **A. 保留** (保留AttentionCAE)
- **无用文件清理**: **A. 激进清理** (删除学术相关文件)

---

## 10阶段实施计划

### 阶段0: 技术栈清理（1周）✅ 立即执行

**任务清单**:
1. 删除 `src/strategies/ablation_study.py`
2. 删除 `src/strategies/baseline_experiments.py`
3. 标记废弃 `src/models/predict_engine.py`
4. 检查并更新所有依赖关系
5. 验证代码库无错误

---

### 阶段1: 胜率计算重构（2周）

**新增任务**:
- 1.7 Transaction Cost高级模型 (`src/strategies/transaction_cost.py`)
- 1.8 Turnover硬约束 (单日/单周最大20%)

---

### 阶段2: 因子有效性分析框架（2周）

**新增任务**:
- 2.8 因子失效多维度检测 (`src/factor_analysis/factor_decay_detector.py`)
- 2.9 因子失效动态权重调整 (`src/strategies/dynamic_weighting.py`)

---

### 阶段6: 分层回测系统（2周）

**新增任务**:
- 6.6 Walk-Forward可选模式 (支持开关)
- 6.7 Stress Testing框架 (`src/backtest/stress_tester.py`)
- 6.8 Stress场景定义 (`config/stress_scenarios.yaml`)

---

### 阶段7: 因子研究框架完善（2周）

**新增任务**:
- 7.7 理论基础文档 (`docs/theoretical_foundation.md`)
- 7.8 因子失效原因量化监测 (`src/factor_research/failure_monitor.py`)
- 7.9 因子失效原因详细分析 (`docs/factor_decay_analysis.md`)

---

## 理论基础框架（多理论融合）

### 1. 行为金融学
- 代表性启发（Representativeness Heuristic）
- 锚定偏差（Anchoring Bias）
- 羊群效应（Herding Behavior）

### 2. 技术分析理论
- 形态识别（Pattern Recognition）
- 支撑阻力（Support/Resistance）
- 趋势延续（Trend Continuation）

### 3. 市场微观结构理论
- 订单流分析（Order Flow）
- 流动性影响（Liquidity Impact）
- 价格发现机制（Price Discovery）

### 4. 机器学习理论
- 无监督学习（CAE）
- 相似度匹配（FAISS）
- 迁移学习（Transfer Learning）

---

## Transaction Cost高级模型设计

```python
class AdvancedTransactionCost:
    """
    高级Transaction Cost模型
    Cost = Commission + Slippage + Market Impact + Opportunity Cost
    """
    def __init__(self):
        self.commission_rate = 0.001  # 0.1%
        self.slippage_rate = 0.001    # 0.1%
        self.market_impact_coef = 0.0001  # 市场冲击系数
        self.opportunity_cost_rate = 0.0005  # 机会成本
    
    def calculate_cost(self, trade_size, price, volume, volatility):
        commission = trade_size * self.commission_rate
        slippage = trade_size * self.slippage_rate
        market_impact = self._calculate_market_impact(trade_size, volume, volatility)
        opportunity_cost = self._calculate_opportunity_cost(trade_size, volatility)
        return commission + slippage + market_impact + opportunity_cost
```

---

## Stress Testing场景定义

```yaml
stress_scenarios:
  financial_crisis_2008:
    start: '2008-09-15'
    end: '2009-03-09'
    description: '2008金融危机'
    market_drop: -50%
  
  covid_crash_2020:
    start: '2020-02-20'
    end: '2020-03-23'
    description: '2020疫情崩盘'
    market_drop: -30%
  
  liquidity_crisis:
    volume_threshold: 0.5  # 成交量降至正常50%
    spread_threshold: 0.02  # 买卖价差扩大至2%
  
  factor_decay:
    ic_threshold: 0.02  # IC持续低于0.02
    duration: 60  # 持续60个交易日
```

---

## 立即执行清单

1. ✅ 阶段0: 技术栈清理（1周）
2. ⏳ 阶段1: 胜率计算重构（2周）
3. ⏳ 阶段2: 因子有效性分析框架（2周）
4. ⏳ 阶段3-9: 按计划执行

**总时间**: 约17周（不含Web迁移）或20周（含Web迁移）
