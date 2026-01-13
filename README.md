# VisionQuant-Pro v2.0

<div align="center">

**Vision-Based Quantitative Trading System with Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Dual-Stream Architecture | GAF Encoding | Triple Barrier | Walk-Forward Validation*

</div>

---

## 📊 Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| **v1.5 Web Interface** | ✅ Working | Uses 400K K-line images, fully functional |
| **v1.5 AttentionCAE Model** | ✅ Trained | 5 epochs on 400K images |
| **v1.5 FAISS Index** | ✅ Built | 400K vectors indexed |
| **v2.0 Framework Code** | ✅ Complete | ~4,600 lines, all imports verified |
| **v2.0 GAF Images** | ⏳ Pending | Run `scripts/prepare_data.py` to generate |
| **v2.0 Dual-Stream Model** | ⏳ Pending | Run `scripts/train_dual_stream.py` to train |

> **Note**: v2.0 is currently a **framework implementation**. The architecture and training scripts are complete, but model training has not been executed yet. The existing v1.5 system remains fully functional.

---

## 🇨🇳 版本迭代说明 (Version Evolution in Chinese)

<details>
<summary>点击展开查看中文版本对比</summary>

### v1.0 → v2.0 核心改进

| 维度 | v1.0 问题 | v2.0 解决方案 |
|------|----------|--------------|
| **信息丢失** | K线截图丢失精确数值 | GAF数学编码 + 双流保留原始OHLCV |
| **标签简单** | 简单涨跌二分类 | Triple Barrier三分类（止盈/止损/震荡） |
| **未来函数** | 随机划分数据集 | Walk-Forward滚动验证 |
| **缺乏理论** | "看图说话"式评分 | 有数学定义的GAF/Triple Barrier |
| **不可解释** | 黑盒模型 | Grad-CAM热力图 + 注意力权重可视化 |
| **回测简陋** | 自写简单回测 | Backtrader专业框架 |

### 架构演进图

```
┌─────────────────────────────────────────────────────────────────┐
│                        VERSION EVOLUTION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  v1.0 (2026-01-05)          v1.5 (2026-01-10)                    │
│  ─────────────────          ─────────────────                    │
│  K线截图                     K线截图                              │
│     │                           │                                 │
│     ↓                           ↓                                 │
│  QuantCAE                   AttentionCAE                          │
│  (4层CNN)                   (CAE + 8头注意力)                     │
│     │                           │                                 │
│     ↓                           ↓                                 │
│  FAISS检索                  FAISS检索                             │
│     │                           │                                 │
│     ↓                           ↓                                 │
│  胜率预测                    V+F+Q多因子评分                       │
│                                                                   │
│                          v2.0 (2026-01-13)                        │
│                          ─────────────────                        │
│                          OHLCV原始数据                            │
│                               │                                   │
│                    ┌──────────┴──────────┐                        │
│                    ↓                     ↓                        │
│               GAF图像               标准化序列                     │
│                    │                     │                        │
│                    ↓                     ↓                        │
│               ResNet18              TCN+Attention                  │
│                    │                     │                        │
│                    └──────────┬──────────┘                        │
│                               ↓                                   │
│                      Cross-Modal Attention                        │
│                               │                                   │
│                               ↓                                   │
│                      Triple Barrier预测                           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 新增代码量统计

| 文件 | 功能 | 代码行数 |
|------|------|---------|
| `gaf_encoder.py` | GAF图像编码 | 491 |
| `triple_barrier.py` | Triple Barrier标签 | 549 |
| `walk_forward.py` | Walk-Forward验证 | 638 |
| `temporal_encoder.py` | TCN+Attention时序编码 | 579 |
| `dual_stream_network.py` | 双流融合网络 | 711 |
| `backtrader_strategy.py` | Backtrader策略集成 | 555 |
| `train_dual_stream.py` | 训练脚本 | 523 |
| `grad_cam.py` | Grad-CAM可视化 | 517 |
| **总计** | | **~4,600** |

</details>

---

## What's New in v2.0

- **Dual-Stream Architecture**: Vision Stream (GAF images) + Temporal Stream (TCN+Attention)
- **GAF Encoding**: Gramian Angular Field - mathematically rigorous time-to-image conversion
- **Triple Barrier Method**: Industry-standard labeling (profit-taking, stop-loss, time horizon)
- **Walk-Forward Validation**: Prevent look-ahead bias with rolling window training
- **Backtrader Integration**: Professional backtesting framework
- **Grad-CAM Explainability**: Visualize what the model "sees" in charts

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      VisionQuant-Pro v2.0                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐         ┌─────────────────────┐           │
│  │    Vision Stream    │         │   Temporal Stream   │           │
│  │                     │         │                     │           │
│  │  OHLCV → GAF Image  │         │  OHLCV → Sequence   │           │
│  │       ↓            │         │       ↓            │           │
│  │  ResNet18/ViT      │         │  TCN + Attention   │           │
│  │       ↓            │         │       ↓            │           │
│  │  [B, 512] features │         │  [B, 256] features │           │
│  └──────────┬──────────┘         └──────────┬──────────┘           │
│             │                               │                       │
│             └───────────────┬───────────────┘                       │
│                             ↓                                       │
│             ┌───────────────────────────────┐                       │
│             │   Cross-Modal Attention       │                       │
│             │      [B, 768] fused           │                       │
│             └───────────────┬───────────────┘                       │
│                             │                                       │
│        ┌────────────────────┼────────────────────┐                  │
│        ↓                    ↓                    ↓                  │
│  ┌───────────┐       ┌───────────┐       ┌───────────┐             │
│  │  FAISS    │       │ Triple    │       │   Risk    │             │
│  │  Search   │       │ Barrier   │       │   Eval    │             │
│  └───────────┘       └───────────┘       └───────────┘             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Innovations

### 1. GAF Encoding (Gramian Angular Field)

Unlike simple K-line chart screenshots, GAF provides **mathematically rigorous** time-to-image conversion:

```python
# Mathematical formulation
x_scaled = (x - min) / (max - min) * 2 - 1  # Normalize to [-1, 1]
φ = arccos(x_scaled)                         # Polar angle
G[i,j] = cos(φ_i + φ_j)                      # GASF matrix
```

**3-Channel GAF Image**:
- **R**: GASF (Gramian Angular Summation Field) - captures overall trends
- **G**: GADF (Gramian Angular Difference Field) - captures local changes
- **B**: MTF (Markov Transition Field) - captures state transitions

### 2. Dual-Stream Fusion

**Vision Stream**: Processes GAF images with ResNet18/ViT
- Captures spatial patterns (Double Bottom, Head-and-Shoulders, etc.)
- Pretrained on ImageNet for transfer learning

**Temporal Stream**: Processes raw OHLCV with TCN + Self-Attention
- TCN: Dilated causal convolutions for local patterns
- Self-Attention: Long-range dependencies across time

**Cross-Modal Attention**: Learns complementary information
- Gate mechanism balances vision vs. temporal importance
- Enables interpretation: "Which modality contributed more?"

### 3. Triple Barrier Labeling

Standard in quantitative finance (López de Prado, 2018):

```python
def get_label(price_series, pt=0.05, sl=0.03, max_holding=20):
    """
    pt: profit-taking threshold (5%)
    sl: stop-loss threshold (3%)
    max_holding: maximum holding period (20 days)
    
    Returns:
    - 1: Hit profit-taking first → Bullish
    - -1: Hit stop-loss first → Bearish
    - 0: Hit time horizon first → Neutral
    """
```

### 4. Walk-Forward Validation

Prevents look-ahead bias by simulating real trading:

```
|------ Train (3 years) ------|-- Val (6mo) --|-- Test (6mo) --|
                              |
                              ↓ Roll forward
|------ Train (3 years) ------|-- Val (6mo) --|-- Test (6mo) --|
```

---

## Project Structure

```
VisionQuant-Pro/
├── src/
│   ├── models/
│   │   ├── dual_stream_network.py  # Core: Dual-Stream Architecture
│   │   ├── temporal_encoder.py      # TCN + Self-Attention
│   │   ├── attention_cae.py         # Legacy: AttentionCAE
│   │   └── vision_engine.py         # FAISS search engine
│   ├── data/
│   │   ├── gaf_encoder.py           # GAF image generation
│   │   ├── triple_barrier.py        # Label generation
│   │   └── data_loader.py           # Stock data loader
│   ├── strategies/
│   │   ├── backtrader_strategy.py   # Backtrader integration
│   │   ├── portfolio_optimizer.py   # Markowitz optimization
│   │   └── factor_mining.py         # Multi-factor scoring
│   └── utils/
│       ├── walk_forward.py          # Walk-Forward validation
│       └── grad_cam.py              # Explainability
├── scripts/
│   ├── train_dual_stream.py         # Training script
│   └── prepare_data.py              # Data preparation
├── web/
│   └── app.py                       # Streamlit interface
├── docs/
│   ├── AttentionCAE切换指南.md
│   ├── 常见问题FAQ.md
│   └── 在线部署教程.md
└── requirements.txt
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/panyisheng095-ux/VisionQuant-Pro.git
cd VisionQuant-Pro

python -m venv venv
source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### Data Preparation

```bash
# Generate GAF images and labels
python scripts/prepare_data.py --symbols 600519 000858 601899 --window 60
```

### Training

```bash
# Train dual-stream network with Walk-Forward validation
python scripts/train_dual_stream.py \
    --data_dir data \
    --gaf_dir data/gaf_images \
    --batch_size 32 \
    --num_epochs 50
```

### Web Interface

```bash
python run.py  # or: PYTHONPATH=. streamlit run web/app.py
```

---

## Comparison with Other Approaches

| Aspect | Traditional Quant | Pure CNN | RD-Agent | VisionQuant v2.0 |
|--------|------------------|----------|----------|------------------|
| Input | Numerical | K-line image | Numerical+Text | **GAF+OHLCV** |
| Time Modeling | Hand-crafted | Ignored | Agent reasoning | **TCN+Attention** |
| Image Encoding | None | Screenshot | None | **GAF (math-based)** |
| Explainability | High | Low | Medium | **High (Grad-CAM)** |
| Labeling | Returns | Up/Down | Returns | **Triple Barrier** |
| Validation | Random split | Random split | Rolling | **Walk-Forward** |

---

## Theoretical Foundation

### Behavioral Finance Justification

> "The market is driven by human behavior, and humans are visual creatures."

- **Anchoring Bias**: Traders anchor to visually prominent patterns (support/resistance)
- **Herding Behavior**: Visual breakouts trigger collective action
- **Representativeness Heuristic**: Similar charts → similar future outcomes

Our model formalizes these intuitions:
- GAF preserves the visual structure traders see
- Cross-modal fusion captures both "what it looks like" and "how it moves"
- Historical pattern matching exploits behavioral repetition

### Information Theoretic View

```
I(FutureReturn; GAF+OHLCV) > I(FutureReturn; OHLCV)
```

The visual representation captures geometric and topological features that are difficult to extract from raw numerical sequences.

---

## Performance Notes

### Expected Results
- **Classification Accuracy**: 45-55% (3-class, beating random 33%)
- **Return Prediction MAE**: 2-4%
- **Alpha vs Buy-and-Hold**: Varies by market condition

### Disclaimer
- **This is a research project, NOT investment advice**
- Past performance does not guarantee future results
- Quantitative trading involves significant risk

---

## Roadmap

### v2.1 (Next)
- [ ] Vision Transformer (ViT) backbone option
- [ ] Contrastive learning (SimCLR) pretraining
- [ ] Multi-timeframe fusion (daily + weekly + monthly)

### v2.2 (Future)
- [ ] Reinforcement learning integration
- [ ] Live trading API integration
- [ ] Multi-market support (US, HK)

---

## Citation

```bibtex
@software{visionquant-pro,
  title = {VisionQuant-Pro: Dual-Stream Vision-Based Quantitative Trading},
  author = {Pan, Yisheng},
  year = {2025},
  url = {https://github.com/panyisheng095-ux/VisionQuant-Pro}
}
```

---

## References

- Wang, Z., & Oates, T. (2015). Imaging time-series to improve classification and imputation. IJCAI.
- López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks.
- Bai, S., et al. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.

---

## Version History

### Detailed Changelog

---

### v2.0.0 (2026-01-13) - Major Architecture Overhaul

**This is a complete rewrite focused on academic rigor and industrial applicability.**

#### ⚡ Core Architecture Changes

| Component | v1.0 | v2.0 | Improvement |
|-----------|------|------|-------------|
| **Image Encoding** | K-line screenshot (matplotlib) | **GAF (Gramian Angular Field)** | 数学严谨的时序→图像转换，保留时间依赖性 |
| **Network** | Single-stream CAE | **Dual-Stream (Vision+Temporal)** | 同时利用视觉空间信息和时序动态信息 |
| **Vision Encoder** | Custom 4-layer CNN | **ResNet18 (pretrained)** | ImageNet预训练，更强的特征提取能力 |
| **Temporal Encoder** | None | **TCN + Self-Attention** | 捕捉长距离时序依赖 |
| **Fusion Method** | None | **Cross-Modal Attention** | 可学习的模态融合权重 |

#### 📊 Data & Labels

| Component | v1.0 | v2.0 | Improvement |
|-----------|------|------|-------------|
| **Input Data** | K线截图 (PNG) | **GAF 3通道图像 + 原始OHLCV** | 无信息丢失，精确数值保留 |
| **Label Definition** | 简单涨跌 (+5天收益率>0) | **Triple Barrier Method** | 业界标准，考虑止盈/止损/时间限制 |
| **Label Classes** | 2类 (涨/跌) | **3类 (看涨/震荡/看跌)** | 更符合实际交易决策 |

#### 🔬 Training & Validation

| Component | v1.0 | v2.0 | Improvement |
|-----------|------|------|-------------|
| **Data Split** | 随机 90/10 | **Walk-Forward 滚动验证** | 防止未来函数泄露 |
| **Validation** | 单次验证集 | **滚动窗口多次验证** | 更可靠的泛化能力评估 |
| **Training Loss** | MSE重建损失 | **分类CE + 回归MSE + 对比损失** | 多任务联合优化 |

#### 📈 Backtesting

| Component | v1.0 | v2.0 | Improvement |
|-----------|------|------|-------------|
| **Framework** | 自写简单回测 | **Backtrader 专业框架** | 工业级回测能力 |
| **Metrics** | 简单收益率 | **Sharpe/Calmar/MaxDD/胜率/盈亏比** | 完整绩效评估 |
| **Look-ahead Bias** | 未严格防范 | **严格时间隔离** | 可信的回测结果 |

#### 🎯 Explainability

| Component | v1.0 | v2.0 | Improvement |
|-----------|------|------|-------------|
| **Model Interpretation** | Attention权重热力图 | **Grad-CAM + Attention + 模态权重** | 多层次可解释性 |
| **Visualization** | 单一注意力图 | **GAF热力图 + 时序注意力 + 融合权重** | 完整的决策解释链 |

#### 📁 New Files Added (v2.0)

```
src/data/
├── gaf_encoder.py          # [NEW] GAF图像编码器 (491 lines)
└── triple_barrier.py       # [NEW] Triple Barrier标签 (549 lines)

src/models/
├── temporal_encoder.py     # [NEW] TCN+Attention时序编码器 (579 lines)
└── dual_stream_network.py  # [NEW] 双流融合网络 (711 lines)

src/strategies/
└── backtrader_strategy.py  # [NEW] Backtrader策略 (555 lines)

src/utils/
├── walk_forward.py         # [NEW] Walk-Forward验证 (638 lines)
└── grad_cam.py             # [NEW] Grad-CAM可视化 (517 lines)

scripts/
└── train_dual_stream.py    # [NEW] 双流网络训练脚本 (523 lines)
```

**Total new code: ~4,600 lines**

---

### v1.5.0 (2026-01-10) - Attention Enhancement

#### Changes from v1.0
- **AttentionCAE**: 在CAE末端添加8头自注意力机制
- **Multi-factor Scoring**: V(视觉)+F(财务)+Q(量化)三因子评分
- **Batch Analysis**: 支持30只股票批量分析
- **Portfolio Optimization**: Markowitz均值-方差优化
- **AI Agent**: 集成Google Gemini大模型辅助分析

#### Files Added (v1.5)
```
src/models/attention_cae.py        # 注意力增强CAE
src/strategies/batch_analyzer.py   # 批量分析引擎
src/strategies/portfolio_optimizer.py  # 组合优化器
src/utils/attention_visualizer.py  # 注意力可视化
```

---

### v1.0.0 (2026-01-05) - Initial Release

#### Core Features
- **QuantCAE**: 4层卷积自编码器，学习K线图形态
- **FAISS Search**: 向量相似度搜索，毫秒级检索
- **Streamlit Web**: 交互式Web界面
- **基础回测**: 简单的买入持有对比

#### Architecture (v1.0)
```
K线截图 (matplotlib)
    ↓
QuantCAE (4-layer CNN)
    ↓
FAISS Index (L2 distance)
    ↓
Top-K Similar Patterns
    ↓
Win Rate Prediction
```

#### Limitations Identified
1. ❌ K线截图丢失精确数值信息
2. ❌ 纯CNN无法捕捉长距离依赖
3. ❌ 简单涨跌标签不符合实际交易
4. ❌ 随机数据划分导致未来函数风险
5. ❌ 缺乏严谨的回测框架

---

### Version Comparison Summary

```
v1.0 Architecture:
──────────────────
K线截图 → CAE Encoder → FAISS → Win Rate → Simple Score

v1.5 Architecture:
──────────────────
K线截图 → AttentionCAE → FAISS → Win Rate → Multi-Factor Score
                ↑                              ↑
          + Attention                    + V+F+Q Factors

v2.0 Architecture:
──────────────────
         ┌→ GAF Image → ResNet18 ──────┐
OHLCV ───┤                              ├→ Cross-Modal Attention → Triple Barrier
         └→ Sequence  → TCN+Attention ─┘
```

---

## License

MIT License - see [LICENSE](LICENSE)

---

<div align="center">

**If you find this project useful, please give it a ⭐ Star!**

Made with ❤️ by [panyisheng095-ux](https://github.com/panyisheng095-ux)

</div>
