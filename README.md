<div align="center">

**AI驱动的K线形态智能投研系统 | AI-Powered K-Line Pattern Research System**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/panyisheng095-ux/VisionQuant-Pro?style=social)](https://github.com/panyisheng095-ux/VisionQuant-Pro)

*K线视觉学习 | Top10历史形态对比 | 多因子评分 | 因子有效性分析 | 回测与组合优化 | 工业级性能与鲁棒性*

</div>

---

**语言**: [中文](#readme-zh) | [English](#readme-en)

---

<a id="readme-zh"></a>
# README（中文完整版）

> 目标：构建“工业级”量化投研系统，强调**性能、鲁棒性、可解释性、可复现性**。该README覆盖项目原理、全流程设计、模块细节与全部关键优化路径。

## 目录（中文）
- [项目定位与目标](#zh-overview)
- [科学背景与理论基础](#zh-theory)
- [设计哲学与Magic Moment](#zh-philosophy)
- [系统架构总览](#zh-architecture)
- [数据工程与质量控制](#zh-data)
- [K线图像与多尺度生成](#zh-images)
- [视觉特征学习：AttentionCAE/QuantCAE/SimCLR](#zh-model)
- [索引与元数据：FAISS与路径体系](#zh-index)
- [相似度检索与DTW主导重排](#zh-search)
- [多尺度检索融合](#zh-multiscale)
- [Top10可解释性输出](#zh-top10)
- [K线学习因子与Triple Barrier标签](#zh-factor)
- [因子有效性分析框架](#zh-factor-analysis)
- [回测系统与严格无未来函数](#zh-backtest)
- [组合构建与风险约束](#zh-portfolio)
- [舆情与AI Agent稳定性设计](#zh-agent)
- [性能优化清单（工业级）](#zh-optimizations)
- [鲁棒性与容错策略](#zh-robustness)
- [配置与环境变量](#zh-config)
- [项目结构与模块索引](#zh-structure)
- [快速开始](#zh-quickstart)
- [完整数据流水线与脚本](#zh-pipeline)
- [API服务（FastAPI）](#zh-api)
- [版本历史与路线](#zh-history)
- [风险提示与许可证](#zh-risk)
- [引用与致谢](#zh-reference)

---

<a id="zh-overview"></a>
## 项目定位与目标

**VisionQuant-Pro** 是一个以**K线视觉形态学习**为核心的量化投研系统，核心思想是：
- **用视觉模型理解K线图形态**，把“形态直觉”转化为**可量化因子**。
- 用“**历史相似Top10形态**”替代黑盒预测，强调**可解释与可复盘**。
- 通过**因子有效性分析+回测体系**，实现策略的**科学验证与工业落地**。

核心目标：
- **性能**：百万级样本检索保持可用响应速度，回测/因子分析有可解释的时间复杂度。
- **鲁棒性**：多源数据与网络接口具备退路与容错。
- **准确性**：DTW与价格形态约束减少“形似神不似”。
- **可解释性**：Top10历史形态+统计胜率+因子曲线共同解释结论。

---

<a id="zh-theory"></a>
## 科学背景与理论基础

**多学科融合**支撑系统设计与评价指标：
- **行为金融学**：代表性启发、锚定效应、羊群效应解释“形态复现”。
- **技术分析理论**：形态识别、趋势延续、支撑阻力的可视化载体。
- **市场微观结构**：流动性、订单流、信息扩散解释“形态后续走势”。
- **机器学习理论**：无监督学习（CAE）提取隐变量，形成可检索向量空间。
- **量化因子研究方法**：IC/Sharpe/Regime/Decay作为科学评估框架。

推荐补充阅读：`docs/theoretical_foundation.md`。

---

<a id="zh-philosophy"></a>
## 设计哲学与Magic Moment

**Magic Moment 1：把“预测”变成“历史证据”**
- 传统模型告诉你“会涨”；VisionQuant-Pro告诉你“历史上最像的10个形态中，有7个上涨”。
- 解释链条变清晰，信任成本显著降低。

**Magic Moment 2：把“形态”变成“因子”**
- 视觉形态本质上是一类隐含因子，必须经过IC/Sharpe/Decay框架验证有效性。

**Magic Moment 3：动态权重与因子失效检测**
- 市场结构在变化，因子需要动态调权与失效识别。

设计原则：
- 透明胜过单点准确（可解释是工业落地第一要义）
- 历史胜过主观预测（模型输出必须可回溯）
- 因子胜过模型（用因子研究标准约束模型）
- 动态胜过静态（自适应与失效检测）

---

<a id="zh-architecture"></a>
## 系统架构总览

```
┌──────────────────────────────────────────────────────────────┐
│                        VisionQuant-Pro                        │
│                AI K线形态学习 + 工业级投研系统                   │
└──────────────────────────────────────────────────────────────┘
           │
           ├── 数据层 (Data)
           │   ├─ DataLoader (多数据源/缓存/质量检查)
           │   ├─ K线图像生成 (日/周/月，多尺度)
           │   └─ NewsHarvester / FundamentalMiner
           │
           ├── 模型层 (Model)
           │   ├─ AttentionCAE / QuantCAE
           │   ├─ SimCLR对比学习 (可选增强)
           │   └─ 特征向量化 + L2归一化
           │
           ├── 检索层 (Retrieval)
           │   ├─ FAISS索引 (向量召回)
           │   ├─ DTW主导重排 + 价格相关性
           │   ├─ 像素/边缘相似度轻量重排
           │   └─ 多尺度融合 (日/周/月)
           │
           ├── 因子层 (Factor)
           │   ├─ Triple Barrier标签系统
           │   ├─ 混合胜率 (TB 70% + 传统 30%)
           │   └─ 因子有效性分析 (IC/Sharpe/Decay)
           │
           ├── 策略层 (Strategy)
           │   ├─ 回测引擎 (严格无未来函数)
           │   ├─ Walk-Forward验证
           │   └─ 组合优化 (Markowitz/Black-Litterman)
           │
           └── 展示层 (UI/API)
               ├─ Streamlit Web
               └─ FastAPI 接口
```

---

<a id="zh-data"></a>
## 数据工程与质量控制

### 1) 数据源抽象与切换
- `DataLoader` 统一访问多数据源：`AkshareDataSource`、`JQDataAdapter`、`RQDataAdapter`。
- 主数据源不可用时自动回退到 AkShare。

### 2) 数据质量检查（DataQualityChecker）
- 检查列完整性（Open/High/Low/Close/Volume）
- 检查缺失值、OHLC一致性、高低价格合理性、极端波动、成交量异常、时间连续性等
- 输出质量分数与诊断报告

### 3) 多级缓存
- **磁盘缓存**：`data/raw/*.csv`
- **内存LRU缓存**：避免重复I/O（`mem_cache_max`可配置，FastAPI默认128）
- **局部增量更新**：当请求区间超出本地缓存时，仅补齐缺失区间

### 4) 日期与范围控制
- 默认起始日期为 `20100101`，确保历史覆盖（可配置）
- 请求范围更早或更晚时，自动“向前/向后补齐”数据

---

<a id="zh-images"></a>
## K线图像与多尺度生成

### 1) 图像生成
- `scripts/build_kline_image_dataset.py`：批量生成K线图像数据集
- 支持参数：起止日期、stride（步长）、目标图片数量、进度恢复
- 输出目录结构：`data/images/` 或 `data/images_v2/`

### 2) 多尺度图像
- `MultiScaleChartGenerator` 支持日线/周线/月线
- 统一样式输出（红涨绿跌、隐藏坐标轴）
- 多尺度图像用于多尺度检索融合

### 3) 关键参数
- lookback/window：常用于20日或更长窗口（由脚本/配置决定）
- image_size：默认 `224×224`

---

<a id="zh-model"></a>
## 视觉特征学习：AttentionCAE / QuantCAE / SimCLR

### 1) AttentionCAE（核心模型）
- 结构：CNN Encoder + 多头自注意力 + 低维潜空间 + Decoder
- 目标：重建损失最小化 + 保留形态信息
- 特征输出：`encode()` 返回 **L2归一化向量**
- 注意力权重可视化（解释形态关注区域）

### 2) QuantCAE（回退模型）
- 当 AttentionCAE 权重不可用时自动回退
- `encode()` 输出高维向量，采用池化降维

### 3) SimCLR对比学习（可选）
- `src/models/simclr_trainer.py` 支持对比学习增强表征

### 4) 训练脚本
- `scripts/train_attention_cae.py`
- `scripts/train_multi_scale.py`

---

<a id="zh-index"></a>
## 索引与元数据：FAISS与路径体系

### 1) 索引文件
- AttentionCAE 索引优先：`data/indices/cae_faiss_attention.bin`
- 备选索引：`data/indices/cae_faiss.bin`

### 2) 元数据文件
- `meta_data_attention.csv` / `meta_data.csv`
- 记录 `symbol, date, path`，用于快速映射图像

### 3) 索引-模型对齐
- 索引模式与模型模式不一致时自动切换

### 4) 高性能元数据加载
- CSV读取使用 `engine='c'` + `low_memory=False`

### 5) 内存路径索引
- `(symbol, date) -> path` 的内存哈希表
- 避免递归 `glob` 导致的巨量I/O

---

<a id="zh-search"></a>
## 相似度检索与DTW主导重排

### 1) 检索流水线
1. 图像 → 向量（L2归一化）
2. FAISS粗筛候选（`search_k`）
3. 对候选进行 DTW / 相关性 / 形态特征重排
4. 最终输出 Top-K

### 2) DTW（Dynamic Time Warping）
- 使用 Sakoe-Chiba 带约束加速（窗口=5）
- 时间复杂度从 `O(n²)` 降到 `O(n·window)`

### 3) 形态特征向量（8维）
- 方向、涨跌幅、波动率、最高/最低点位置、头/中/尾三段趋势

### 4) 综合评分（核心逻辑）
- 若有价格序列：
```
combined_score = 0.50*dtw_sim + 0.30*corr + 0.15*feat_sim + 0.05*visual_sim
```
- 若无价格序列：回退到纯视觉相似度

### 5) 趋势约束 + 时间隔离
- 查询趋势与候选趋势必须方向一致
- 同一股票相邻日期隔离，减少“连片”偏差
- `max_date` 控制严格无未来函数

### 6) 快速模式（fast_mode）
- 降低候选数量与价格计算开销
- 因子分析/回测场景下加速

---

<a id="zh-multiscale"></a>
## 多尺度检索融合

- 日/周/月分别检索后加权融合（默认权重：0.6/0.3/0.1）
- 通过 `(symbol, date)` 融合多尺度结果
- 保留元数据路径，避免重复查找
- 支持像素/边缘重排提升“肉眼相似”程度

---

<a id="zh-top10"></a>
## Top10可解释性输出

- `src/utils/visualizer.py` 绘制 “1张查询 + 10张相似” 对比图
- 支持路径优先、目录兜底、glob 兜底
- 输出信息包含：相似度、相关性、日期、股票代码

---

<a id="zh-factor"></a>
## K线学习因子与Triple Barrier标签

### 1) Triple Barrier 标签
- 上边界：+5%
- 下边界：-3%
- 最大持有期：20天
- 标签定义：1(止盈)、0(超时)、-1(止损)

### 2) 混合胜率
```
Hybrid Win Rate = 0.7 * TB_WinRate + 0.3 * Traditional_WinRate
```

### 3) 时间衰减与收益分布
- 支持收益分布统计（均值/分位数/CVaR/偏度/峰度）
- 可结合市场Regime进行更稳健解释

---

<a id="zh-factor-analysis"></a>
## 因子有效性分析框架

核心输出：
- Rolling IC / IC衰减
- Sharpe 曲线
- Regime 识别（牛/熊/震荡）
- 因子失效检测（CUSUM/拐点）
- 多持有期IC矩阵

**工业级优化：**
- 600样本保持不变，但使用 `ThreadPoolExecutor` 并行计算
- 自适应步长采样（在保持样本量的前提下降低计算负担）
- 快速模式检索 + 像素重排关闭 + 限制价格相关性计算
- 失败点自动回退到“自匹配窗口”
- 进度条与诊断指标（成功/失败计数）

---

<a id="zh-backtest"></a>
## 回测系统与严格无未来函数

### 1) 回测模式
- 简单回测
- Walk-Forward验证
- Stress Testing

### 2) 严格无未来函数
- `max_date` 控制匹配只使用历史数据
- AI胜率严格按当期计算

### 3) 工业级速度优化
- AI胜率批量并行预计算
- `ai_stride` 控制AI计算频率
- `ai_fast_mode` 降低检索开销

### 4) 交易成本与A股约束
- 高级交易成本模型：手续费+滑点+市场冲击+机会成本
- 涨跌停、停牌、T+1约束
- 多基线对比与统计显著性检验

---

<a id="zh-portfolio"></a>
## 组合构建与风险约束

- Black-Litterman + Markowitz优化
- 支持CVaR与最大回撤约束
- 核心/增强双层组合结构
- 最小/最大仓位约束与持仓数量限制

---

<a id="zh-agent"></a>
## 舆情与AI Agent稳定性设计

### 1) NewsHarvester（舆情）
- 东方财富JSONP稳健解析
- Google News RSS + Yahoo Finance 兜底
- 请求超时缩短 + 重试 + 缓存（TTL=600s）

### 2) QuantAgent（AI Agent）
- 多模型候选自动回退（gemini-2.5-pro → 2.0-flash → 1.5-pro...）
- `transport="rest"` 增强稳定性
- 指数退避重试 + 异常兜底结果

---

<a id="zh-optimizations"></a>
## 性能优化清单（工业级）

**I/O层优化**
- 图像路径建立内存索引（避免递归glob）
- 元数据CSV加载使用C引擎
- DataLoader内存LRU缓存减少磁盘读取

**计算层优化**
- DTW带窗口约束降复杂度
- 检索过程中早停（高质量候选足够则终止）
- fast_mode降低复杂度（关闭价格相关、减少search_k）
- 像素/边缘重排缓存（最多500条）

**并行化优化**
- 因子分析使用线程池并行
- 严格无未来回测预计算AI胜率

**启动与运行优化**
- 视觉索引延迟加载（第一次检索时加载）
- FastAPI/Streamlit单例引擎复用

---

<a id="zh-robustness"></a>
## 鲁棒性与容错策略

- 数据源失败自动回退（AkShare兜底）
- 图像路径缺失时就近日期回退
- 新闻获取失败自动切源 + 缓存兜底
- LLM连接失败返回降级结果
- 回测/因子分析失败时给出诊断信息

---

<a id="zh-config"></a>
## 配置与环境变量

配置文件：`config/config.yaml`

关键字段：
- `data.raw_dir / images_dir / indices_dir`
- `model.cae.latent_dim`
- `strategy.scoring`（多因子评分权重）
- `web.port`
- `agent.llm.model`

环境变量：
- `.env` 中配置 `GOOGLE_API_KEY` 或 `GEMINI_API_KEY`

---

<a id="zh-structure"></a>
## 项目结构与模块索引

```
VisionQuant-Pro/
├─ config/                     # 配置文件
├─ data/                       # 数据目录（raw/images/indices/...）
├─ docs/                       # 文档与截图
├─ scripts/                    # 训练/索引/标签构建脚本
├─ src/
│  ├─ models/                  # AttentionCAE/QuantCAE/SimCLR
│  ├─ data/                    # DataLoader/数据源/质量检查
│  ├─ strategies/              # 因子/组合/回测策略
│  ├─ factor_analysis/         # IC/Regime/Decay等
│  ├─ utils/                   # 可视化、工具函数
│  └─ agent/                   # AI Agent
├─ web/                        # Streamlit UI + FastAPI
├─ run.py                      # Web启动脚本
└─ README.md                   # 本文档
```

---

<a id="zh-quickstart"></a>
## 快速开始

### 1) 安装依赖
```bash
git clone https://github.com/panyisheng095-ux/VisionQuant-Pro.git
cd VisionQuant-Pro
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2) 运行Web界面
```bash
python run.py
# 或
PYTHONPATH=. streamlit run web/app.py --server.port 8501
```
访问：`http://localhost:8501`

---

<a id="zh-pipeline"></a>
## 完整数据流水线与脚本

### 1) 数据准备
```bash
python scripts/prepare_data.py
```

### 2) 构建K线图像数据集（百万级）
```bash
python scripts/build_kline_image_dataset.py --start-date 20100101 --end-date 20251231 --stride 8 --target-images 1000000
```

### 3) 训练AttentionCAE
```bash
python scripts/train_attention_cae.py
```

### 4) 重建FAISS索引
```bash
python scripts/rebuild_index_attention.py
```

### 5) 计算Triple Barrier标签
```bash
python scripts/batch_triple_barrier.py
```

### 6) 重新计算历史胜率
```bash
python scripts/recalculate_win_rates.py
```

---

<a id="zh-api"></a>
## API服务（FastAPI）

```bash
uvicorn web.api.main:app --host 0.0.0.0 --port 8000 --reload
```

- API 内部复用单例引擎（减少加载开销）
- 提供Top10检索、因子分析、组合优化、新闻与AI接口

---

<a id="zh-history"></a>
## 版本历史与路线

- v1.0：K线视觉检索 + Top10对比 + 评分系统
- v1.5：修复交互与统计、组合优化增强
- v2.0：因子分析框架、Triple Barrier标签体系
- v3.x：工业级性能与稳定性强化（延迟加载/并行计算/多源容错）

---

<a id="zh-risk"></a>
## 风险提示与许可证

- **本项目仅供研究与学习，不构成任何投资建议**
- 历史表现不代表未来收益
- 量化交易存在显著风险，请自行评估

许可证：MIT（详见 `LICENSE`）

---

<a id="zh-reference"></a>
## 引用与致谢

```bibtex
@software{visionquant-pro,
  title = {VisionQuant-Pro: AI-Powered K-Line Pattern Research System},
  author = {Pan, Yisheng},
  year = {2026},
  url = {https://github.com/panyisheng095-ux/VisionQuant-Pro}
}
```

致谢：PyTorch / FAISS / Streamlit / FastAPI / AkShare / Google News

---

<a id="readme-en"></a>
# README (English, Full)

> Goal: build an **industrial-grade** quant research system with emphasis on **performance, robustness, interpretability, and reproducibility**. This README documents principles, pipeline, modules, and key optimizations in detail.

## Table of Contents (English)
- [Positioning & Goals](#en-overview)
- [Scientific Foundations](#en-theory)
- [Design Philosophy & Magic Moments](#en-philosophy)
- [System Architecture](#en-architecture)
- [Data Engineering & Quality Control](#en-data)
- [K-Line Images & Multi-Scale Generation](#en-images)
- [Visual Representation Learning](#en-model)
- [Indexing & Metadata](#en-index)
- [Retrieval & DTW-Driven Re-Ranking](#en-search)
- [Multi-Scale Fusion](#en-multiscale)
- [Top10 Explainability](#en-top10)
- [K-Line Factor & Triple Barrier](#en-factor)
- [Factor Effectiveness Analysis](#en-factor-analysis)
- [Backtesting & Strict No-Future](#en-backtest)
- [Portfolio Construction](#en-portfolio)
- [News & AI Agent Reliability](#en-agent)
- [Performance Optimization Checklist](#en-optimizations)
- [Robustness & Fallbacks](#en-robustness)
- [Configuration & Env Vars](#en-config)
- [Project Structure](#en-structure)
- [Quick Start](#en-quickstart)
- [Full Pipeline Scripts](#en-pipeline)
- [API Service](#en-api)
- [History & Roadmap](#en-history)
- [Risk & License](#en-risk)

---

<a id="en-overview"></a>
## Positioning & Goals

**VisionQuant-Pro** is an AI-powered quant research system centered on **visual K-line pattern learning**. It transforms visual patterns into a **quantitative factor** and provides a transparent Top10 historical match view instead of opaque prediction outputs.

Primary goals:
- **Performance**: scalable retrieval and analysis under large index sizes.
- **Robustness**: multi-source data with reliable fallbacks and retry logic.
- **Accuracy**: DTW and price-shape constraints to reduce false similarity.
- **Explainability**: Top10 historical evidence + statistics + factor curves.

---

<a id="en-theory"></a>
## Scientific Foundations

Multi-disciplinary support:
- **Behavioral finance**: representativeness, anchoring, herding.
- **Technical analysis**: pattern recognition, support/resistance, trend continuation.
- **Market microstructure**: liquidity, order flow, information diffusion.
- **Machine learning**: unsupervised representation learning (CAE).
- **Factor research methodology**: IC/Sharpe/Regime/Decay.

---

<a id="en-philosophy"></a>
## Design Philosophy & Magic Moments

Magic Moment 1: Replace “prediction” with **historical evidence** (Top10 most similar patterns + outcomes).

Magic Moment 2: Treat visual patterns as a **factor**, evaluated by IC/Sharpe/Decay.

Magic Moment 3: **Dynamic** weighting and factor invalidation detection.

Principles:
- Transparency > black-box accuracy
- Evidence from history > subjective prediction
- Factors > one-off models
- Dynamic > static rules

---

<a id="en-architecture"></a>
## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        VisionQuant-Pro                        │
│            AI K-Line Pattern Learning + Research System        │
└──────────────────────────────────────────────────────────────┘
  Data → Images → CAE → FAISS → DTW/Price Re-rank → Top10
      → K-Line Factor → IC/Sharpe/Decay → Backtest → Portfolio
```

---

<a id="en-data"></a>
## Data Engineering & Quality Control

### 1) Multi-source data abstraction
- Unified access via `DataLoader`: AkShare / JQData / RQData.
- Automatic fallback to AkShare when upstream is unavailable.

### 2) Data quality checks
- Column completeness (Open/High/Low/Close/Volume)
- Missing values, OHLC consistency, extreme moves, volume anomalies
- Returns a quality score and diagnostics

### 3) Multi-level caching
- **Disk cache**: `data/raw/*.csv`
- **Memory LRU**: reduces repeated I/O (`mem_cache_max` configurable)
- **Range fill**: fetch earlier/later ranges if cache is incomplete

### 4) Date handling
- Default start date `20100101`
- Auto range correction when end < start

---

<a id="en-images"></a>
## K-Line Images & Multi-Scale Generation

### 1) Image dataset building
- `scripts/build_kline_image_dataset.py` for large-scale generation
- Supports `start-date`, `end-date`, `stride`, `target-images`
- Output: `data/images/` or `data/images_v2/`

### 2) Multi-scale charts
- `MultiScaleChartGenerator` generates daily/weekly/monthly charts
- Unified style, no axes, compact size for embedding and retrieval

### 3) Key parameters
- Window length commonly 20 days for retrieval, configurable for dataset building
- Image size default `224×224`

---

<a id="en-model"></a>
## Visual Representation Learning

### 1) AttentionCAE (primary model)
- CNN encoder + Multi-Head Self-Attention + latent vector + decoder
- Encoded feature is **L2-normalized** for FAISS similarity
- Attention weights can be visualized for interpretability

### 2) QuantCAE (fallback)
- Used when attention weights or model are unavailable
- High-dimensional features are pooled to stable embeddings

### 3) SimCLR (optional enhancement)
- Contrastive learning trainer available in `src/models/simclr_trainer.py`

---

<a id="en-index"></a>
## Indexing & Metadata

### 1) FAISS index files
- Attention index preferred: `data/indices/cae_faiss_attention.bin`
- Fallback index: `data/indices/cae_faiss.bin`

### 2) Metadata
- `meta_data_attention.csv` / `meta_data.csv`
- Stores `symbol`, `date`, `path` for fast image resolution

### 3) Index-model alignment
- Model mode automatically aligned with index mode

### 4) Fast metadata loading
- CSV loading with `engine='c'` and `low_memory=False`

### 5) In-memory path map
- `(symbol, date) -> path` map to avoid expensive glob scans

---

<a id="en-search"></a>
## Retrieval & DTW-Driven Re-Ranking

### 1) Pipeline
1. Image → embedding (L2 normalized)
2. FAISS coarse recall (`search_k`)
3. DTW + correlation + shape features for re-ranking
4. Final Top-K

### 2) DTW with constraint
- Sakoe-Chiba band (window=5) to reduce complexity

### 3) Shape feature vector (8 dims)
- Trend direction, return, volatility, high/low positions, head/mid/tail trends

### 4) Core scoring (when price series available)
```
combined_score = 0.50*dtw_sim + 0.30*corr + 0.15*feat_sim + 0.05*visual_sim
```

### 5) Trend constraint & time isolation
- Trend direction must be consistent
- Same-stock candidates require minimum day gap
- `max_date` enforces strict no-future

### 6) fast_mode
- Reduced price checks and smaller search_k for speed-sensitive scenarios

---

<a id="en-multiscale"></a>
## Multi-Scale Fusion

- Daily/weekly/monthly searches are weighted and merged
- `symbol,date` key ensures consistent fusion
- Optional pixel/edge re-rank for visual alignment

---

<a id="en-top10"></a>
## Top10 Explainability

- Generates “1 query + 10 matches” comparison grid
- Shows symbol/date/score for each match
- Robust path resolution with metadata path priority

---

<a id="en-factor"></a>
## K-Line Factor & Triple Barrier

### 1) Triple Barrier labels
- Upper: +5%
- Lower: -3%
- Max holding: 20 days

### 2) Hybrid win rate
```
Hybrid = 0.7*TB_WinRate + 0.3*Traditional_WinRate
```

### 3) Return distribution
- Weighted returns distribution, quantiles, CVaR, skewness, kurtosis

---

<a id="en-factor-analysis"></a>
## Factor Effectiveness Analysis

Outputs:
- Rolling IC, Sharpe, multi-horizon IC
- Regime detection and decay analysis
- Factor invalidation diagnostics

Industrial optimizations:
- 600-sample parallel computation
- fast_mode search + limited price checks
- progress reporting + fallback matching

---

<a id="en-backtest"></a>
## Backtesting & Strict No-Future

Modes:
- Simple backtest
- Walk-Forward validation
- Stress testing

Strict no-future:
- `max_date` filters future matches
- AI win-rate computed per date

Performance:
- Parallel AI win-rate precompute
- `ai_stride` reduces frequency of AI calls
- `ai_fast_mode` reduces retrieval cost

Risk & cost:
- Advanced transaction cost (commission + slippage + market impact + opportunity cost)
- A-share constraints: limit up/down, suspension, T+1

---

<a id="en-portfolio"></a>
## Portfolio Construction

- Markowitz + Black-Litterman optimizer
- CVaR and max drawdown constraints
- Core + Enhanced two-tier allocation

---

<a id="en-agent"></a>
## News & AI Agent Reliability

News:
- Eastmoney JSONP parsing + retry + timeout control
- Google News RSS + Yahoo Finance as fallbacks
- In-memory cache with TTL

AI Agent:
- Multi-model fallback (Gemini 2.5/2.0/1.5)
- REST transport for stability
- Exponential backoff and graceful fallback output

---

<a id="en-optimizations"></a>
## Performance Optimization Checklist

I/O:
- In-memory path map for image resolution
- Faster CSV loading (`engine='c'`)
- DataLoader LRU cache

Compute:
- DTW with constraint window
- Early stop once high-quality candidates are sufficient
- fast_mode to reduce expensive computation
- Pixel/edge cache (lightweight re-rank)

Parallelization:
- Factor analysis thread pool
- AI win-rate precompute

Runtime:
- Lazy-loading FAISS index
- Singleton engines for Streamlit/FastAPI

---

<a id="en-robustness"></a>
## Robustness & Fallbacks

- Data source failover (AkShare fallback)
- Nearest-date image fallback for missing charts
- News cache + multi-source fallback
- LLM fallback result on connection failure

---

<a id="en-config"></a>
## Configuration & Env Vars

Config file: `config/config.yaml`

Key sections:
- `data.*` paths
- `model.cae.latent_dim`
- `strategy.scoring` weights
- `web.port`
- `agent.llm.model`

Environment:
- `.env` with `GOOGLE_API_KEY` or `GEMINI_API_KEY`

---

<a id="en-structure"></a>
## Project Structure

```
VisionQuant-Pro/
├─ config/                     # configs
├─ data/                       # raw/images/indices
├─ docs/                       # docs and screenshots
├─ scripts/                    # training/indexing/labels
├─ src/                        # core modules
├─ web/                        # Streamlit UI + FastAPI
└─ run.py                      # launcher
```

---

<a id="en-quickstart"></a>
## Quick Start

```bash
git clone https://github.com/panyisheng095-ux/VisionQuant-Pro.git
cd VisionQuant-Pro
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run.py
```

---

<a id="en-pipeline"></a>
## Full Pipeline Scripts

```bash
python scripts/build_kline_image_dataset.py --start-date 20100101 --end-date 20251231 --stride 8 --target-images 1000000
python scripts/train_attention_cae.py
python scripts/rebuild_index_attention.py
python scripts/batch_triple_barrier.py
python scripts/recalculate_win_rates.py
```

---

<a id="en-api"></a>
## API Service

```bash
uvicorn web.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

<a id="en-history"></a>
## History & Roadmap

- v1.0: Top10 retrieval + scoring
- v2.0: factor framework + Triple Barrier
- v3.x: industrial-grade optimization & robustness

---

<a id="en-risk"></a>
## Risk & License

- Research only; no investment advice
- Past performance does not guarantee future returns
- Licensed under MIT
