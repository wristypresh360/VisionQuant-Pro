# VisionQuant-Pro Documentation

This directory contains user-facing documentation for VisionQuant-Pro v2.0.

## Architecture Overview

VisionQuant-Pro v2.0 introduces a **Dual-Stream Architecture**:

```
Input: Stock Data (OHLCV)
         ↓
    ┌────┴────┐
    ↓         ↓
 GAF Image   Raw Sequence
    ↓         ↓
 ResNet18   TCN+Attention
    ↓         ↓
    └────┬────┘
         ↓
 Cross-Modal Fusion
         ↓
 Triple Barrier Prediction
```

## Key Components

### 1. Data Processing
- **GAF Encoder** (`src/data/gaf_encoder.py`): Converts OHLCV to Gramian Angular Field images
- **Triple Barrier** (`src/data/triple_barrier.py`): Industry-standard labeling method

### 2. Models
- **Dual-Stream Network** (`src/models/dual_stream_network.py`): Vision + Temporal fusion
- **Temporal Encoder** (`src/models/temporal_encoder.py`): TCN + Self-Attention

### 3. Validation
- **Walk-Forward** (`src/utils/walk_forward.py`): Prevents look-ahead bias

### 4. Backtesting
- **Backtrader Integration** (`src/strategies/backtrader_strategy.py`): Professional backtesting

### 5. Explainability
- **Grad-CAM** (`src/utils/grad_cam.py`): Visualize model attention

## Available Documents

| Document | Description |
|----------|-------------|
| [FAQ](常见问题FAQ.md) | Frequently asked questions |
| [Deployment Guide](在线部署教程.md) | Online deployment instructions |
| [Index & Model Guide](索引与模型关系说明.md) | Technical explanation of FAISS indices |
| [AttentionCAE Guide](AttentionCAE切换指南.md) | Legacy model switching guide |

## Quick Links

- **Main README**: [../README.md](../README.md)
- **Technical Report**: [papers/VisionQuant-Pro_Technical_Report.pdf](papers/VisionQuant-Pro_Technical_Report.pdf)
- **Screenshots**: [images/](images/)

---

## Version History

### v2.0 (Current)
- Dual-Stream Architecture (Vision + Temporal)
- GAF Encoding
- Triple Barrier Labeling
- Walk-Forward Validation
- Backtrader Integration
- Grad-CAM Explainability

### v1.0 (Legacy)
- AttentionCAE Model
- Simple K-line Image Encoding
- Binary Classification
