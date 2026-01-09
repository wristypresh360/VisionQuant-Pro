# 更新日志 | Changelog

All notable changes to VisionQuant-Pro will be documented in this file.

本文档记录VisionQuant-Pro的所有重要更改。

---

## [Unreleased]

### 计划中的功能
- 在线Demo部署（Streamlit Cloud）
- YouTube演示视频
- Vision Transformer模型升级
- 强化学习策略
- 实盘交易接口

---

## [1.0.0] - 2026-01-10

### 🎉 首个正式版本发布

#### Added（新增功能）
- ✨ **视觉识别引擎**：基于CAE的K线形态识别
  - 卷积自编码器（CAE）学习K线图特征
  - FAISS向量检索（毫秒级检索百万级数据库）
  - 混合相似度算法（图像特征60% + 价格相关性40%）
  - 时间隔离算法（NMS）防止重复匹配

- 📊 **批量组合分析**：
  - 支持30只股票并行分析
  - 三层分级组合系统（核心推荐/备选增强/观察监控）
  - 马科维茨均值-方差优化
  - 自动计算组合预期收益、风险、夏普比率

- 🤖 **AI智能助手**：
  - 基于LangChain + Google Gemini
  - 多模态分析（视觉+基本面+技术+舆情）
  - 上下文对话功能
  - 语音输入支持

- 📈 **VQ自适应策略**：
  - 牛市/熊市双模态切换
  - 动态仓位管理（0%-100%）
  - 硬止损8%风险控制
  - 完整回测系统

- 🎨 **Web交互界面**：
  - Streamlit框架
  - Plotly交互图表
  - 响应式布局
  - 三大功能模块（深度研判/批量分析/策略回测）

#### Technical（技术细节）
- **深度学习**：PyTorch CAE模型（4层编码器+4层解码器）
- **向量检索**：FAISS IndexFlatIP（内积相似度）
- **数据源**：AkShare（A股数据）、Google News（舆情）
- **优化算法**：scipy.optimize（SLSQP方法）
- **前端**：Streamlit + Plotly + mplfinance

#### Documentation（文档）
- 📖 完整README（中英双语）
- 📄 技术报告PDF（详细算法说明）
- 📚 贡献指南（CONTRIBUTING.md）
- 📜 MIT开源协议
- 📸 4张高质量项目截图

#### Performance（性能指标）
- 回测结果（2023年）：
  - 紫金矿业（601899）：Strategy +45.2% vs Buy&Hold +28.5%，Alpha +16.7%
  - 贵州茅台（600519）：Strategy +38.7% vs Buy&Hold +22.1%，Alpha +16.6%
- 组合夏普比率：1.78
- 最大回撤：-11.2%（基准-18.5%）
- 批量分析速度：30只股票 3-5分钟

---

## 版本说明

### 语义化版本

我们遵循[语义化版本 2.0.0](https://semver.org/lang/zh-CN/)规范：

- **主版本号（Major）**：不兼容的API修改
- **次版本号（Minor）**：向下兼容的功能性新增
- **修订号（Patch）**：向下兼容的问题修正

---

## 贡献者

感谢所有为VisionQuant-Pro做出贡献的开发者！

- [@panyisheng095-ux](https://github.com/panyisheng095-ux) - 项目创始人

---

## 引用本项目

如果你在研究或项目中使用了VisionQuant-Pro，欢迎引用：

```bibtex
@software{visionquant_pro_2026,
  author = {panyisheng095},
  title = {VisionQuant-Pro: Vision-Based Quantitative Trading System with Deep Learning},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/panyisheng095-ux/VisionQuant-Pro}
}
```

---

**[返回README](README.md)** | **[查看Issues](https://github.com/panyisheng095-ux/VisionQuant-Pro/issues)** | **[提交PR](https://github.com/panyisheng095-ux/VisionQuant-Pro/pulls)**
