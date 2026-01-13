# Git 提交指南 - 当前待提交文件整理

## 📋 待提交文件清单（18个文件）

### ✅ 应该提交的文件（17个）

#### 1. 核心功能更新（6个）
- ✅ `src/models/predict_engine.py` - **修复 PredictEngine 导入错误**（重要）
- ✅ `src/models/__init__.py` - 更新导入逻辑
- ✅ `src/models/vision_engine.py` - AttentionCAE 支持
- ✅ `src/strategies/__init__.py` - 更新策略导入
- ✅ `src/strategies/fundamental.py` - 财务数据获取优化
- ✅ `web/app.py` - Web界面更新

#### 2. 新功能脚本（3个）
- ✅ `scripts/train_attention_cae.py` - AttentionCAE 训练脚本
- ✅ `scripts/rebuild_index_attention.py` - 索引重建脚本
- ✅ `src/strategies/ablation_study.py` - 消融实验框架

#### 3. 文档文件（7个）
- ✅ `docs/AttentionCAE切换指南.md` - 模型切换指南
- ✅ `docs/完整实验任务清单.md` - 实验任务清单
- ✅ `docs/当前状态与下一步选项.md` - 状态与选项
- ✅ `docs/查看索引重建进度.md` - 进度查看指南
- ✅ `docs/消融实验方案.md` - 消融实验方案
- ✅ `docs/索引与模型关系说明.md` - 索引说明
- ✅ `docs/论文增强完成总结.md` - 论文总结

#### 4. 项目文档（1个）
- ✅ `README.md` - 项目README更新

### ⚠️ 需要处理的文件（1个）

#### 不应该提交的文件
- ❌ `paper/visionquant_arxiv.tex` - **已在 .gitignore 中，应从 Git 跟踪中移除**

---

## 🔧 处理步骤

### 步骤 1: 从 Git 中移除 paper/visionquant_arxiv.tex

```bash
cd /Users/bytedance/PycharmProjects/pythonProject/VisionQuant-Pro
git rm --cached paper/visionquant_arxiv.tex
```

**原因**：`paper/` 目录已在 `.gitignore` 中，但该文件之前已被 Git 跟踪，需要手动移除。

---

### 步骤 2: 提交文件（建议分2-3次提交）

#### 提交 1: 核心功能修复（重要，优先提交）
```bash
git add src/models/predict_engine.py
git add src/models/__init__.py
git add src/models/vision_engine.py
git add src/strategies/__init__.py
git add src/strategies/fundamental.py
git add web/app.py

git commit -m "fix: 修复 PredictEngine 导入错误并支持 AttentionCAE

- 修复 PredictEngine 导入错误（添加别名兼容）
- 更新 VisionEngine 支持 AttentionCAE 自动检测
- 优化财务数据获取稳定性
- 更新 Web 界面以支持新模型"
```

#### 提交 2: 新功能脚本
```bash
git add scripts/train_attention_cae.py
git add scripts/rebuild_index_attention.py
git add src/strategies/ablation_study.py

git commit -m "feat: 添加 AttentionCAE 训练和索引重建脚本

- 添加 AttentionCAE 训练脚本（支持 MPS/CUDA）
- 添加索引重建脚本（使用 AttentionCAE 编码）
- 添加消融实验框架（9种配置）"
```

#### 提交 3: 文档更新
```bash
git add docs/AttentionCAE切换指南.md
git add docs/完整实验任务清单.md
git add docs/当前状态与下一步选项.md
git add docs/查看索引重建进度.md
git add docs/消融实验方案.md
git add docs/索引与模型关系说明.md
git add docs/论文增强完成总结.md
git add README.md

git commit -m "docs: 添加 AttentionCAE 相关文档和实验指南

- 添加模型切换指南
- 添加完整实验任务清单
- 添加当前状态与下一步选项
- 添加索引重建进度查看指南
- 添加消融实验方案
- 更新 README"
```

---

## 📝 推荐的 Commit Message 格式

### 格式规范
```
<type>: <subject>

<body>

<footer>
```

### Type 类型
- `fix`: 修复bug
- `feat`: 新功能
- `docs`: 文档更新
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建/工具相关

### 示例
```
fix: 修复 PredictEngine 导入错误

- 在 predict_engine.py 中添加 PredictEngine 别名
- 确保从模块直接导入和从包导入都能正常工作
- 解决用户反馈的导入错误问题

Closes #123
```

---

## ⚠️ 注意事项

1. **paper/visionquant_arxiv.tex 不应提交**
   - 已在 `.gitignore` 中
   - 是私人论文资料，不应公开

2. **提交前检查**
   - 确保没有敏感信息（API密钥、个人数据等）
   - 确保代码可以正常运行
   - 确保文档格式正确

3. **提交后验证**
   - 在 GitHub 上检查文件是否正确提交
   - 确认 `paper/visionquant_arxiv.tex` 不在仓库中

---

## 🚀 快速执行命令

### 一键处理（推荐）
```bash
cd /Users/bytedance/PycharmProjects/pythonProject/VisionQuant-Pro

# 1. 移除 paper/visionquant_arxiv.tex
git rm --cached paper/visionquant_arxiv.tex

# 2. 提交所有其他文件（一次性提交）
git add .
git commit -m "feat: 添加 AttentionCAE 支持并修复导入错误

- 修复 PredictEngine 导入错误（添加别名兼容）
- 添加 AttentionCAE 训练和索引重建脚本
- 更新 VisionEngine 支持新模型自动检测
- 添加完整的实验文档和指南
- 优化财务数据获取稳定性"
```

### 分步提交（更规范）
按照上面的"步骤 2"分别提交。
