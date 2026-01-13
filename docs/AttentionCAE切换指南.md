# AttentionCAE 切换指南

## ✅ 已完成的工作

### 1. 代码修改
- ✅ `src/models/vision_engine.py` 已更新，支持自动检测并使用 AttentionCAE
- ✅ 如果 `attention_cae_best.pth` 存在，自动加载；否则回退到旧的 `QuantCAE`
- ✅ 索引加载逻辑已更新，优先使用新索引

### 2. 索引重建脚本
- ✅ `scripts/rebuild_index_attention.py` 已创建
- 功能：用 AttentionCAE 重新编码所有 40 万张图片并构建新索引

---

## 📋 下一步操作（必须执行）

### 步骤 1: 重建 FAISS 索引

**重要**：旧的索引是用 `QuantCAE` 构建的（50176 维 → 1024 维），新模型 `AttentionCAE` 直接输出 1024 维，**必须重建索引才能正常工作**。

```bash
cd /Users/bytedance/PycharmProjects/pythonProject/VisionQuant-Pro
python3 scripts/rebuild_index_attention.py
```

**预计时间**：
- CPU: 2-3 小时
- MPS (Mac GPU): 1-2 小时
- CUDA: 30-60 分钟

**输出文件**：
- `data/indices/cae_faiss_attention.bin` (新索引)
- `data/indices/meta_data_attention.csv` (新元数据)

---

### 步骤 2: 验证新索引

重建完成后，重启 Streamlit 应用：

```bash
python3 -m streamlit run web/app.py --server.port 8501
```

在网页中测试几只股票，检查：
1. ✅ 对比图是否正常显示 Top 10
2. ✅ 相似度分数是否合理
3. ✅ 检索速度是否正常

---

## 🔄 回退方案

如果新模型效果不好，可以快速回退：

1. **方法 A（推荐）**：删除新模型文件，自动回退
   ```bash
   mv data/models/attention_cae_best.pth data/models/attention_cae_best.pth.backup
   ```
   重启应用后会自动使用 `QuantCAE` + 旧索引

2. **方法 B**：手动修改代码
   在 `src/models/vision_engine.py` 中强制使用 `QuantCAE`：
   ```python
   # 在 __init__ 中直接设置
   use_attention = False  # 强制禁用
   ```

---

## 📊 架构对比

### 旧架构（QuantCAE）
- 模型：`QuantCAE` (3.0M)
- 编码：50176 维 → Pool → 1024 维
- 索引：40 万条记录
- 特点：纯卷积，局部特征强

### 新架构（AttentionCAE）
- 模型：`AttentionCAE` (201M)
- 编码：直接输出 1024 维（已 L2 归一化）
- 索引：40 万条记录（需重建）
- 特点：卷积 + Self-Attention，全局特征强

---

## ⚠️ 注意事项

1. **索引不兼容**：新旧索引不能混用，必须重建
2. **内存占用**：AttentionCAE 模型更大（201M vs 3M），但推理速度相近
3. **精度预期**：理论上 AttentionCAE 应该更好（能捕获长距离依赖），但需要实际测试验证

---

## 🐛 故障排查

### 问题 1: "索引文件不存在"
**原因**：还没运行重建脚本  
**解决**：执行 `python3 scripts/rebuild_index_attention.py`

### 问题 2: "模型加载失败"
**原因**：`attention_cae_best.pth` 不存在或损坏  
**解决**：检查文件是否存在，或重新训练

### 问题 3: "检索结果为空"
**原因**：使用了旧索引（维度不匹配）  
**解决**：确保使用新索引 `cae_faiss_attention.bin`

---

## 📝 更新日志

- **2026-01-13**: 完成代码修改和脚本创建
- 待执行：索引重建（用户操作）
