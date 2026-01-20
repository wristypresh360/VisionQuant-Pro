# GitHub 推送说明

## 当前状态

✅ **所有代码已提交到本地仓库**
- 提交1: `性能优化：添加AI评估步长和快速模式，优化图像路径解析，增强数据加载缓存`
- 提交2: `性能优化：延迟加载FAISS索引，避免启动时20-30分钟等待；优化CSV读取速度`

本地分支领先远程 `origin/main` 2个提交。

## 推送到GitHub

由于需要GitHub认证，请选择以下方式之一：

### 方式1：使用Personal Access Token（推荐）

1. 在GitHub上创建Personal Access Token：
   - 访问：https://github.com/settings/tokens
   - 点击 "Generate new token (classic)"
   - 选择权限：至少勾选 `repo`
   - 生成并复制token

2. 推送代码：
```bash
cd /Users/bytedance/PycharmProjects/pythonProject/VisionQuant-Pro
git push origin main
# 用户名：输入你的GitHub用户名
# 密码：输入刚才复制的Personal Access Token（不是GitHub密码）
```

### 方式2：配置SSH密钥

1. 生成SSH密钥（如果还没有）：
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

2. 将公钥添加到GitHub：
```bash
cat ~/.ssh/id_ed25519.pub
# 复制输出，然后到GitHub Settings > SSH and GPG keys > New SSH key
```

3. 切换远程URL并推送：
```bash
cd /Users/bytedance/PycharmProjects/pythonProject/VisionQuant-Pro
git remote set-url origin git@github.com:panyisheng095-ux/VisionQuant-Pro.git
git push origin main
```

### 方式3：使用GitHub CLI

```bash
gh auth login
cd /Users/bytedance/PycharmProjects/pythonProject/VisionQuant-Pro
git push origin main
```

## 性能优化说明

### 问题诊断

**根本原因**：Streamlit启动时，`load_all_engines()`函数会立即调用`v.reload_index()`，这会：
1. 加载100万条记录的FAISS索引文件（可能几GB）
2. 读取100万行的CSV元数据文件
3. 构建图像路径索引（遍历100万条记录）

这些操作导致启动时间从2-3分钟增加到20-30分钟。

### 优化方案

✅ **延迟加载索引**：
- 移除了启动时的`v.reload_index()`调用
- 索引将在第一次调用`search_similar_patterns()`时自动加载
- 这样启动时只加载模型（几秒），索引在真正需要时才加载

✅ **优化CSV读取**：
- 使用`engine='c'`和`low_memory=False`参数加速CSV读取
- 添加了加载时间统计，便于监控性能

### 预期效果

- **启动时间**：从20-30分钟降低到**2-3分钟**（只加载模型）
- **首次搜索**：第一次使用视觉搜索时会加载索引（约1-2分钟），之后会缓存
- **后续操作**：与之前相同，无性能损失

## 验证优化效果

启动Streamlit后，你应该看到：
1. 启动速度明显加快（2-3分钟而不是20-30分钟）
2. 首次使用"Top10对比"或"回测"功能时，会显示"加载索引"的提示
3. 索引加载后会被缓存，后续操作不会重复加载

## 注意事项

- 索引文件（`data/indices/cae_faiss_attention.bin`）和元数据文件（`data/indices/meta_data_attention.csv`）不会被推送到GitHub（已在`.gitignore`中排除）
- 这些大文件需要单独管理或使用Git LFS
