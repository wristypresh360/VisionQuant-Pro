"""
用 AttentionCAE 重建 FAISS 索引

这个脚本会：
1. 加载训练好的 AttentionCAE 模型
2. 扫描所有 K 线图（40万张）
3. 用新模型重新编码所有图片
4. 构建新的 FAISS 索引

运行时间：约 1-2 小时（取决于 CPU/GPU）
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import faiss
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import glob

# === 路径配置 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.attention_cae import AttentionCAE

# 输入输出路径（默认值，可被命令行覆盖）
DEFAULT_IMG_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "images")
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "models", "attention_cae_best.pth")
DEFAULT_INDEX_FILE = os.path.join(PROJECT_ROOT, "data", "indices", "cae_faiss_attention.bin")
DEFAULT_META_CSV = os.path.join(PROJECT_ROOT, "data", "indices", "meta_data_attention.csv")

parser = argparse.ArgumentParser(description="用 AttentionCAE 重建 FAISS 索引")
parser.add_argument("--img-dir", type=str, default=DEFAULT_IMG_BASE_DIR, help="K线图目录（默认 data/images）")
parser.add_argument("--index-file", type=str, default=DEFAULT_INDEX_FILE, help="输出索引文件路径")
parser.add_argument("--meta-csv", type=str, default=DEFAULT_META_CSV, help="输出元数据CSV路径")
args = parser.parse_args()

IMG_BASE_DIR = args.img_dir
INDEX_FILE = args.index_file
META_CSV = args.meta_csv

# 设备选择
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 使用 Apple MPS GPU 加速")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 使用 CUDA GPU 加速")
else:
    device = torch.device("cpu")
    print("💻 使用 CPU（较慢，建议用 GPU）")

# === 1. 加载模型 ===
print("\n" + "="*60)
print("📦 步骤 1: 加载 AttentionCAE 模型")
print("="*60)

model = AttentionCAE(latent_dim=1024, num_attention_heads=8).to(device)
if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ 模型加载成功: {MODEL_PATH}")
else:
    print(f"❌ 模型文件不存在: {MODEL_PATH}")
    sys.exit(1)

# 预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === 2. 扫描所有图片 ===
print("\n" + "="*60)
print("📂 步骤 2: 扫描 K 线图目录")
print("="*60)

# 查找所有 PNG 文件
all_img_paths = glob.glob(os.path.join(IMG_BASE_DIR, "**", "*.png"), recursive=True)
print(f"✅ 找到 {len(all_img_paths)} 张图片 (目录: {IMG_BASE_DIR})")

if len(all_img_paths) == 0:
    print("❌ 没有找到图片文件！请检查路径:", IMG_BASE_DIR)
    sys.exit(1)

# === 3. 提取特征向量 ===
print("\n" + "="*60)
print("🔍 步骤 3: 用 AttentionCAE 编码所有图片")
print("="*60)
print("⚠️  这可能需要 1-2 小时，请耐心等待...")

features_list = []
meta_list = []
batch_size = 32  # 批处理大小

# 创建输出目录
os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)

with torch.no_grad():
    for i, img_path in enumerate(tqdm(all_img_paths, desc="编码中")):
        try:
            # 加载图片
            img = Image.open(img_path).convert('RGB')
            input_tensor = preprocess(img).unsqueeze(0).to(device)
            
            # 编码（AttentionCAE.encode() 返回 1024 维，已 L2 归一化）
            feature = model.encode(input_tensor)
            feature_np = feature.cpu().numpy().flatten().astype('float32')
            
            features_list.append(feature_np)
            
            # 从路径提取股票代码和日期
            # 路径格式: data/images/600519/600519_20230101.png
            filename = os.path.basename(img_path)
            parts = filename.replace('.png', '').split('_')
            if len(parts) >= 2:
                symbol = parts[0].zfill(6)
                date_str = parts[1]
            else:
                # 备用解析
                symbol = os.path.basename(os.path.dirname(img_path)).zfill(6)
                date_str = filename.replace('.png', '')
            
            meta_list.append({
                'symbol': symbol,
                'date': date_str,
                'path': img_path
            })
            
        except Exception as e:
            print(f"\n⚠️  跳过损坏图片 {img_path}: {e}")
            continue

print(f"\n✅ 编码完成！共处理 {len(features_list)} 张图片")

# === 4. 构建 FAISS 索引 ===
print("\n" + "="*60)
print("🔨 步骤 4: 构建 FAISS 索引")
print("="*60)

features_array = np.array(features_list)
dim = features_array.shape[1]
print(f"特征维度: {dim} (应该是 1024)")

# 创建索引（使用内积，因为特征已 L2 归一化）
index = faiss.IndexFlatIP(dim)

# 归一化（确保是单位向量）
faiss.normalize_L2(features_array)

# 添加向量
print("正在添加向量到索引...")
index.add(features_array)

print(f"✅ 索引构建完成！包含 {index.ntotal} 条记录")

# === 5. 保存索引和元数据 ===
print("\n" + "="*60)
print("💾 步骤 5: 保存索引和元数据")
print("="*60)

# 保存 FAISS 索引
faiss.write_index(index, INDEX_FILE)
print(f"✅ FAISS 索引已保存: {INDEX_FILE}")

# 保存元数据 CSV
meta_df = pd.DataFrame(meta_list)
meta_df.to_csv(META_CSV, index=False)
print(f"✅ 元数据已保存: {META_CSV}")

# === 6. 更新 VisionEngine 配置 ===
print("\n" + "="*60)
print("📝 步骤 6: 更新配置")
print("="*60)
print("⚠️  请手动更新 src/models/vision_engine.py 中的索引路径：")
print(f"   INDEX_FILE = '{INDEX_FILE}'")
print(f"   META_CSV = '{META_CSV}'")
print("\n或者直接替换旧索引文件（备份后）：")
print(f"   mv {INDEX_FILE} {os.path.join(PROJECT_ROOT, 'data', 'indices', 'cae_faiss.bin')}")
print(f"   mv {META_CSV} {os.path.join(PROJECT_ROOT, 'data', 'indices', 'meta_data.csv')}")

print("\n" + "="*60)
print("🎉 索引重建完成！")
print("="*60)
print(f"总记录数: {index.ntotal}")
print(f"特征维度: {dim}")
print(f"索引文件: {INDEX_FILE}")
print(f"元数据文件: {META_CSV}")
