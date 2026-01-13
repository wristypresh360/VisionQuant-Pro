"""
AttentionCAE 训练脚本

训练带Self-Attention的卷积自编码器
使用40万张K线图进行无监督学习

Author: Yisheng Pan
Date: 2026-01
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 直接导入，避免 __init__.py 的导入问题
import sys
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'models'))
from attention_cae import AttentionCAE


class KLineImageDataset(Dataset):
    """K线图数据集"""
    
    def __init__(self, image_dir, transform=None, max_samples=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # 获取所有PNG文件
        print(f"📂 扫描图片目录: {image_dir}")
        self.image_files = []
        
        for f in os.listdir(image_dir):
            if f.endswith('.png'):
                self.image_files.append(os.path.join(image_dir, f))
        
        # 限制样本数量（可选）
        if max_samples and len(self.image_files) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(self.image_files), max_samples, replace=False)
            self.image_files = [self.image_files[i] for i in indices]
        
        print(f"✅ 找到 {len(self.image_files)} 张图片")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, 0  # 返回图像和假标签（无监督学习不需要标签）
        except Exception as e:
            # 如果图片损坏，返回随机噪声
            print(f"⚠️ 无法加载图片 {img_path}: {e}")
            return torch.randn(3, 224, 224), 0


def train_attention_cae(
    image_dir: str,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    max_samples: int = None,
    use_attention: bool = True
):
    """
    训练 AttentionCAE 模型
    
    Args:
        image_dir: K线图目录
        output_dir: 模型保存目录
        epochs: 训练轮数
        batch_size: 批大小
        learning_rate: 学习率
        max_samples: 最大样本数（None表示全部）
        use_attention: 是否使用Attention模块
    """
    
    # 设备选择
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("🚀 使用 CUDA GPU 加速")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🚀 使用 Apple MPS GPU 加速")
    else:
        device = torch.device('cpu')
        print("⚠️ 使用 CPU（较慢）")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # 创建数据集
    dataset = KLineImageDataset(image_dir, transform, max_samples)
    
    # 划分训练集和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"📊 训练集: {train_size} 张, 验证集: {val_size} 张")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 创建模型
    model_name = "AttentionCAE" if use_attention else "CAE"
    print(f"\n🧠 创建模型: {model_name}")
    model = AttentionCAE(latent_dim=1024, num_attention_heads=8, use_attention=use_attention)
    model = model.to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📐 模型参数量: {total_params:,}")
    
    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 训练循环
    print(f"\n🏃 开始训练 ({epochs} epochs)")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)
            
            # 前向传播
            recon, latent = model(images)
            loss = criterion(recon, images)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                images = images.to(device)
                recon, _ = model(images)
                loss = criterion(recon, images)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 打印结果
        print(f"\n📈 Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {current_lr:.2e}")
        
        # 保存检查点
        checkpoint_path = os.path.join(output_dir, f"attention_cae_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        print(f"💾 保存检查点: {checkpoint_path}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(output_dir, "attention_cae_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"🏆 保存最佳模型: {best_path} (Val Loss: {val_loss:.6f})")
        
        print("-" * 60)
    
    print("\n✅ 训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"模型保存位置: {output_dir}")
    
    return model


if __name__ == "__main__":
    # 配置
    IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "images")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "models")
    
    # 开始训练
    print("=" * 60)
    print("🎯 AttentionCAE 训练脚本")
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    model = train_attention_cae(
        image_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        epochs=5,              # 5轮训练
        batch_size=32,         # 批大小
        learning_rate=1e-3,    # 学习率
        max_samples=None,      # 使用全部数据
        use_attention=True     # 使用Attention
    )
    
    print(f"\n📅 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
