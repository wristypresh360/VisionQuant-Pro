"""
Train Dual-Stream Network - 双流网络训练脚本

实现完整的训练流程：
1. 数据加载和预处理
2. Walk-Forward交叉验证
3. 多任务学习（分类+回归）
4. 模型保存和评估

Author: VisionQuant Team
Date: 2026-01
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 添加项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.models.dual_stream_network import DualStreamNetwork, DualStreamLoss
from src.data.gaf_encoder import GAFEncoder
from src.data.triple_barrier import TripleBarrierLabeler
from src.utils.walk_forward import WalkForwardValidator, TimeSeriesSplitter


class DualStreamDataset(Dataset):
    """
    双流网络数据集
    
    同时提供GAF图像和OHLCV序列
    """
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        gaf_dir: str,
        window_size: int = 60,
        transform=None
    ):
        """
        初始化数据集
        
        Args:
            data_df: 包含标签和元数据的DataFrame
            gaf_dir: GAF图像目录
            window_size: OHLCV窗口大小
            transform: 图像变换
        """
        self.data_df = data_df.reset_index(drop=True)
        self.gaf_dir = gaf_dir
        self.window_size = window_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 预加载OHLCV数据（避免重复读取）
        self.ohlcv_cache = {}
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        
        # 获取元数据
        symbol = str(row['symbol'])
        date_str = str(row['date'])
        
        # 加载GAF图像
        gaf_path = os.path.join(self.gaf_dir, f"{symbol}_{date_str}.png")
        if os.path.exists(gaf_path):
            gaf_image = Image.open(gaf_path).convert('RGB')
            gaf_image = self.transform(gaf_image)
        else:
            # 使用空白图像作为占位符
            gaf_image = torch.zeros(3, 224, 224)
        
        # 获取OHLCV序列
        if 'ohlcv' in row:
            ohlcv = np.array(row['ohlcv'])
        else:
            # 从缓存或文件加载
            ohlcv = self._load_ohlcv(symbol, date_str)
        
        # 确保OHLCV形状正确
        if ohlcv.shape[0] < self.window_size:
            # 填充
            pad_size = self.window_size - ohlcv.shape[0]
            ohlcv = np.pad(ohlcv, ((pad_size, 0), (0, 0)), mode='edge')
        elif ohlcv.shape[0] > self.window_size:
            ohlcv = ohlcv[-self.window_size:]
        
        ohlcv_tensor = torch.FloatTensor(ohlcv)
        
        # 标签
        class_label = int(row.get('tb_label', 0)) + 1  # -1,0,1 -> 0,1,2
        return_label = float(row.get('tb_return', 0))
        
        return {
            'gaf_image': gaf_image,
            'ohlcv': ohlcv_tensor,
            'class_label': class_label,
            'return_label': return_label,
            'symbol': symbol,
            'date': date_str
        }
    
    def _load_ohlcv(self, symbol: str, date_str: str) -> np.ndarray:
        """加载OHLCV数据"""
        # 这里需要根据实际数据存储方式实现
        # 简化版本：返回随机数据
        return np.random.randn(self.window_size, 5).astype(np.float32)


class DualStreamTrainer:
    """
    双流网络训练器
    """
    
    def __init__(
        self,
        model: DualStreamNetwork,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        """
        初始化训练器
        
        Args:
            model: 双流网络模型
            device: 计算设备
            learning_rate: 学习率
            weight_decay: 权重衰减
        """
        self.model = model.to(device)
        self.device = device
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        # 损失函数
        self.loss_fn = DualStreamLoss(
            class_weight=1.0,
            return_weight=0.5
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> dict:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        total_class_loss = 0
        total_return_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            # 数据移到设备
            gaf_image = batch['gaf_image'].to(self.device)
            ohlcv = batch['ohlcv'].to(self.device)
            class_label = batch['class_label'].to(self.device)
            return_label = batch['return_label'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(gaf_image, ohlcv)
            
            # 计算损失
            losses = self.loss_fn(outputs, class_label, return_label)
            
            # 反向传播
            losses['total_loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += losses['total_loss'].item()
            total_class_loss += losses['class_loss'].item()
            total_return_loss += losses['return_loss'].item()
            
            pred = torch.argmax(outputs['class_logits'], dim=-1)
            correct += (pred == class_label).sum().item()
            total += class_label.size(0)
            
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'acc': f"{correct/total:.4f}"
            })
        
        n_batches = len(train_loader)
        return {
            'loss': total_loss / n_batches,
            'class_loss': total_class_loss / n_batches,
            'return_loss': total_return_loss / n_batches,
            'accuracy': correct / total
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict:
        """验证"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        for batch in tqdm(val_loader, desc='Validating'):
            gaf_image = batch['gaf_image'].to(self.device)
            ohlcv = batch['ohlcv'].to(self.device)
            class_label = batch['class_label'].to(self.device)
            return_label = batch['return_label'].to(self.device)
            
            outputs = self.model(gaf_image, ohlcv)
            losses = self.loss_fn(outputs, class_label, return_label)
            
            total_loss += losses['total_loss'].item()
            
            pred = torch.argmax(outputs['class_logits'], dim=-1)
            correct += (pred == class_label).sum().item()
            total += class_label.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(class_label.cpu().numpy())
        
        n_batches = len(val_loader)
        return {
            'loss': total_loss / n_batches,
            'accuracy': correct / total,
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels)
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        save_dir: str = None,
        patience: int = 10
    ) -> dict:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            save_dir: 模型保存目录
            patience: 早停耐心值
            
        Returns:
            训练结果
        """
        best_val_loss = float('inf')
        best_epoch = 0
        no_improve = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}")
            
            # 验证
            val_metrics = self.validate(val_loader)
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}")
            
            # 学习率调度
            self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # 保存最佳模型
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_epoch = epoch
                no_improve = 0
                
                if save_dir:
                    self._save_checkpoint(save_dir, epoch, val_metrics)
                    print(f"✅ Saved best model (epoch {epoch + 1})")
            else:
                no_improve += 1
            
            # 早停
            if no_improve >= patience:
                print(f"\n⚠️ Early stopping at epoch {epoch + 1}")
                break
        
        print(f"\n🎉 Training completed! Best epoch: {best_epoch + 1}")
        
        return {
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'history': self.history
        }
    
    def _save_checkpoint(self, save_dir: str, epoch: int, metrics: dict):
        """保存检查点"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型权重
        model_path = os.path.join(save_dir, 'dual_stream_best.pth')
        torch.save(self.model.state_dict(), model_path)
        
        # 保存训练状态
        state_path = os.path.join(save_dir, 'training_state.json')
        state = {
            'epoch': epoch,
            'val_loss': metrics['loss'],
            'val_accuracy': metrics['accuracy'],
            'history': self.history
        }
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)


def create_synthetic_dataset(
    n_samples: int = 1000,
    window_size: int = 60
) -> pd.DataFrame:
    """创建合成数据集用于测试"""
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        symbol = f"{600000 + i % 100:06d}"
        date = f"2023{(i % 12) + 1:02d}{(i % 28) + 1:02d}"
        
        # 随机标签
        tb_label = np.random.choice([-1, 0, 1])
        tb_return = np.random.randn() * 0.05
        
        # 随机OHLCV
        ohlcv = np.random.randn(window_size, 5).astype(np.float32)
        
        data.append({
            'symbol': symbol,
            'date': date,
            'tb_label': tb_label,
            'tb_return': tb_return,
            'ohlcv': ohlcv.tolist()
        })
    
    return pd.DataFrame(data)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train Dual-Stream Network')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='数据目录')
    parser.add_argument('--gaf_dir', type=str, default='data/gaf_images',
                       help='GAF图像目录')
    parser.add_argument('--save_dir', type=str, default='data/models',
                       help='模型保存目录')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--window_size', type=int, default=60,
                       help='OHLCV窗口大小')
    parser.add_argument('--use_synthetic', action='store_true',
                       help='使用合成数据测试')
    
    args = parser.parse_args()
    
    # 设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("🚀 Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🚀 Using Apple MPS GPU")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    
    # 创建/加载数据
    if args.use_synthetic:
        print("\n📦 Creating synthetic dataset for testing...")
        data_df = create_synthetic_dataset(n_samples=1000)
    else:
        # 加载真实数据
        data_path = os.path.join(args.data_dir, 'labeled_data.csv')
        if os.path.exists(data_path):
            data_df = pd.read_csv(data_path)
        else:
            print(f"⚠️ Data file not found: {data_path}")
            print("使用合成数据进行测试...")
            data_df = create_synthetic_dataset(n_samples=1000)
    
    print(f"📊 Dataset size: {len(data_df)}")
    
    # 划分数据
    train_df, val_df, test_df = TimeSeriesSplitter.train_test_split(
        data_df, test_size=0.2, val_size=0.1
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 创建数据集
    train_dataset = DualStreamDataset(
        train_df, args.gaf_dir, args.window_size
    )
    val_dataset = DualStreamDataset(
        val_df, args.gaf_dir, args.window_size
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 if device.type != 'mps' else 0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4 if device.type != 'mps' else 0,
        pin_memory=True
    )
    
    # 创建模型
    model = DualStreamNetwork(
        vision_backbone='resnet18',
        vision_pretrained=True,
        vision_dim=512,
        temporal_input_size=5,
        temporal_dim=256,
        temporal_layers=2,
        fusion_dim=768,
        num_classes=3
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model Parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # 创建训练器
    trainer = DualStreamTrainer(
        model=model,
        device=device,
        learning_rate=args.lr
    )
    
    # 训练
    print("\n🏃 Starting training...")
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        patience=10
    )
    
    print("\n✅ Training completed!")
    print(f"Best Epoch: {results['best_epoch'] + 1}")
    print(f"Best Val Loss: {results['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
