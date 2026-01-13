"""
Dual-Stream Network - 双流网络

多模态融合架构：图像流 (Vision Stream) + 时序流 (Temporal Stream)

核心思想：
1. 图像流：从GAF图像中提取视觉特征（空间模式、形态特征）
2. 时序流：从原始OHLCV中提取时序特征（动态模式、趋势特征）
3. 跨模态融合：通过Cross-Attention融合两种模态的互补信息

理论基础：
- 图像捕捉"整体形态"（人类交易员看到的）
- 时序捕捉"动态变化"（数值指标难以表达的）
- 两者融合提供更完整的市场理解

Author: VisionQuant Team
Date: 2026-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional, Dict, Any
import numpy as np

# 导入自定义模块
from .temporal_encoder import TemporalEncoder


class VisionStream(nn.Module):
    """
    视觉流 - 处理GAF图像
    
    使用预训练的ResNet18作为骨干网络
    可选替换为ViT等更先进的架构
    """
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        output_dim: int = 512,
        freeze_backbone: bool = False
    ):
        """
        初始化视觉流
        
        Args:
            backbone: 骨干网络类型 ('resnet18', 'resnet34', 'resnet50')
            pretrained: 是否使用预训练权重
            output_dim: 输出特征维度
            freeze_backbone: 是否冻结骨干网络
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.output_dim = output_dim
        
        # 加载骨干网络
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            backbone_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            backbone_dim = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # 移除原始分类头
        self.backbone.fc = nn.Identity()
        
        # 冻结骨干网络（可选）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 特征投影层
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 用于Grad-CAM的钩子
        self.gradients = None
        self.activations = None
    
    def _save_gradient(self, grad):
        """保存梯度（用于Grad-CAM）"""
        self.gradients = grad
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [B, 3, H, W] GAF图像
            return_features: 是否返回中间特征（用于Grad-CAM）
            
        Returns:
            [B, output_dim] 视觉特征
        """
        # 获取最后一个卷积层的特征（用于Grad-CAM）
        if return_features:
            # 逐层前向传播以获取中间特征
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            
            # 保存激活值
            self.activations = x
            
            # 注册梯度钩子
            if x.requires_grad:
                x.register_hook(self._save_gradient)
            
            # 全局平均池化
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
        else:
            x = self.backbone(x)
        
        # 特征投影
        x = self.projection(x)
        
        return x
    
    def get_cam(self, grad_output: torch.Tensor) -> torch.Tensor:
        """
        计算Grad-CAM
        
        Args:
            grad_output: 输出梯度
            
        Returns:
            CAM热力图
        """
        if self.gradients is None or self.activations is None:
            raise RuntimeError("需要先调用forward(return_features=True)")
        
        # 计算权重
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # 加权求和
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam


class TemporalStream(nn.Module):
    """
    时序流 - 处理原始OHLCV数据
    
    使用 TCN + Self-Attention 架构
    """
    
    def __init__(
        self,
        input_size: int = 5,          # OHLCV
        output_dim: int = 256,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
        """
        初始化时序流
        
        Args:
            input_size: 输入特征维度（OHLCV=5）
            output_dim: 输出特征维度
            d_model: 模型隐藏维度
            num_layers: Attention层数
            num_heads: 注意力头数
            dropout: Dropout比率
        """
        super().__init__()
        
        self.encoder = TemporalEncoder(
            input_size=input_size,
            d_model=d_model,
            tcn_channels=[64, 128, d_model],
            num_attention_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 输出投影
        if d_model != output_dim:
            self.output_projection = nn.Linear(d_model, output_dim)
        else:
            self.output_projection = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: [B, T, 5] OHLCV序列
            return_attention: 是否返回注意力权重
            
        Returns:
            features: [B, output_dim] 时序特征
            attention_weights: 可选的注意力权重
        """
        _, global_output, attention_weights = self.encoder(
            x, return_attention=return_attention
        )
        
        features = self.output_projection(global_output)
        
        if return_attention and attention_weights:
            return features, attention_weights[-1]  # 返回最后一层的注意力
        
        return features, None


class CrossModalAttention(nn.Module):
    """
    跨模态注意力融合层
    
    让视觉特征和时序特征相互"关注"，学习互补信息
    
    例如：
    - 视觉流检测到"双底"形态
    - 时序流检测到"放量"信号
    - 融合后：确认"双底反转"的有效性
    """
    
    def __init__(
        self,
        vision_dim: int = 512,
        temporal_dim: int = 256,
        fusion_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        初始化跨模态注意力
        
        Args:
            vision_dim: 视觉特征维度
            temporal_dim: 时序特征维度
            fusion_dim: 融合后的维度
            num_heads: 注意力头数
            dropout: Dropout比率
        """
        super().__init__()
        
        self.vision_dim = vision_dim
        self.temporal_dim = temporal_dim
        self.fusion_dim = fusion_dim
        
        # 特征投影到同一空间
        self.vision_proj = nn.Linear(vision_dim, fusion_dim)
        self.temporal_proj = nn.Linear(temporal_dim, fusion_dim)
        
        # 跨模态注意力（Vision -> Temporal）
        self.v2t_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 跨模态注意力（Temporal -> Vision）
        self.t2v_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 门控机制（学习每个模态的重要性）
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        vision_features: torch.Tensor,
        temporal_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        跨模态融合
        
        Args:
            vision_features: [B, vision_dim] 视觉特征
            temporal_features: [B, temporal_dim] 时序特征
            
        Returns:
            fused_features: [B, fusion_dim] 融合特征
            gate_weights: [B, 2] 模态门控权重
        """
        # 投影到同一空间
        v_proj = self.vision_proj(vision_features)  # [B, fusion_dim]
        t_proj = self.temporal_proj(temporal_features)  # [B, fusion_dim]
        
        # 扩展维度以适应attention（需要序列维度）
        v_seq = v_proj.unsqueeze(1)  # [B, 1, fusion_dim]
        t_seq = t_proj.unsqueeze(1)  # [B, 1, fusion_dim]
        
        # Vision attends to Temporal
        v2t_out, _ = self.v2t_attention(v_seq, t_seq, t_seq)  # [B, 1, fusion_dim]
        
        # Temporal attends to Vision
        t2v_out, _ = self.t2v_attention(t_seq, v_seq, v_seq)  # [B, 1, fusion_dim]
        
        # 移除序列维度
        v2t_out = v2t_out.squeeze(1)
        t2v_out = t2v_out.squeeze(1)
        
        # 拼接
        concat = torch.cat([v2t_out, t2v_out], dim=-1)  # [B, fusion_dim * 2]
        
        # 门控权重
        gate_weights = self.gate(concat)  # [B, 2]
        
        # 加权融合
        weighted_v = gate_weights[:, 0:1] * v2t_out
        weighted_t = gate_weights[:, 1:2] * t2v_out
        
        # 最终融合
        fused = self.fusion_layer(torch.cat([weighted_v, weighted_t], dim=-1))
        
        return fused, gate_weights


class DualStreamNetwork(nn.Module):
    """
    双流网络 - 完整架构
    
    融合视觉流和时序流的多模态预测网络
    
    用途：
    1. 相似度检索：提取融合特征用于FAISS搜索
    2. 分类预测：Triple Barrier标签预测
    3. 回归预测：收益率预测
    """
    
    def __init__(
        self,
        # 视觉流参数
        vision_backbone: str = 'resnet18',
        vision_pretrained: bool = True,
        vision_dim: int = 512,
        # 时序流参数
        temporal_input_size: int = 5,
        temporal_dim: int = 256,
        temporal_layers: int = 2,
        # 融合参数
        fusion_dim: int = 768,
        num_classes: int = 3,  # Triple Barrier: -1, 0, 1
        dropout: float = 0.2
    ):
        """
        初始化双流网络
        
        Args:
            vision_backbone: 视觉骨干网络
            vision_pretrained: 是否使用预训练权重
            vision_dim: 视觉特征维度
            temporal_input_size: 时序输入维度
            temporal_dim: 时序特征维度
            temporal_layers: 时序Attention层数
            fusion_dim: 融合特征维度
            num_classes: 分类数量
            dropout: Dropout比率
        """
        super().__init__()
        
        self.vision_dim = vision_dim
        self.temporal_dim = temporal_dim
        self.fusion_dim = fusion_dim
        
        # 视觉流
        self.vision_stream = VisionStream(
            backbone=vision_backbone,
            pretrained=vision_pretrained,
            output_dim=vision_dim
        )
        
        # 时序流
        self.temporal_stream = TemporalStream(
            input_size=temporal_input_size,
            output_dim=temporal_dim,
            d_model=temporal_dim,
            num_layers=temporal_layers,
            dropout=dropout
        )
        
        # 跨模态融合
        self.cross_modal_fusion = CrossModalAttention(
            vision_dim=vision_dim,
            temporal_dim=temporal_dim,
            fusion_dim=fusion_dim,
            dropout=dropout
        )
        
        # 分类头（Triple Barrier预测）
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        # 回归头（收益率预测）
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1)
        )
        
        # 特征投影（用于相似度检索）
        self.feature_projection = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
    
    def forward(
        self,
        gaf_image: torch.Tensor,
        ohlcv_sequence: torch.Tensor,
        return_features: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            gaf_image: [B, 3, H, W] GAF图像
            ohlcv_sequence: [B, T, 5] OHLCV序列
            return_features: 是否返回中间特征
            return_attention: 是否返回注意力权重
            
        Returns:
            包含各种输出的字典：
            - 'class_logits': 分类logits
            - 'return_pred': 收益率预测
            - 'fused_features': 融合特征
            - 'gate_weights': 模态门控权重
            - 'vision_features': 视觉特征（可选）
            - 'temporal_features': 时序特征（可选）
            - 'temporal_attention': 时序注意力权重（可选）
        """
        results = {}
        
        # 视觉流
        vision_features = self.vision_stream(gaf_image, return_features=return_features)
        
        # 时序流
        temporal_features, temporal_attention = self.temporal_stream(
            ohlcv_sequence, return_attention=return_attention
        )
        
        # 跨模态融合
        fused_features, gate_weights = self.cross_modal_fusion(
            vision_features, temporal_features
        )
        
        # 分类预测
        class_logits = self.classifier(fused_features)
        
        # 回归预测
        return_pred = self.regressor(fused_features)
        
        # 特征投影（L2归一化，用于相似度检索）
        projected_features = self.feature_projection(fused_features)
        projected_features = F.normalize(projected_features, p=2, dim=-1)
        
        results['class_logits'] = class_logits
        results['return_pred'] = return_pred.squeeze(-1)
        results['fused_features'] = projected_features
        results['gate_weights'] = gate_weights
        
        if return_features:
            results['vision_features'] = vision_features
            results['temporal_features'] = temporal_features
        
        if return_attention and temporal_attention is not None:
            results['temporal_attention'] = temporal_attention
        
        return results
    
    def encode(
        self,
        gaf_image: torch.Tensor,
        ohlcv_sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        仅返回融合特征（用于FAISS检索）
        
        Args:
            gaf_image: [B, 3, H, W] GAF图像
            ohlcv_sequence: [B, T, 5] OHLCV序列
            
        Returns:
            [B, fusion_dim] L2归一化的融合特征
        """
        results = self.forward(gaf_image, ohlcv_sequence)
        return results['fused_features']
    
    def predict_class(
        self,
        gaf_image: torch.Tensor,
        ohlcv_sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        分类预测
        
        Returns:
            [B] 预测类别
        """
        results = self.forward(gaf_image, ohlcv_sequence)
        return torch.argmax(results['class_logits'], dim=-1)
    
    def predict_return(
        self,
        gaf_image: torch.Tensor,
        ohlcv_sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        收益率预测
        
        Returns:
            [B] 预测收益率
        """
        results = self.forward(gaf_image, ohlcv_sequence)
        return results['return_pred']


class DualStreamLoss(nn.Module):
    """
    双流网络损失函数
    
    多任务学习：分类损失 + 回归损失 + 对比损失
    """
    
    def __init__(
        self,
        class_weight: float = 1.0,
        return_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.class_weight = class_weight
        self.return_weight = return_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        class_labels: torch.Tensor,
        return_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            outputs: 模型输出字典
            class_labels: [B] 分类标签
            return_labels: [B] 收益率标签
            
        Returns:
            损失字典
        """
        # 分类损失
        class_loss = self.ce_loss(outputs['class_logits'], class_labels)
        
        # 回归损失（过滤NaN）
        valid_mask = ~torch.isnan(return_labels)
        if valid_mask.sum() > 0:
            return_loss = self.mse_loss(
                outputs['return_pred'][valid_mask],
                return_labels[valid_mask]
            )
        else:
            return_loss = torch.tensor(0.0, device=class_labels.device)
        
        # 总损失
        total_loss = (
            self.class_weight * class_loss +
            self.return_weight * return_loss
        )
        
        return {
            'total_loss': total_loss,
            'class_loss': class_loss,
            'return_loss': return_loss
        }


if __name__ == "__main__":
    # 测试双流网络
    print("Testing Dual-Stream Network...")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = DualStreamNetwork(
        vision_backbone='resnet18',
        vision_pretrained=False,  # 测试时不下载预训练权重
        vision_dim=512,
        temporal_input_size=5,
        temporal_dim=256,
        temporal_layers=2,
        fusion_dim=768,
        num_classes=3
    ).to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 测试输入
    batch_size = 4
    gaf_image = torch.randn(batch_size, 3, 224, 224).to(device)
    ohlcv_sequence = torch.randn(batch_size, 60, 5).to(device)
    
    print(f"\nGAF image shape: {gaf_image.shape}")
    print(f"OHLCV sequence shape: {ohlcv_sequence.shape}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(
            gaf_image, ohlcv_sequence,
            return_features=True,
            return_attention=True
        )
    
    print(f"\nOutputs:")
    print(f"  - class_logits: {outputs['class_logits'].shape}")
    print(f"  - return_pred: {outputs['return_pred'].shape}")
    print(f"  - fused_features: {outputs['fused_features'].shape}")
    print(f"  - gate_weights: {outputs['gate_weights'].shape}")
    print(f"  - Gate weights (Vision, Temporal): {outputs['gate_weights'][0].tolist()}")
    
    # 测试编码
    with torch.no_grad():
        encoded = model.encode(gaf_image, ohlcv_sequence)
    print(f"\nEncoded features shape: {encoded.shape}")
    
    # 测试预测
    with torch.no_grad():
        class_pred = model.predict_class(gaf_image, ohlcv_sequence)
        return_pred = model.predict_return(gaf_image, ohlcv_sequence)
    print(f"Class predictions: {class_pred.tolist()}")
    print(f"Return predictions: {return_pred.tolist()}")
    
    # 测试损失函数
    print("\nTesting Loss Function...")
    loss_fn = DualStreamLoss()
    
    class_labels = torch.randint(0, 3, (batch_size,)).to(device)
    return_labels = torch.randn(batch_size).to(device)
    
    with torch.no_grad():
        outputs = model(gaf_image, ohlcv_sequence)
    
    losses = loss_fn(outputs, class_labels, return_labels)
    print(f"Total loss: {losses['total_loss'].item():.4f}")
    print(f"Class loss: {losses['class_loss'].item():.4f}")
    print(f"Return loss: {losses['return_loss'].item():.4f}")
    
    print("\n✅ All tests passed!")
