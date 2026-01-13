"""
Grad-CAM (Gradient-weighted Class Activation Mapping)

可解释性可视化工具，显示模型关注的图像区域

理论基础：
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from 
  Deep Networks via Gradient-based Localization.

应用场景：
- 解释为什么模型预测"看涨"或"看跌"
- 验证模型是否学习到有意义的K线形态
- 调试和改进模型

Author: VisionQuant Team
Date: 2026-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from typing import Optional, Tuple, List, Dict, Union
import warnings


class GradCAM:
    """
    Grad-CAM 可解释性可视化
    
    通过计算目标类别相对于特征图的梯度，生成类激活热力图
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module = None,
        device: torch.device = None
    ):
        """
        初始化Grad-CAM
        
        Args:
            model: PyTorch模型
            target_layer: 目标层（用于计算CAM）
            device: 计算设备
        """
        self.model = model
        self.device = device or torch.device('cpu')
        self.target_layer = target_layer
        
        # 存储梯度和激活值
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        if target_layer is not None:
            self._register_hooks(target_layer)
    
    def _register_hooks(self, target_layer: nn.Module):
        """注册前向和反向钩子"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
    
    def _find_target_layer(self, model: nn.Module) -> nn.Module:
        """自动查找目标层（最后一个卷积层）"""
        target_layer = None
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
        
        if target_layer is None:
            raise ValueError("未找到卷积层，请手动指定target_layer")
        
        return target_layer
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        eigen_smooth: bool = False
    ) -> np.ndarray:
        """
        生成Grad-CAM热力图
        
        Args:
            input_tensor: 输入图像 [1, C, H, W]
            target_class: 目标类别（None则使用预测类别）
            eigen_smooth: 是否使用特征值平滑
            
        Returns:
            CAM热力图 [H, W]
        """
        self.model.eval()
        
        # 确保需要梯度
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        # 前向传播
        output = self.model(input_tensor)
        
        # 处理不同的输出格式
        if isinstance(output, dict):
            logits = output.get('class_logits', output.get('logits'))
        elif isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        # 确定目标类别
        if target_class is None:
            target_class = torch.argmax(logits, dim=-1).item()
        
        # 反向传播
        self.model.zero_grad()
        
        # 创建one-hot目标
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        
        # 反向传播
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # 检查梯度和激活值
        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "未捕获到梯度或激活值，请确保正确设置了target_layer"
            )
        
        # 计算权重（全局平均池化梯度）
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # 加权求和
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # ReLU（只保留正值）
        cam = F.relu(cam)
        
        # 归一化
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    def generate_cam_for_dual_stream(
        self,
        model: nn.Module,
        gaf_image: torch.Tensor,
        ohlcv_sequence: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        为双流网络生成CAM
        
        Args:
            model: 双流网络模型
            gaf_image: GAF图像 [1, 3, H, W]
            ohlcv_sequence: OHLCV序列 [1, T, 5]
            target_class: 目标类别
            
        Returns:
            vision_cam: 视觉流CAM
            temporal_attention: 时序流注意力权重
        """
        model.eval()
        
        # 需要梯度
        gaf_image = gaf_image.to(self.device).requires_grad_(True)
        ohlcv_sequence = ohlcv_sequence.to(self.device)
        
        # 获取视觉流的最后一个卷积层
        vision_target_layer = None
        for name, module in model.vision_stream.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                vision_target_layer = module
        
        if vision_target_layer is None:
            raise ValueError("未找到视觉流的卷积层")
        
        # 注册钩子
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        handle_f = vision_target_layer.register_forward_hook(forward_hook)
        handle_b = vision_target_layer.register_full_backward_hook(backward_hook)
        
        try:
            # 前向传播
            outputs = model(
                gaf_image, ohlcv_sequence,
                return_attention=True
            )
            
            logits = outputs['class_logits']
            
            # 确定目标类别
            if target_class is None:
                target_class = torch.argmax(logits, dim=-1).item()
            
            # 反向传播
            model.zero_grad()
            one_hot = torch.zeros_like(logits)
            one_hot[0, target_class] = 1
            logits.backward(gradient=one_hot, retain_graph=True)
            
            # 计算视觉CAM
            if activations and gradients:
                act = activations[-1]
                grad = gradients[-1]
                
                weights = torch.mean(grad, dim=[2, 3], keepdim=True)
                cam = torch.sum(weights * act, dim=1, keepdim=True)
                cam = F.relu(cam)
                cam = cam.squeeze().cpu().numpy()
                cam = cam - cam.min()
                cam = cam / (cam.max() + 1e-8)
                vision_cam = cam
            else:
                vision_cam = np.zeros((14, 14))
            
            # 获取时序注意力
            if 'temporal_attention' in outputs:
                temporal_attention = outputs['temporal_attention'][0].mean(dim=0).cpu().numpy()
            else:
                temporal_attention = np.zeros((60, 60))
            
        finally:
            handle_f.remove()
            handle_b.remove()
        
        return vision_cam, temporal_attention


def overlay_cam_on_image(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    将CAM热力图叠加到原始图像上
    
    Args:
        image: 原始图像 [H, W, 3]
        cam: CAM热力图 [h, w]
        alpha: 透明度
        colormap: OpenCV颜色映射
        
    Returns:
        叠加后的图像 [H, W, 3]
    """
    # 确保图像是uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # 调整CAM尺寸
    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
    
    # 应用颜色映射
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8),
        colormap
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 叠加
    overlaid = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
    
    return overlaid


def visualize_dual_stream_attention(
    gaf_image: np.ndarray,
    vision_cam: np.ndarray,
    temporal_attention: np.ndarray,
    ohlcv: np.ndarray,
    prediction: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8)
):
    """
    可视化双流网络的注意力
    
    Args:
        gaf_image: 原始GAF图像
        vision_cam: 视觉流CAM
        temporal_attention: 时序流注意力权重
        ohlcv: OHLCV数据
        prediction: 预测结果字典
        save_path: 保存路径
        figsize: 图像尺寸
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # 1. 原始GAF图像
    axes[0, 0].imshow(gaf_image)
    axes[0, 0].set_title('GAF Image (GASF+GADF+MTF)', fontsize=12)
    axes[0, 0].axis('off')
    
    # 2. 视觉CAM叠加
    cam_overlay = overlay_cam_on_image(gaf_image, vision_cam)
    axes[0, 1].imshow(cam_overlay)
    axes[0, 1].set_title('Vision Grad-CAM', fontsize=12)
    axes[0, 1].axis('off')
    
    # 3. 视觉CAM热力图
    im = axes[0, 2].imshow(vision_cam, cmap='jet', aspect='auto')
    axes[0, 2].set_title('CAM Heatmap', fontsize=12)
    plt.colorbar(im, ax=axes[0, 2])
    
    # 4. 价格走势
    close_prices = ohlcv[:, 3] if ohlcv.shape[1] > 3 else ohlcv[:, 0]
    axes[1, 0].plot(close_prices, 'b-', linewidth=1.5)
    axes[1, 0].fill_between(range(len(close_prices)), close_prices.min(), close_prices,
                            alpha=0.3)
    axes[1, 0].set_title('Price Series', fontsize=12)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Price')
    
    # 5. 时序注意力热力图
    if temporal_attention.size > 0:
        im2 = axes[1, 1].imshow(temporal_attention, cmap='viridis', aspect='auto')
        axes[1, 1].set_title('Temporal Self-Attention', fontsize=12)
        axes[1, 1].set_xlabel('Key Position')
        axes[1, 1].set_ylabel('Query Position')
        plt.colorbar(im2, ax=axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, 'No Attention Data', ha='center', va='center')
        axes[1, 1].axis('off')
    
    # 6. 预测结果
    axes[1, 2].axis('off')
    
    # 预测类别颜色
    class_colors = {0: 'red', 1: 'gray', 2: 'green'}
    class_names = {0: 'BEARISH', 1: 'NEUTRAL', 2: 'BULLISH'}
    
    pred_class = prediction.get('class', 1)
    confidence = prediction.get('confidence', 0)
    pred_return = prediction.get('return', 0)
    gate_weights = prediction.get('gate_weights', [0.5, 0.5])
    
    text = f"""
    Prediction Summary
    ──────────────────
    
    Class: {class_names.get(pred_class, 'UNKNOWN')}
    Confidence: {confidence:.2%}
    Expected Return: {pred_return:+.2%}
    
    Modal Weights:
    • Vision: {gate_weights[0]:.1%}
    • Temporal: {gate_weights[1]:.1%}
    """
    
    axes[1, 2].text(
        0.1, 0.5, text,
        transform=axes[1, 2].transAxes,
        fontsize=11,
        verticalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # 添加预测类别颜色标记
    axes[1, 2].add_patch(plt.Circle(
        (0.8, 0.7), 0.1,
        transform=axes[1, 2].transAxes,
        color=class_colors.get(pred_class, 'gray'),
        alpha=0.7
    ))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {save_path}")
    
    plt.show()
    
    return fig


def batch_generate_explanations(
    model: nn.Module,
    dataloader,
    output_dir: str,
    device: torch.device,
    num_samples: int = 10
):
    """
    批量生成可解释性可视化
    
    Args:
        model: 模型
        dataloader: 数据加载器
        output_dir: 输出目录
        device: 设备
        num_samples: 样本数量
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    grad_cam = GradCAM(model, device=device)
    
    count = 0
    for batch in dataloader:
        if count >= num_samples:
            break
        
        gaf_image = batch['gaf_image'].to(device)
        ohlcv = batch['ohlcv'].to(device)
        
        for i in range(gaf_image.size(0)):
            if count >= num_samples:
                break
            
            try:
                # 生成CAM
                vision_cam, temporal_attention = grad_cam.generate_cam_for_dual_stream(
                    model,
                    gaf_image[i:i+1],
                    ohlcv[i:i+1]
                )
                
                # 获取预测
                with torch.no_grad():
                    outputs = model(gaf_image[i:i+1], ohlcv[i:i+1])
                    pred_class = torch.argmax(outputs['class_logits'], dim=-1).item()
                    pred_return = outputs['return_pred'][0].item()
                    gate_weights = outputs['gate_weights'][0].cpu().numpy()
                
                # 准备数据
                img_np = gaf_image[i].cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                
                ohlcv_np = ohlcv[i].cpu().numpy()
                
                prediction = {
                    'class': pred_class,
                    'confidence': F.softmax(outputs['class_logits'][0], dim=-1)[pred_class].item(),
                    'return': pred_return,
                    'gate_weights': gate_weights
                }
                
                # 可视化
                save_path = os.path.join(output_dir, f'explanation_{count:03d}.png')
                visualize_dual_stream_attention(
                    img_np, vision_cam, temporal_attention,
                    ohlcv_np, prediction, save_path
                )
                
                count += 1
                
            except Exception as e:
                print(f"Error generating explanation: {e}")
                continue
    
    print(f"✅ Generated {count} explanations in {output_dir}")


if __name__ == "__main__":
    # 测试Grad-CAM
    print("Testing Grad-CAM...")
    
    # 创建简单的测试模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 3)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    model = SimpleModel()
    
    # 创建Grad-CAM
    grad_cam = GradCAM(model, target_layer=model.conv2)
    
    # 测试输入
    x = torch.randn(1, 3, 224, 224)
    
    # 生成CAM
    cam = grad_cam.generate_cam(x)
    
    print(f"Input shape: {x.shape}")
    print(f"CAM shape: {cam.shape}")
    print(f"CAM range: [{cam.min():.3f}, {cam.max():.3f}]")
    
    print("\n✅ Grad-CAM test passed!")
