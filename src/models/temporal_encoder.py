"""
Temporal Encoder - 时序编码器

基于 TCN (Temporal Convolutional Network) + Self-Attention 的时序特征提取器

理论基础：
- Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of 
  Generic Convolutional and Recurrent Networks for Sequence Modeling.
- Vaswani, A., et al. (2017). Attention is All You Need.

创新点：
1. TCN 捕捉局部时序模式（支撑位、阻力位等）
2. Self-Attention 捕捉长距离依赖（跨周期模式）
3. 多尺度特征融合

Author: VisionQuant Team
Date: 2026-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class CausalConv1d(nn.Module):
    """
    因果卷积层
    
    确保输出只依赖于当前和过去的输入，不包含未来信息
    这对于金融时序预测至关重要（防止未来函数）
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        **kwargs
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            **kwargs
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] 输入序列
            
        Returns:
            [B, C, T] 输出序列（保持长度不变）
        """
        out = self.conv(x)
        # 移除未来信息（因果性）
        return out[:, :, :-self.padding] if self.padding > 0 else out


class TemporalBlock(nn.Module):
    """
    TCN 基础块
    
    包含两层因果卷积 + 残差连接
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # 残差连接（如果通道数不同）
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
            
        Returns:
            [B, C', T]
        """
        residual = x if self.downsample is None else self.downsample(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        
        return self.relu(out + residual)


class TCN(nn.Module):
    """
    Temporal Convolutional Network (TCN)
    
    使用膨胀因果卷积实现大感受野，同时保持计算效率
    
    感受野计算：receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilations)
    """
    
    def __init__(
        self,
        input_size: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        """
        初始化TCN
        
        Args:
            input_size: 输入特征维度
            num_channels: 各层通道数列表，如 [64, 128, 256]
            kernel_size: 卷积核大小
            dropout: Dropout比率
        """
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i  # 指数膨胀
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, dilation, dropout
            ))
        
        self.network = nn.Sequential(*layers)
        self.output_size = num_channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C] 输入序列
            
        Returns:
            [B, T, C'] 输出序列
        """
        # 转换维度: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        
        out = self.network(x)
        
        # 转换回: [B, C', T] -> [B, T, C']
        return out.transpose(1, 2)


class PositionalEncoding(nn.Module):
    """
    位置编码
    
    为序列中的每个位置添加位置信息
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]
            
        Returns:
            [B, T, C]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalSelfAttention(nn.Module):
    """
    时序自注意力模块
    
    捕捉序列中的长距离依赖关系
    例如：识别20天前的支撑位与当前价格的关系
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = True
    ):
        """
        初始化时序自注意力
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: Dropout比率
            causal: 是否使用因果掩码（只看过去）
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.causal = causal
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, C] 输入序列
            mask: 可选的注意力掩码
            
        Returns:
            output: [B, T, C] 输出序列
            attention_weights: [B, num_heads, T, T] 注意力权重
        """
        B, T, C = x.shape
        residual = x
        
        # 线性变换
        Q = self.w_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # 因果掩码（只看过去）
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # 应用额外掩码
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        context = torch.matmul(attention_weights, V)
        
        # 拼接多头
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        
        # 输出投影
        output = self.w_o(context)
        output = self.dropout(output)
        
        # 残差连接 + 层归一化
        output = self.layer_norm(output + residual)
        
        return output, attention_weights


class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear2(F.gelu(self.linear1(x)))
        x = self.dropout(x)
        return self.layer_norm(x + residual)


class TemporalEncoder(nn.Module):
    """
    时序编码器
    
    组合 TCN + Self-Attention 的完整编码器
    
    架构：
    1. 输入投影层
    2. TCN层（捕捉局部模式）
    3. Self-Attention层（捕捉全局依赖）
    4. 输出投影层
    """
    
    def __init__(
        self,
        input_size: int = 5,           # OHLCV
        d_model: int = 256,            # 模型维度
        tcn_channels: List[int] = [64, 128, 256],
        num_attention_layers: int = 2,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.2,
        max_seq_len: int = 256
    ):
        """
        初始化时序编码器
        
        Args:
            input_size: 输入特征维度（OHLCV=5）
            d_model: 模型隐藏维度
            tcn_channels: TCN各层通道数
            num_attention_layers: Self-Attention层数
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏维度
            dropout: Dropout比率
            max_seq_len: 最大序列长度
        """
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # 输入投影
        self.input_projection = nn.Linear(input_size, tcn_channels[0])
        
        # TCN 层
        self.tcn = TCN(
            input_size=tcn_channels[0],
            num_channels=tcn_channels,
            kernel_size=3,
            dropout=dropout
        )
        
        # 维度调整（TCN输出 -> Attention输入）
        self.tcn_to_attention = nn.Linear(tcn_channels[-1], d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Self-Attention 层
        self.attention_layers = nn.ModuleList([
            nn.ModuleList([
                TemporalSelfAttention(d_model, num_heads, dropout, causal=True),
                FeedForward(d_model, d_ff, dropout)
            ])
            for _ in range(num_attention_layers)
        ])
        
        # 输出投影
        self.output_projection = nn.Linear(d_model, d_model)
        
        # 全局池化后的投影
        self.global_projection = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        前向传播
        
        Args:
            x: [B, T, C] 输入序列（OHLCV数据）
            return_attention: 是否返回注意力权重
            
        Returns:
            sequence_output: [B, T, d_model] 序列输出
            global_output: [B, d_model] 全局表示（用于分类/回归）
            attention_weights: 可选的注意力权重列表
        """
        # 输入投影
        x = self.input_projection(x)
        
        # TCN 编码
        x = self.tcn(x)
        
        # 维度调整
        x = self.tcn_to_attention(x)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Self-Attention 层
        attention_weights = []
        for attn, ff in self.attention_layers:
            x, attn_w = attn(x)
            x = ff(x)
            if return_attention:
                attention_weights.append(attn_w)
        
        # 序列输出
        sequence_output = self.output_projection(x)
        
        # 全局表示（取最后一个时间步 或 平均池化）
        # 使用最后一个时间步（因果性）
        global_output = self.global_projection(sequence_output[:, -1, :])
        
        if return_attention:
            return sequence_output, global_output, attention_weights
        else:
            return sequence_output, global_output, None
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        仅返回全局特征向量
        
        Args:
            x: [B, T, C] 输入序列
            
        Returns:
            [B, d_model] 全局特征向量
        """
        _, global_output, _ = self.forward(x)
        return global_output


class TemporalEncoderLite(nn.Module):
    """
    轻量级时序编码器
    
    只使用 TCN，不含 Attention，适用于资源受限场景
    """
    
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        output_size: int = 256,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        channels = [hidden_size] * num_layers
        channels[-1] = output_size
        
        self.tcn = TCN(
            input_size=input_size,
            num_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        self.output_projection = nn.Linear(output_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]
            
        Returns:
            [B, output_size] 全局特征
        """
        # TCN 编码
        sequence = self.tcn(x)  # [B, T, output_size]
        
        # 取最后一个时间步
        global_feature = sequence[:, -1, :]
        
        return self.output_projection(global_feature)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


if __name__ == "__main__":
    # 测试时序编码器
    print("Testing Temporal Encoder...")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    encoder = TemporalEncoder(
        input_size=5,      # OHLCV
        d_model=256,
        tcn_channels=[64, 128, 256],
        num_attention_layers=2,
        num_heads=8
    ).to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 测试输入
    batch_size = 4
    seq_len = 60
    input_size = 5
    
    x = torch.randn(batch_size, seq_len, input_size).to(device)
    print(f"\nInput shape: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        seq_out, global_out, attn_weights = encoder(x, return_attention=True)
    
    print(f"Sequence output shape: {seq_out.shape}")
    print(f"Global output shape: {global_out.shape}")
    print(f"Number of attention layers: {len(attn_weights)}")
    print(f"Attention weights shape: {attn_weights[0].shape}")
    
    # 测试编码
    with torch.no_grad():
        encoded = encoder.encode(x)
    print(f"Encoded shape: {encoded.shape}")
    
    # 测试轻量级版本
    print("\n" + "=" * 50)
    print("Testing Temporal Encoder Lite...")
    
    encoder_lite = TemporalEncoderLite(
        input_size=5,
        hidden_size=128,
        output_size=256
    ).to(device)
    
    lite_params = sum(p.numel() for p in encoder_lite.parameters())
    print(f"Lite model parameters: {lite_params:,}")
    
    with torch.no_grad():
        encoded_lite = encoder_lite.encode(x)
    print(f"Lite encoded shape: {encoded_lite.shape}")
    
    print("\n✅ All tests passed!")
