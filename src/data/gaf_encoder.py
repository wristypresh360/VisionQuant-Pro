"""
GAF (Gramian Angular Field) Encoder
将时序数据转换为图像的数学方法

理论基础：
- Wang, Z., & Oates, T. (2015). Imaging time-series to improve classification and imputation.
- 通过极坐标转换，将时序数据编码为图像，保留时间依赖关系

创新点：
1. GASF (Gramian Angular Summation Field): 捕捉时序的整体趋势
2. GADF (Gramian Angular Difference Field): 捕捉时序的局部变化
3. MTF (Markov Transition Field): 捕捉状态转移概率
4. 多通道融合：GASF + GADF + MTF 作为RGB三通道

Author: VisionQuant Team
Date: 2026-01
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union, List
import warnings


class GAFEncoder:
    """
    Gramian Angular Field 编码器
    
    将一维时序数据转换为二维图像，用于CNN/ViT等视觉模型
    
    数学原理：
    1. 标准化: x_scaled ∈ [-1, 1]
    2. 角度转换: φ = arccos(x_scaled)
    3. GASF: G[i,j] = cos(φ_i + φ_j) = x_i * x_j - sqrt(1-x_i²) * sqrt(1-x_j²)
    4. GADF: G[i,j] = sin(φ_i - φ_j) = sqrt(1-x_i²) * x_j - x_i * sqrt(1-x_j²)
    """
    
    def __init__(self, image_size: int = 224, method: str = 'summation'):
        """
        初始化GAF编码器
        
        Args:
            image_size: 输出图像尺寸 (image_size x image_size)
            method: 'summation' (GASF), 'difference' (GADF), 或 'both'
        """
        self.image_size = image_size
        self.method = method
        
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """
        将时序标准化到[-1, 1]区间
        
        使用 min-max 标准化，确保数据在 arccos 的定义域内
        """
        x_min = np.min(x)
        x_max = np.max(x)
        
        if x_max - x_min < 1e-8:
            # 处理常数序列
            return np.zeros_like(x)
        
        # 标准化到 [-1, 1]
        x_scaled = (x - x_min) / (x_max - x_min) * 2 - 1
        
        # 确保在 [-1, 1] 范围内（处理数值误差）
        x_scaled = np.clip(x_scaled, -1 + 1e-8, 1 - 1e-8)
        
        return x_scaled
    
    def _paa(self, x: np.ndarray, segments: int) -> np.ndarray:
        """
        Piecewise Aggregate Approximation (PAA)
        
        将长时序压缩为指定长度，同时保留主要特征
        
        Args:
            x: 输入时序
            segments: 目标长度
            
        Returns:
            压缩后的时序
        """
        n = len(x)
        if n == segments:
            return x
        elif n < segments:
            # 使用线性插值扩展
            indices = np.linspace(0, n - 1, segments)
            return np.interp(indices, np.arange(n), x)
        else:
            # PAA 压缩
            segment_size = n / segments
            paa_result = np.zeros(segments)
            for i in range(segments):
                start = int(i * segment_size)
                end = int((i + 1) * segment_size)
                paa_result[i] = np.mean(x[start:end])
            return paa_result
    
    def _gasf(self, x_scaled: np.ndarray) -> np.ndarray:
        """
        计算 Gramian Angular Summation Field (GASF)
        
        GASF[i,j] = cos(φ_i + φ_j)
                  = cos(φ_i)cos(φ_j) - sin(φ_i)sin(φ_j)
                  = x_i * x_j - sqrt(1-x_i²) * sqrt(1-x_j²)
        
        物理含义：捕捉时序的整体趋势和时间点之间的相关性
        """
        # 计算 cos(φ) = x_scaled
        cos_phi = x_scaled
        
        # 计算 sin(φ) = sqrt(1 - x²)
        sin_phi = np.sqrt(1 - x_scaled ** 2)
        
        # GASF = cos(φ_i + φ_j) = cos_i * cos_j - sin_i * sin_j
        gasf = np.outer(cos_phi, cos_phi) - np.outer(sin_phi, sin_phi)
        
        return gasf
    
    def _gadf(self, x_scaled: np.ndarray) -> np.ndarray:
        """
        计算 Gramian Angular Difference Field (GADF)
        
        GADF[i,j] = sin(φ_i - φ_j)
                  = sin(φ_i)cos(φ_j) - cos(φ_i)sin(φ_j)
                  = sqrt(1-x_i²) * x_j - x_i * sqrt(1-x_j²)
        
        物理含义：捕捉时序的局部变化和方向性
        """
        # 计算 cos(φ) = x_scaled
        cos_phi = x_scaled
        
        # 计算 sin(φ) = sqrt(1 - x²)
        sin_phi = np.sqrt(1 - x_scaled ** 2)
        
        # GADF = sin(φ_i - φ_j) = sin_i * cos_j - cos_i * sin_j
        gadf = np.outer(sin_phi, cos_phi) - np.outer(cos_phi, sin_phi)
        
        return gadf
    
    def _mtf(self, x: np.ndarray, n_bins: int = 8) -> np.ndarray:
        """
        计算 Markov Transition Field (MTF)
        
        MTF[i,j] = P(q_j | q_i)
        
        将时序量化为离散状态，计算状态转移概率矩阵
        
        物理含义：捕捉时序的动态转移特性
        """
        n = len(x)
        
        # 将数据量化为 n_bins 个区间
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < 1e-8:
            return np.zeros((n, n))
        
        # 计算每个点所属的区间
        bins = np.linspace(x_min, x_max, n_bins + 1)
        quantized = np.digitize(x, bins[1:-1])  # [0, n_bins-1]
        
        # 计算马尔可夫转移矩阵
        transition_matrix = np.zeros((n_bins, n_bins))
        for i in range(n - 1):
            transition_matrix[quantized[i], quantized[i + 1]] += 1
        
        # 归一化（按行）
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除零
        transition_matrix = transition_matrix / row_sums
        
        # 构建 MTF
        mtf = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                mtf[i, j] = transition_matrix[quantized[i], quantized[j]]
        
        return mtf
    
    def encode(self, time_series: np.ndarray) -> np.ndarray:
        """
        将时序编码为GAF图像
        
        Args:
            time_series: 1D 时序数据 [T]
            
        Returns:
            GAF图像 [H, W] 或 [H, W, C]
        """
        # 确保是1D数组
        x = np.asarray(time_series).flatten()
        
        # PAA 压缩到目标尺寸
        x_paa = self._paa(x, self.image_size)
        
        # 标准化
        x_scaled = self._normalize(x_paa)
        
        if self.method == 'summation':
            return self._gasf(x_scaled)
        elif self.method == 'difference':
            return self._gadf(x_scaled)
        elif self.method == 'both':
            gasf = self._gasf(x_scaled)
            gadf = self._gadf(x_scaled)
            # 堆叠为2通道
            return np.stack([gasf, gadf], axis=-1)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def encode_multichannel(self, time_series: np.ndarray) -> np.ndarray:
        """
        将时序编码为多通道GAF图像 (GASF + GADF + MTF 作为 RGB)
        
        Args:
            time_series: 1D 时序数据 [T]
            
        Returns:
            3通道图像 [H, W, 3]，值域 [0, 1]
        """
        x = np.asarray(time_series).flatten()
        
        # PAA 压缩
        x_paa = self._paa(x, self.image_size)
        
        # 标准化
        x_scaled = self._normalize(x_paa)
        
        # 计算三个通道
        gasf = self._gasf(x_scaled)  # [-1, 1]
        gadf = self._gadf(x_scaled)  # [-1, 1]
        mtf = self._mtf(x_paa)       # [0, 1]
        
        # 归一化到 [0, 1]
        gasf_norm = (gasf + 1) / 2
        gadf_norm = (gadf + 1) / 2
        mtf_norm = mtf  # 已经在 [0, 1]
        
        # 堆叠为RGB图像
        rgb = np.stack([gasf_norm, gadf_norm, mtf_norm], axis=-1)
        
        return rgb.astype(np.float32)
    
    def encode_ohlcv(self, ohlcv: np.ndarray) -> np.ndarray:
        """
        将OHLCV数据编码为多通道GAF图像
        
        Args:
            ohlcv: [T, 5] 数组，列为 [Open, High, Low, Close, Volume]
            
        Returns:
            多通道图像 [H, W, 3]
            - R: Close价格的GASF
            - G: High-Low波动的GADF
            - B: Volume的MTF
        """
        ohlcv = np.asarray(ohlcv)
        
        if ohlcv.ndim == 1:
            # 如果只是单列数据，使用简单编码
            return self.encode_multichannel(ohlcv)
        
        if ohlcv.shape[1] < 4:
            raise ValueError(f"Expected at least 4 columns (OHLC), got {ohlcv.shape[1]}")
        
        # 提取各列
        close = ohlcv[:, 3]  # Close
        volatility = ohlcv[:, 1] - ohlcv[:, 2]  # High - Low
        volume = ohlcv[:, 4] if ohlcv.shape[1] > 4 else np.ones(len(close))
        
        # PAA 压缩
        close_paa = self._paa(close, self.image_size)
        vol_paa = self._paa(volatility, self.image_size)
        volume_paa = self._paa(volume, self.image_size)
        
        # 标准化
        close_scaled = self._normalize(close_paa)
        vol_scaled = self._normalize(vol_paa)
        
        # 计算各通道
        r_channel = self._gasf(close_scaled)  # Close的GASF
        g_channel = self._gadf(vol_scaled)    # 波动的GADF
        b_channel = self._mtf(volume_paa)      # 成交量的MTF
        
        # 归一化到 [0, 1]
        r_norm = (r_channel + 1) / 2
        g_norm = (g_channel + 1) / 2
        b_norm = b_channel
        
        # 堆叠为RGB
        rgb = np.stack([r_norm, g_norm, b_norm], axis=-1)
        
        return rgb.astype(np.float32)
    
    def to_pil_image(self, gaf_image: np.ndarray) -> Image.Image:
        """
        将GAF数组转换为PIL Image
        
        Args:
            gaf_image: GAF图像数组，可以是 [H, W] 或 [H, W, C]
            
        Returns:
            PIL Image对象
        """
        if gaf_image.ndim == 2:
            # 单通道，归一化到 [0, 255]
            img_normalized = ((gaf_image + 1) / 2 * 255).astype(np.uint8)
            return Image.fromarray(img_normalized, mode='L')
        else:
            # 多通道，假设已经在 [0, 1]
            img_normalized = (gaf_image * 255).astype(np.uint8)
            if gaf_image.shape[-1] == 3:
                return Image.fromarray(img_normalized, mode='RGB')
            else:
                return Image.fromarray(img_normalized[..., 0], mode='L')
    
    def save(self, gaf_image: np.ndarray, path: str) -> None:
        """
        保存GAF图像到文件
        
        Args:
            gaf_image: GAF图像数组
            path: 保存路径
        """
        pil_image = self.to_pil_image(gaf_image)
        pil_image.save(path)


class MultiScaleGAFEncoder:
    """
    多尺度GAF编码器
    
    同时生成多个时间尺度的GAF图像，捕捉不同周期的模式
    """
    
    def __init__(self, image_size: int = 224, scales: List[int] = [20, 60, 120]):
        """
        初始化多尺度编码器
        
        Args:
            image_size: 输出图像尺寸
            scales: 时间尺度列表（天数）
        """
        self.image_size = image_size
        self.scales = scales
        self.encoders = [GAFEncoder(image_size=image_size) for _ in scales]
    
    def encode(self, time_series: np.ndarray) -> List[np.ndarray]:
        """
        生成多尺度GAF图像
        
        Args:
            time_series: 完整时序数据
            
        Returns:
            多个尺度的GAF图像列表
        """
        results = []
        n = len(time_series)
        
        for scale, encoder in zip(self.scales, self.encoders):
            # 取最近 scale 天的数据
            if n >= scale:
                segment = time_series[-scale:]
            else:
                segment = time_series
            
            gaf = encoder.encode_multichannel(segment)
            results.append(gaf)
        
        return results
    
    def encode_concatenated(self, time_series: np.ndarray) -> np.ndarray:
        """
        生成拼接的多尺度GAF图像
        
        Args:
            time_series: 完整时序数据
            
        Returns:
            拼接后的图像 [H, W*num_scales, 3]
        """
        gaf_images = self.encode(time_series)
        return np.concatenate(gaf_images, axis=1)


def generate_gaf_dataset(
    data_loader,
    symbols: List[str],
    output_dir: str,
    window_size: int = 60,
    image_size: int = 224,
    stride: int = 5
) -> None:
    """
    批量生成GAF图像数据集
    
    Args:
        data_loader: 数据加载器实例
        symbols: 股票代码列表
        output_dir: 输出目录
        window_size: 时间窗口大小（天）
        image_size: 图像尺寸
        stride: 滑动窗口步长
    """
    import os
    from tqdm import tqdm
    
    encoder = GAFEncoder(image_size=image_size)
    os.makedirs(output_dir, exist_ok=True)
    
    for symbol in tqdm(symbols, desc="Generating GAF images"):
        try:
            df = data_loader.get_stock_data(symbol)
            if df is None or len(df) < window_size:
                continue
            
            # 提取OHLCV
            ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [c for c in ohlcv_cols if c in df.columns]
            if len(available_cols) < 4:
                continue
            
            ohlcv = df[available_cols].values
            dates = df.index if hasattr(df.index, 'strftime') else df.index
            
            # 滑动窗口生成
            for i in range(0, len(ohlcv) - window_size + 1, stride):
                window = ohlcv[i:i + window_size]
                end_date = dates[i + window_size - 1]
                
                # 格式化日期
                if hasattr(end_date, 'strftime'):
                    date_str = end_date.strftime('%Y%m%d')
                else:
                    date_str = str(end_date).replace('-', '')[:8]
                
                # 生成GAF图像
                gaf_image = encoder.encode_ohlcv(window)
                
                # 保存
                filename = f"{symbol}_{date_str}.png"
                filepath = os.path.join(output_dir, filename)
                encoder.save(gaf_image, filepath)
                
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue


if __name__ == "__main__":
    # 测试GAF编码器
    import matplotlib.pyplot as plt
    
    # 生成测试数据（模拟股价）
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 100)
    price = 100 + 10 * np.sin(t) + 5 * np.sin(2 * t) + np.random.randn(100) * 2
    
    # 创建编码器
    encoder = GAFEncoder(image_size=64)
    
    # 编码
    gasf = encoder.encode(price)
    
    # 多通道编码
    gaf_rgb = encoder.encode_multichannel(price)
    
    print(f"Input shape: {price.shape}")
    print(f"GASF shape: {gasf.shape}")
    print(f"RGB GAF shape: {gaf_rgb.shape}")
    
    # 可视化
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].plot(price)
    axes[0].set_title('Original Time Series')
    
    axes[1].imshow(gasf, cmap='RdBu_r', origin='lower')
    axes[1].set_title('GASF')
    
    axes[2].imshow(gaf_rgb[..., 0], cmap='RdBu_r', origin='lower')
    axes[2].set_title('GASF (R channel)')
    
    axes[3].imshow(gaf_rgb)
    axes[3].set_title('RGB GAF')
    
    plt.tight_layout()
    plt.savefig('gaf_test.png', dpi=150)
    print("Test image saved to gaf_test.png")
