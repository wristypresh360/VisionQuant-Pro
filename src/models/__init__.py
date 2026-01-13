"""
模型模块
Deep learning models for visual pattern recognition
"""

from .autoencoder import QuantCAE
from .vision_engine import VisionEngine

# PredictEngine 不存在，使用 IndustrialPredictorReduced 作为替代
try:
    from .predict_engine import IndustrialPredictorReduced as PredictEngine
except ImportError:
    PredictEngine = None

__all__ = ['QuantCAE', 'VisionEngine']
if PredictEngine:
    __all__.append('PredictEngine')
