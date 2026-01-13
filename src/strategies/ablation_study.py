"""
消融实验框架 (Ablation Study Framework)

用于系统性地评估VisionQuant各个组件的贡献：
1. Self-Attention模块的影响
2. 价格相关性过滤的影响
3. 时间隔离(NMS)的影响
4. 不同特征提取器的对比
5. 不同注意力头数的影响
6. 不同特征维度的对比

Author: Yisheng Pan
Date: 2026-01
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.models.attention_cae import AttentionCAE
from src.models.autoencoder import QuantCAE
from src.models.vision_engine import VisionEngine
from src.data.data_loader import DataLoader
from src.strategies.backtester import Backtester
from src.strategies.factor_mining import FactorMiner


class AblationStudy:
    """
    消融实验主类
    
    实验配置：
    - Baseline: 无Attention的CAE
    - w/o Attention: 移除Attention模块
    - w/o Correlation: 移除价格相关性过滤
    - w/o Time Isolation: 移除时间隔离
    - ResNet Features: 使用预训练ResNet特征
    - Different Heads: 不同注意力头数 (4, 8, 16)
    - Different Dims: 不同特征维度 (512, 1024, 2048)
    """
    
    def __init__(self, data_dir: str = None, model_dir: str = None):
        """
        Args:
            data_dir: 数据目录
            model_dir: 模型目录
        """
        self.data_dir = data_dir or os.path.join(PROJECT_ROOT, "data")
        self.model_dir = model_dir or os.path.join(self.data_dir, "models")
        
        self.loader = DataLoader()
        self.factor_miner = FactorMiner()
        
        # 实验配置
        self.configs = {
            'full_model': {
                'use_attention': True,
                'num_heads': 8,
                'latent_dim': 1024,
                'use_correlation': True,
                'correlation_threshold': 0.5,
                'use_time_isolation': True,
                'isolation_days': 20,
                'feature_extractor': 'attention_cae'
            },
            'w_o_attention': {
                'use_attention': False,
                'num_heads': 0,
                'latent_dim': 1024,
                'use_correlation': True,
                'correlation_threshold': 0.5,
                'use_time_isolation': True,
                'isolation_days': 20,
                'feature_extractor': 'cae'
            },
            'w_o_correlation': {
                'use_attention': True,
                'num_heads': 8,
                'latent_dim': 1024,
                'use_correlation': False,
                'correlation_threshold': 0.0,
                'use_time_isolation': True,
                'isolation_days': 20,
                'feature_extractor': 'attention_cae'
            },
            'w_o_time_isolation': {
                'use_attention': True,
                'num_heads': 8,
                'latent_dim': 1024,
                'use_correlation': True,
                'correlation_threshold': 0.5,
                'use_time_isolation': False,
                'isolation_days': 0,
                'feature_extractor': 'attention_cae'
            },
            'resnet_features': {
                'use_attention': False,
                'num_heads': 0,
                'latent_dim': 2048,
                'use_correlation': True,
                'correlation_threshold': 0.5,
                'use_time_isolation': True,
                'isolation_days': 20,
                'feature_extractor': 'resnet50'
            },
            'heads_4': {
                'use_attention': True,
                'num_heads': 4,
                'latent_dim': 1024,
                'use_correlation': True,
                'correlation_threshold': 0.5,
                'use_time_isolation': True,
                'isolation_days': 20,
                'feature_extractor': 'attention_cae'
            },
            'heads_16': {
                'use_attention': True,
                'num_heads': 16,
                'latent_dim': 1024,
                'use_correlation': True,
                'correlation_threshold': 0.5,
                'use_time_isolation': True,
                'isolation_days': 20,
                'feature_extractor': 'attention_cae'
            },
            'dim_512': {
                'use_attention': True,
                'num_heads': 8,
                'latent_dim': 512,
                'use_correlation': True,
                'correlation_threshold': 0.5,
                'use_time_isolation': True,
                'isolation_days': 20,
                'feature_extractor': 'attention_cae'
            },
            'dim_2048': {
                'use_attention': True,
                'num_heads': 8,
                'latent_dim': 2048,
                'use_correlation': True,
                'correlation_threshold': 0.5,
                'use_time_isolation': True,
                'isolation_days': 20,
                'feature_extractor': 'attention_cae'
            }
        }
    
    def load_model(self, config: Dict) -> torch.nn.Module:
        """
        根据配置加载模型
        
        Args:
            config: 实验配置字典
            
        Returns:
            加载的模型
        """
        feature_extractor = config['feature_extractor']
        
        if feature_extractor == 'attention_cae':
            model = AttentionCAE(
                latent_dim=config['latent_dim'],
                num_attention_heads=config['num_heads'],
                use_attention=config['use_attention']
            )
            model_path = os.path.join(self.model_dir, "attention_cae_best.pth")
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
            return model
        
        elif feature_extractor == 'cae':
            model = QuantCAE()
            model_path = os.path.join(self.model_dir, "cae_best.pth")
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
            return model
        
        elif feature_extractor == 'resnet50':
            import torchvision.models as models
            model = models.resnet50(pretrained=True)
            model.fc = torch.nn.Identity()  # 移除分类头
            return model
        
        else:
            raise ValueError(f"Unknown feature extractor: {feature_extractor}")
    
    def run_single_experiment(
        self, 
        config_name: str, 
        config: Dict,
        test_symbols: List[str],
        start_date: str = "2023-07-01",
        end_date: str = "2025-01-01"
    ) -> Dict:
        """
        运行单个实验配置
        
        Args:
            config_name: 配置名称
            config: 配置字典
            test_symbols: 测试股票列表
            start_date: 回测开始日期
            end_date: 回测结束日期
            
        Returns:
            实验结果字典
        """
        print(f"\n{'='*60}")
        print(f"🔬 运行实验: {config_name}")
        print(f"{'='*60}")
        print(f"配置: {json.dumps(config, indent=2)}")
        
        # 加载模型
        model = self.load_model(config)
        model.eval()
        
        # 创建VisionEngine（需要根据配置修改）
        # 这里简化处理，实际需要修改VisionEngine以支持不同配置
        vision_engine = VisionEngine()
        
        # 运行回测
        results = []
        for symbol in test_symbols[:10]:  # 先用10只股票测试
            try:
                # 获取数据
                df = self.loader.get_stock_data(symbol)
                if df.empty:
                    continue
                
                # 运行策略（简化版，实际需要完整回测逻辑）
                # 这里只是示例，实际需要调用完整的回测框架
                result = self._run_backtest_single_stock(
                    symbol, df, vision_engine, config, start_date, end_date
                )
                results.append(result)
                
            except Exception as e:
                print(f"⚠️ 股票 {symbol} 回测失败: {e}")
                continue
        
        # 汇总结果
        if not results:
            return None
        
        summary = self._summarize_results(results, config_name)
        return summary
    
    def _run_backtest_single_stock(
        self,
        symbol: str,
        df: pd.DataFrame,
        vision_engine: VisionEngine,
        config: Dict,
        start_date: str,
        end_date: str
    ) -> Dict:
        """
        单只股票的回测（简化版）
        
        实际实现需要：
        1. 根据config修改vision_engine的搜索逻辑
        2. 应用相关性过滤和时间隔离
        3. 计算收益
        """
        # 这里是占位符，实际需要完整的回测逻辑
        return {
            'symbol': symbol,
            'return': np.random.uniform(-0.2, 0.4),  # 占位符
            'sharpe': np.random.uniform(0.5, 2.0),  # 占位符
            'max_drawdown': np.random.uniform(-0.3, -0.1),  # 占位符
            'win_rate': np.random.uniform(0.4, 0.7)  # 占位符
        }
    
    def _summarize_results(self, results: List[Dict], config_name: str) -> Dict:
        """
        汇总实验结果
        """
        returns = [r['return'] for r in results]
        sharpes = [r['sharpe'] for r in results]
        max_dds = [r['max_drawdown'] for r in results]
        win_rates = [r['win_rate'] for r in results]
        
        return {
            'config_name': config_name,
            'num_stocks': len(results),
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'avg_sharpe': np.mean(sharpes),
            'avg_max_dd': np.mean(max_dds),
            'avg_win_rate': np.mean(win_rates),
            'returns': returns,
            'sharpes': sharpes
        }
    
    def run_all_experiments(
        self,
        test_symbols: List[str] = None,
        start_date: str = "2023-07-01",
        end_date: str = "2025-01-01"
    ) -> pd.DataFrame:
        """
        运行所有消融实验
        
        Args:
            test_symbols: 测试股票列表（None则使用默认）
            start_date: 回测开始日期
            end_date: 回测结束日期
            
        Returns:
            结果DataFrame
        """
        if test_symbols is None:
            # 默认测试股票列表
            test_symbols = [
                '000001', '000002', '600000', '600036', '600519',
                '600887', '000858', '002415', '300059', '601318'
            ]
        
        all_results = []
        
        for config_name, config in self.configs.items():
            result = self.run_single_experiment(
                config_name, config, test_symbols, start_date, end_date
            )
            if result:
                all_results.append(result)
        
        # 转换为DataFrame
        df_results = pd.DataFrame(all_results)
        
        # 计算相对于Full Model的差异
        if 'full_model' in df_results['config_name'].values:
            full_model_metrics = df_results[df_results['config_name'] == 'full_model'].iloc[0]
            df_results['delta_return'] = df_results['avg_return'] - full_model_metrics['avg_return']
            df_results['delta_sharpe'] = df_results['avg_sharpe'] - full_model_metrics['avg_sharpe']
        
        return df_results
    
    def generate_latex_table(self, df_results: pd.DataFrame, output_path: str = None) -> str:
        """
        生成LaTeX格式的结果表格
        
        Args:
            df_results: 实验结果DataFrame
            output_path: 输出路径（可选）
            
        Returns:
            LaTeX表格字符串
        """
        # 重命名配置名称以便显示
        name_map = {
            'full_model': 'Full Model (VQ)',
            'w_o_attention': 'w/o Attention',
            'w_o_correlation': 'w/o Correlation',
            'w_o_time_isolation': 'w/o Time Isolation',
            'resnet_features': 'ResNet Features',
            'heads_4': 'Heads=4',
            'heads_16': 'Heads=16',
            'dim_512': 'Dim=512',
            'dim_2048': 'Dim=2048'
        }
        
        df_display = df_results.copy()
        df_display['config_name'] = df_display['config_name'].map(name_map)
        
        latex = "\\begin{table}[t]\n"
        latex += "\\centering\n"
        latex += "\\caption{Ablation Study Results}\n"
        latex += "\\label{tab:ablation}\n"
        latex += "\\begin{tabular}{lcccc}\n"
        latex += "\\toprule\n"
        latex += "Configuration & Return & Alpha & Sharpe & $\\Delta$Alpha \\\\\n"
        latex += "\\midrule\n"
        
        for _, row in df_display.iterrows():
            config = row['config_name']
            ret = row['avg_return'] * 100
            alpha = row.get('delta_return', 0) * 100
            sharpe = row['avg_sharpe']
            
            latex += f"{config} & {ret:.1f}\\% & {alpha:+.1f}\\% & {sharpe:.2f} & {alpha:+.1f}\\% \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(latex)
        
        return latex


def main():
    """
    主函数：运行消融实验
    """
    print("="*60)
    print("🔬 VisionQuant 消融实验")
    print("="*60)
    
    study = AblationStudy()
    
    # 运行所有实验
    df_results = study.run_all_experiments()
    
    # 保存结果
    output_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "ablation_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n✅ 结果已保存: {csv_path}")
    
    # 生成LaTeX表格
    latex_path = os.path.join(output_dir, "ablation_table.tex")
    latex_table = study.generate_latex_table(df_results, latex_path)
    print(f"✅ LaTeX表格已保存: {latex_path}")
    
    # 打印结果
    print("\n" + "="*60)
    print("📊 实验结果汇总")
    print("="*60)
    print(df_results[['config_name', 'avg_return', 'avg_sharpe', 'delta_return']].to_string())
    
    return df_results


if __name__ == "__main__":
    main()
