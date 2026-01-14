"""
股票分层逻辑
Stock Stratification Logic

按市值/行业对股票进行分层

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class StratificationConfig:
    """分层配置"""
    market_cap_bins: int = 3          # 市值分层数（大/中/小盘）
    market_cap_labels: List[str] = None  # 分层标签
    min_stocks_per_stratum: int = 10  # 每层最少股票数


class StockStratifier:
    """
    股票分层器
    
    功能：
    1. 按市值分层（大/中/小盘）
    2. 按行业分层
    3. 组合分层（市值×行业）
    """
    
    def __init__(self, config: StratificationConfig = None):
        """
        初始化股票分层器
        
        Args:
            config: 分层配置
        """
        self.config = config or StratificationConfig()
        
        if self.config.market_cap_labels is None:
            self.config.market_cap_labels = ['small', 'medium', 'large']
    
    def stratify_by_market_cap(
        self,
        stocks_df: pd.DataFrame,
        market_cap_col: str = 'market_cap'
    ) -> pd.DataFrame:
        """
        按市值分层
        
        Args:
            stocks_df: 股票DataFrame，必须包含market_cap列
            market_cap_col: 市值列名
            
        Returns:
            添加了market_cap_rank列的DataFrame
        """
        if market_cap_col not in stocks_df.columns:
            raise ValueError(f"DataFrame缺少市值列: {market_cap_col}")
        
        df = stocks_df.copy()
        
        # 去除缺失值
        valid_df = df[df[market_cap_col].notna()].copy()
        
        if len(valid_df) < self.config.market_cap_bins:
            # 如果股票数太少，不分层
            df['market_cap_rank'] = 'medium'
            return df
        
        # 分位数分层
        try:
            valid_df['market_cap_rank'] = pd.qcut(
                valid_df[market_cap_col],
                q=self.config.market_cap_bins,
                labels=self.config.market_cap_labels,
                duplicates='drop'
            )
        except:
            # 如果分位数失败，使用等距分层
            bins = np.linspace(
                valid_df[market_cap_col].min(),
                valid_df[market_cap_col].max(),
                self.config.market_cap_bins + 1
            )
            valid_df['market_cap_rank'] = pd.cut(
                valid_df[market_cap_col],
                bins=bins,
                labels=self.config.market_cap_labels,
                include_lowest=True
            )
        
        # 合并回原DataFrame
        df = df.merge(
            valid_df[['market_cap_rank']],
            left_index=True,
            right_index=True,
            how='left'
        )
        df['market_cap_rank'] = df['market_cap_rank'].fillna('medium')
        
        return df
    
    def stratify_by_industry(
        self,
        stocks_df: pd.DataFrame,
        industry_col: str = 'industry',
        industry_mapping: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        按行业分层
        
        Args:
            stocks_df: 股票DataFrame
            industry_col: 行业列名
            industry_mapping: 行业映射（将细分行业映射到行业组）
            
        Returns:
            添加了industry_group列的DataFrame
        """
        df = stocks_df.copy()
        
        if industry_col not in df.columns:
            # 如果没有行业列，尝试从其他来源获取
            df[industry_col] = 'Unknown'
        
        if industry_mapping:
            # 应用行业映射
            df['industry_group'] = df[industry_col].map(industry_mapping).fillna('Other')
        else:
            # 直接使用原行业
            df['industry_group'] = df[industry_col]
        
        return df
    
    def stratify_combined(
        self,
        stocks_df: pd.DataFrame,
        market_cap_col: str = 'market_cap',
        industry_col: str = 'industry',
        industry_mapping: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        组合分层（市值×行业）
        
        Args:
            stocks_df: 股票DataFrame
            market_cap_col: 市值列名
            industry_col: 行业列名
            industry_mapping: 行业映射
            
        Returns:
            添加了分层信息的DataFrame
        """
        # 1. 按市值分层
        df = self.stratify_by_market_cap(stocks_df, market_cap_col)
        
        # 2. 按行业分层
        df = self.stratify_by_industry(df, industry_col, industry_mapping)
        
        # 3. 创建组合分层标识
        df['stratum'] = df['market_cap_rank'].astype(str) + '_' + df['industry_group'].astype(str)
        
        return df
    
    def get_stratum_list(self, stratified_df: pd.DataFrame) -> List[str]:
        """
        获取所有分层列表
        
        Args:
            stratified_df: 已分层的DataFrame
            
        Returns:
            分层标识列表
        """
        if 'stratum' in stratified_df.columns:
            return stratified_df['stratum'].unique().tolist()
        else:
            # 如果没有stratum列，组合market_cap_rank和industry_group
            if 'market_cap_rank' in stratified_df.columns and 'industry_group' in stratified_df.columns:
                return (stratified_df['market_cap_rank'].astype(str) + '_' + 
                       stratified_df['industry_group'].astype(str)).unique().tolist()
            return []
    
    def get_stratum_stocks(
        self,
        stratified_df: pd.DataFrame,
        stratum: str
    ) -> pd.DataFrame:
        """
        获取指定分层的股票
        
        Args:
            stratified_df: 已分层的DataFrame
            stratum: 分层标识（如'small_银行'）
            
        Returns:
            该分层的股票DataFrame
        """
        if 'stratum' in stratified_df.columns:
            return stratified_df[stratified_df['stratum'] == stratum].copy()
        else:
            # 解析stratum
            parts = stratum.split('_', 1)
            if len(parts) == 2:
                market_cap, industry = parts
                mask = (
                    (stratified_df['market_cap_rank'] == market_cap) &
                    (stratified_df['industry_group'] == industry)
                )
                return stratified_df[mask].copy()
            return pd.DataFrame()


if __name__ == "__main__":
    print("=== 股票分层器测试 ===")
    
    # 模拟数据
    np.random.seed(42)
    stocks = pd.DataFrame({
        'symbol': [f'Stock_{i}' for i in range(100)],
        'market_cap': np.random.uniform(10, 1000, 100),  # 10-1000亿
        'industry': np.random.choice(['银行', '地产', '科技', '消费'], 100)
    })
    
    stratifier = StockStratifier()
    stratified = stratifier.stratify_combined(stocks)
    
    print(f"\n分层结果:")
    print(stratified[['symbol', 'market_cap_rank', 'industry_group', 'stratum']].head(10))
    
    # 获取分层列表
    strata = stratifier.get_stratum_list(stratified)
    print(f"\n总分层数: {len(strata)}")
    print(f"分层列表: {strata[:5]}...")
