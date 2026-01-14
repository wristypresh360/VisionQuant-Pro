"""
因子分析主页面
Factor Analysis Main Page

Streamlit页面：展示因子有效性分析结果

Author: VisionQuant Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import sys
import os

# 添加项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.factor_analysis.ic_analysis import ICAnalyzer
from src.factor_analysis.regime_detector import RegimeDetector
from src.factor_analysis.decay_analysis import DecayAnalyzer
from src.factor_analysis.crowding_detector import CrowdingDetector
from src.factor_analysis.risk_compensation import RiskCompensationAnalyzer
from src.factor_analysis.industry_stratification import IndustryStratifier
from src.factor_analysis.report_generator import ReportGenerator


def main():
    """因子分析主页面"""
    st.set_page_config(page_title="因子分析", page_icon="📊", layout="wide")
    
    st.title("📊 K线学习因子有效性分析")
    st.markdown("---")
    
    # 侧边栏：数据选择
    with st.sidebar:
        st.header("数据选择")
        
        # 股票选择
        symbol = st.text_input("股票代码", value="600519", help="输入6位股票代码")
        
        # 时间范围
        start_date = st.date_input("开始日期", value=pd.to_datetime('2020-01-01'))
        end_date = st.date_input("结束日期", value=pd.to_datetime('2024-12-31'))
        
        # 分析选项
        st.header("分析选项")
        show_ic = st.checkbox("IC/Sharpe分析", value=True)
        show_regime = st.checkbox("Regime识别", value=True)
        show_decay = st.checkbox("因子衰减", value=True)
        show_crowding = st.checkbox("拥挤交易", value=True)
        show_risk = st.checkbox("风险补偿", value=True)
        show_industry = st.checkbox("行业分层", value=True)
        
        # 生成报告按钮
        if st.button("生成完整报告", type="primary"):
            st.session_state['generate_report'] = True
    
    # 主内容区
    if st.button("开始分析", type="primary"):
        with st.spinner("正在分析..."):
            # 加载数据（这里简化，实际应从数据加载器获取）
            # 假设已有因子值和收益率数据
            st.warning("⚠️ 需要实现数据加载逻辑")
            
            # 示例：IC/Sharpe曲线图
            if show_ic:
                st.subheader("📈 IC/Sharpe曲线分析")
                plot_ic_sharpe_curves()
            
            # Regime识别图
            if show_regime:
                st.subheader("🌍 市场Regime识别")
                plot_regime_chart()
            
            # 因子衰减曲线
            if show_decay:
                st.subheader("📉 因子衰减分析")
                plot_decay_curve()
            
            # 拥挤交易热力图
            if show_crowding:
                st.subheader("🔥 拥挤交易检测")
                plot_crowding_heatmap()
            
            # 风险补偿散点图
            if show_risk:
                st.subheader("⚖️ 风险补偿分析")
                plot_risk_scatter()
            
            # 行业IC对比表
            if show_industry:
                st.subheader("🏢 行业IC对比")
                plot_industry_ic_table()


def plot_ic_sharpe_curves():
    """绘制IC/Sharpe曲线"""
    # 示例数据
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    ic_series = pd.Series(np.random.randn(200) * 0.05, index=dates)
    sharpe_series = pd.Series(np.random.randn(200) * 0.5 + 1.0, index=dates)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Rolling IC', 'Rolling Sharpe Ratio'),
        vertical_spacing=0.1
    )
    
    # IC曲线
    fig.add_trace(
        go.Scatter(
            x=ic_series.index,
            y=ic_series.values,
            mode='lines',
            name='Rolling IC',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # IC阈值线
    fig.add_hline(y=0.05, line_dash="dash", line_color="green", 
                  annotation_text="IC阈值(0.05)", row=1, col=1)
    fig.add_hline(y=-0.05, line_dash="dash", line_color="red", row=1, col=1)
    
    # Sharpe曲线
    fig.add_trace(
        go.Scatter(
            x=sharpe_series.index,
            y=sharpe_series.values,
            mode='lines',
            name='Rolling Sharpe',
            line=dict(color='orange', width=2)
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=True, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)


def plot_regime_chart():
    """绘制Regime识别图"""
    # 示例数据
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    regimes = np.random.choice(['Bull', 'Bear', 'Oscillating'], 200)
    
    fig = go.Figure()
    
    # 为每个regime添加区域
    colors = {'Bull': 'green', 'Bear': 'red', 'Oscillating': 'yellow'}
    for regime_type in ['Bull', 'Bear', 'Oscillating']:
        mask = regimes == regime_type
        if mask.any():
            fig.add_trace(go.Scatter(
                x=dates[mask],
                y=[regime_type] * mask.sum(),
                mode='markers',
                name=regime_type,
                marker=dict(color=colors[regime_type], size=10)
            ))
    
    fig.update_layout(
        title="Market Regime Timeline",
        xaxis_title="Date",
        yaxis_title="Regime",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_decay_curve():
    """绘制因子衰减曲线"""
    forward_days = [1, 5, 10, 20, 60, 120]
    ic_values = [0.08, 0.06, 0.04, 0.02, 0.01, 0.005]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forward_days,
        y=ic_values,
        mode='lines+markers',
        name='IC Decay',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Factor IC Decay Curve",
        xaxis_title="Forward Days",
        yaxis_title="IC Value",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_crowding_heatmap():
    """绘制拥挤交易热力图"""
    # 示例数据
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    stocks = [f'Stock_{i}' for i in range(20)]
    hhi_values = np.random.uniform(0.1, 0.3, (50, 20))
    
    fig = go.Figure(data=go.Heatmap(
        z=hhi_values,
        x=stocks,
        y=dates.strftime('%Y-%m-%d'),
        colorscale='RdYlGn',
        colorbar=dict(title="HHI")
    ))
    
    fig.update_layout(
        title="Crowding Trade Heatmap (HHI)",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_risk_scatter():
    """绘制风险补偿散点图"""
    # 示例数据
    returns = np.random.uniform(0.05, 0.25, 10)
    volatilities = np.random.uniform(0.15, 0.35, 10)
    industries = np.random.choice(['银行', '科技', '消费'], 10)
    
    fig = go.Figure()
    
    for industry in ['银行', '科技', '消费']:
        mask = industries == industry
        fig.add_trace(go.Scatter(
            x=volatilities[mask],
            y=returns[mask],
            mode='markers',
            name=industry,
            marker=dict(size=10)
        ))
    
    fig.update_layout(
        title="Risk-Return Scatter Plot",
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_industry_ic_table():
    """绘制行业IC对比表"""
    # 示例数据
    industries = ['银行', '地产', '科技', '消费', '医药']
    mean_ics = [0.05, 0.03, 0.08, 0.04, 0.06]
    ic_irs = [1.2, 0.8, 1.5, 1.0, 1.3]
    
    df = pd.DataFrame({
        '行业': industries,
        '平均IC': mean_ics,
        'ICIR': ic_irs
    })
    df = df.sort_values('平均IC', ascending=False)
    
    st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
