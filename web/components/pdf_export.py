"""
PDF报告导出组件
PDF Report Export Component

生成因子分析报告的PDF

Author: VisionQuant Team
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from typing import Dict, List
import os


class PDFReportExporter:
    """
    PDF报告导出器
    
    功能：
    1. 生成因子分析报告PDF
    2. 包含图表和表格
    3. 支持中文字体
    """
    
    def __init__(self):
        """初始化PDF导出器"""
        self.styles = getSampleStyleSheet()
        self._setup_chinese_style()
    
    def _setup_chinese_style(self):
        """设置中文字体样式"""
        # 注意：需要系统中文字体支持
        try:
            self.chinese_style = ParagraphStyle(
                'ChineseStyle',
                parent=self.styles['Normal'],
                fontName='SimSun',  # 宋体
                fontSize=12,
                leading=14
            )
        except:
            # 如果中文字体不可用，使用默认样式
            self.chinese_style = self.styles['Normal']
    
    def export_factor_report(
        self,
        report_data: Dict,
        output_path: str
    ):
        """
        导出因子分析报告
        
        Args:
            report_data: 报告数据字典
            output_path: 输出路径
        """
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # 标题
        title = Paragraph("K线学习因子有效性分析报告", self.styles['Title'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # 基本信息
        story.append(Paragraph(f"生成时间: {report_data.get('generation_date', 'N/A')}", self.chinese_style))
        story.append(Paragraph(f"因子名称: {report_data.get('factor_name', 'KLine Visual Factor')}", self.chinese_style))
        story.append(Spacer(1, 0.2*inch))
        
        # IC/Sharpe分析
        if 'ic_analysis' in report_data:
            story.append(Paragraph("1. IC/Sharpe分析", self.styles['Heading1']))
            ic_data = report_data['ic_analysis']
            
            # 创建表格
            data = [
                ['指标', '数值'],
                ['平均IC', f"{ic_data.get('mean_ic', 0):.4f}"],
                ['IC标准差', f"{ic_data.get('std_ic', 0):.4f}"],
                ['ICIR', f"{ic_data.get('ic_ir', 0):.2f}"],
                ['正IC比例', f"{ic_data.get('positive_ic_ratio', 0)*100:.1f}%"]
            ]
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
            story.append(Spacer(1, 0.2*inch))
        
        # Regime分析
        if 'regime_analysis' in report_data:
            story.append(Paragraph("2. 市场Regime分析", self.styles['Heading1']))
            regime_data = report_data['regime_analysis']
            
            data = [
                ['Regime', '天数', '占比'],
                ['牛市', f"{regime_data.get('Bull_days', 0)}", f"{regime_data.get('Bull_percentage', 0):.1f}%"],
                ['熊市', f"{regime_data.get('Bear_days', 0)}", f"{regime_data.get('Bear_percentage', 0):.1f}%"],
                ['震荡', f"{regime_data.get('Oscillating_days', 0)}", f"{regime_data.get('Oscillating_percentage', 0):.1f}%"]
            ]
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
            story.append(Spacer(1, 0.2*inch))
        
        # 构建PDF
        doc.build(story)
        
        return output_path


if __name__ == "__main__":
    print("=== PDF报告导出器测试 ===")
    
    exporter = PDFReportExporter()
    
    # 示例数据
    report_data = {
        'generation_date': '2024-01-01',
        'factor_name': 'KLine Visual Factor',
        'ic_analysis': {
            'mean_ic': 0.05,
            'std_ic': 0.02,
            'ic_ir': 2.5,
            'positive_ic_ratio': 0.65
        },
        'regime_analysis': {
            'Bull_days': 100,
            'Bull_percentage': 50.0,
            'Bear_days': 30,
            'Bear_percentage': 15.0,
            'Oscillating_days': 70,
            'Oscillating_percentage': 35.0
        }
    }
    
    output_path = exporter.export_factor_report(report_data, 'test_report.pdf')
    print(f"✅ PDF报告已生成: {output_path}")
