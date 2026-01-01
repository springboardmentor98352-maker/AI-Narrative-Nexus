"""
Report Generation Module for NarrativeNexus

This module provides comprehensive report generation capabilities:
- HTML reports with embedded visualizations
- PDF reports using ReportLab or WeasyPrint
- Export to various formats (CSV, JSON, Markdown)
- Customizable templates and styling
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import os
import base64
from io import BytesIO

# Try importing report generation libraries
REPORTLAB_AVAILABLE = False
WEASYPRINT_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    pass

try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    pass


class ReportGenerator:
    """Generate comprehensive analysis reports in various formats."""
    
    def __init__(self, project_name: str = "NarrativeNexus Analysis"):
        """Initialize report generator.
        
        Args:
            project_name: Name of the project/analysis
        """
        self.project_name = project_name
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_html_report(self,
                            topics: List[Tuple[int, List[Tuple[str, float]]]],
                            sentiment_data: Optional[Dict[int, Dict[str, any]]] = None,
                            insights: Optional[List[Dict[str, str]]] = None,
                            summary: Optional[str] = None,
                            stats: Optional[Dict[str, any]] = None) -> str:
        """Generate HTML report.
        
        Args:
            topics: List of topics with words
            sentiment_data: Sentiment analysis results
            insights: Generated insights
            summary: Executive summary
            stats: Statistics dictionary
            
        Returns:
            HTML string
        """
        html_parts = []
        
        # HTML header with styling
        html_parts.append("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project_name} - Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #764ba2;
            margin-top: 30px;
        }}
        h3 {{
            color: #555;
        }}
        .meta {{
            color: #777;
            font-size: 0.9em;
            margin-bottom: 30px;
        }}
        .topic-card {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        .topic-title {{
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }}
        .word-list {{
            color: #555;
        }}
        .sentiment-positive {{
            color: #28a745;
            font-weight: bold;
        }}
        .sentiment-negative {{
            color: #dc3545;
            font-weight: bold;
        }}
        .sentiment-neutral {{
            color: #6c757d;
            font-weight: bold;
        }}
        .insight-card {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        .insight-high {{
            background: #f8d7da;
            border-left-color: #dc3545;
        }}
        .insight-opportunity {{
            background: #d4edda;
            border-left-color: #28a745;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{project_name}</h1>
        <div class="meta">Generated on: {timestamp}</div>
""".format(project_name=self.project_name, timestamp=self.timestamp))
        
        # Executive Summary
        if summary:
            html_parts.append(f"""
        <h2>Executive Summary</h2>
        <div class="summary">
            {self._markdown_to_html(summary)}
        </div>
""")
        
        # Statistics
        if stats:
            html_parts.append("""
        <h2>Analysis Statistics</h2>
        <div class="stats-grid">
""")
            for key, value in stats.items():
                label = key.replace('_', ' ').title()
                html_parts.append(f"""
            <div class="stat-card">
                <div class="stat-value">{value:,}</div>
                <div class="stat-label">{label}</div>
            </div>
""")
            html_parts.append("        </div>\n")
        
        # Topics
        html_parts.append("""
        <h2>Discovered Topics</h2>
""")
        
        for topic_id, words in topics:
            word_list = ", ".join([f"{word} ({prob:.3f})" for word, prob in words[:10]])
            
            sentiment_info = ""
            if sentiment_data and topic_id in sentiment_data:
                sent = sentiment_data[topic_id]
                pos_pct = sent.get("positive_pct", 0)
                neg_pct = sent.get("negative_pct", 0)
                neu_pct = sent.get("neutral_pct", 0)
                
                dominant = "positive" if pos_pct > max(neg_pct, neu_pct) else ("negative" if neg_pct > neu_pct else "neutral")
                sentiment_class = f"sentiment-{dominant}"
                
                sentiment_info = f"""
                <div style="margin-top: 10px;">
                    <strong>Sentiment:</strong> 
                    <span class="{sentiment_class}">{dominant.title()}</span> 
                    (Pos: {pos_pct:.1f}%, Neg: {neg_pct:.1f}%, Neu: {neu_pct:.1f}%)
                </div>
"""
            
            html_parts.append(f"""
        <div class="topic-card">
            <div class="topic-title">Topic {topic_id + 1}</div>
            <div class="word-list">{word_list}</div>
            {sentiment_info}
        </div>
""")
        
        # Insights
        if insights:
            html_parts.append("""
        <h2>Key Insights & Recommendations</h2>
""")
            for insight in insights:
                insight_class = "insight-card"
                if insight.get("priority") == "high":
                    insight_class += " insight-high"
                elif insight.get("type") == "opportunity":
                    insight_class += " insight-opportunity"
                
                html_parts.append(f"""
        <div class="{insight_class}">
            <strong>{insight.get('message', '')}</strong>
            <div style="margin-top: 5px; font-size: 0.9em;">
                <em>Recommendation:</em> {insight.get('recommendation', '')}
            </div>
        </div>
""")
        
        # Close HTML
        html_parts.append("""
    </div>
</body>
</html>
""")
        
        return "".join(html_parts)
    
    def _markdown_to_html(self, markdown_text: str) -> str:
        """Convert simple markdown to HTML."""
        html = markdown_text
        
        # Headers
        html = html.replace("### ", "<h3>").replace("\n", "</h3>\n", html.count("### "))
        html = html.replace("## ", "<h2>").replace("\n", "</h2>\n", html.count("## "))
        
        # Bold
        import re
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
        
        # Lists
        lines = html.split('\n')
        in_list = False
        new_lines = []
        
        for line in lines:
            if line.strip().startswith('- '):
                if not in_list:
                    new_lines.append('<ul>')
                    in_list = True
                new_lines.append(f'<li>{line.strip()[2:]}</li>')
            else:
                if in_list:
                    new_lines.append('</ul>')
                    in_list = False
                new_lines.append(line)
        
        if in_list:
            new_lines.append('</ul>')
        
        html = '\n'.join(new_lines)
        
        # Paragraphs
        html = '<p>' + html.replace('\n\n', '</p><p>') + '</p>'
        
        return html
    
    def generate_pdf_report(self,
                           topics: List[Tuple[int, List[Tuple[str, float]]]],
                           sentiment_data: Optional[Dict[int, Dict[str, any]]] = None,
                           insights: Optional[List[Dict[str, str]]] = None,
                           summary: Optional[str] = None,
                           stats: Optional[Dict[str, any]] = None,
                           output_path: str = "report.pdf") -> str:
        """Generate PDF report using ReportLab.
        
        Args:
            topics: Topics data
            sentiment_data: Sentiment data
            insights: Insights data
            summary: Executive summary
            stats: Statistics
            output_path: Output file path
            
        Returns:
            Path to generated PDF
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab not available. Install with: pip install reportlab")
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#764ba2'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        story.append(Paragraph(self.project_name, title_style))
        story.append(Paragraph(f"Generated: {self.timestamp}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        if summary:
            story.append(Paragraph("Executive Summary", heading_style))
            for line in summary.split('\n'):
                if line.strip():
                    story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Statistics
        if stats:
            story.append(Paragraph("Analysis Statistics", heading_style))
            stat_data = [['Metric', 'Value']]
            for key, value in stats.items():
                label = key.replace('_', ' ').title()
                stat_data.append([label, str(value)])
            
            stat_table = Table(stat_data, colWidths=[3*inch, 2*inch])
            stat_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(stat_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Topics
        story.append(Paragraph("Discovered Topics", heading_style))
        
        for topic_id, words in topics:
            word_list = ", ".join([f"{word} ({prob:.3f})" for word, prob in words[:10]])
            
            story.append(Paragraph(f"<b>Topic {topic_id + 1}</b>", styles['Normal']))
            story.append(Paragraph(word_list, styles['Normal']))
            
            if sentiment_data and topic_id in sentiment_data:
                sent = sentiment_data[topic_id]
                pos_pct = sent.get("positive_pct", 0)
                neg_pct = sent.get("negative_pct", 0)
                neu_pct = sent.get("neutral_pct", 0)
                
                sentiment_text = f"Sentiment: Pos: {pos_pct:.1f}%, Neg: {neg_pct:.1f}%, Neu: {neu_pct:.1f}%"
                story.append(Paragraph(sentiment_text, styles['Italic']))
            
            story.append(Spacer(1, 0.1*inch))
        
        # Insights
        if insights:
            story.append(Paragraph("Key Insights & Recommendations", heading_style))
            
            for i, insight in enumerate(insights, 1):
                story.append(Paragraph(f"<b>{i}. {insight.get('message', '')}</b>", styles['Normal']))
                story.append(Paragraph(f"<i>Recommendation: {insight.get('recommendation', '')}</i>", styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
        
        # Build PDF
        doc.build(story)
        return output_path
    
    def export_to_json(self,
                      topics: List[Tuple[int, List[Tuple[str, float]]]],
                      sentiment_data: Optional[Dict[int, Dict[str, any]]] = None,
                      insights: Optional[List[Dict[str, str]]] = None,
                      stats: Optional[Dict[str, any]] = None,
                      output_path: str = "report.json") -> str:
        """Export analysis to JSON format.
        
        Args:
            topics: Topics data
            sentiment_data: Sentiment data
            insights: Insights
            stats: Statistics
            output_path: Output path
            
        Returns:
            Path to JSON file
        """
        data = {
            "project_name": self.project_name,
            "timestamp": self.timestamp,
            "stats": stats or {},
            "topics": [
                {
                    "topic_id": topic_id,
                    "words": [{"word": w, "score": float(s)} for w, s in words]
                }
                for topic_id, words in topics
            ],
            "sentiment": sentiment_data or {},
            "insights": insights or []
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def export_to_csv(self,
                     topics: List[Tuple[int, List[Tuple[str, float]]]],
                     output_path: str = "topics.csv") -> str:
        """Export topics to CSV format.
        
        Args:
            topics: Topics data
            output_path: Output path
            
        Returns:
            Path to CSV file
        """
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Topic ID', 'Word', 'Score'])
            
            for topic_id, words in topics:
                for word, score in words:
                    writer.writerow([topic_id + 1, word, f"{score:.6f}"])
        
        return output_path


def generate_quick_html_report(topics: List[Tuple[int, List[Tuple[str, float]]]],
                               output_path: str = "report.html") -> str:
    """Quick HTML report generation.
    
    Args:
        topics: Topics to include
        output_path: Output file path
        
    Returns:
        Path to generated file
    """
    generator = ReportGenerator()
    html = generator.generate_html_report(topics)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path


__all__ = [
    "ReportGenerator",
    "generate_quick_html_report",
    "REPORTLAB_AVAILABLE",
    "WEASYPRINT_AVAILABLE"
]
