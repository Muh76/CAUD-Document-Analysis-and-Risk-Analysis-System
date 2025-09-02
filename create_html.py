#!/usr/bin/env python3
"""
Script to convert markdown methodology document to HTML
"""

import markdown
import os

def create_html_from_markdown():
    # Read the markdown file
    with open('Contract_Review_Risk_Analysis_Methodology.md', 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=['toc', 'tables', 'fenced_code', 'codehilite'])
    
    # Create full HTML document
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Contract Review & Risk Analysis System - Methodology</title>
        <style>
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                font-family: 'Arial', sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
                font-size: 12px;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                font-size: 24px;
                page-break-after: avoid;
            }}
            h2 {{
                color: #34495e;
                border-bottom: 2px solid #bdc3c7;
                padding-bottom: 5px;
                font-size: 18px;
                page-break-after: avoid;
            }}
            h3 {{
                color: #7f8c8d;
                font-size: 14px;
                page-break-after: avoid;
            }}
            h4 {{
                color: #7f8c8d;
                font-size: 12px;
                page-break-after: avoid;
            }}
            code {{
                background-color: #f8f9fa;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }}
            pre {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                border-left: 4px solid #3498db;
                font-size: 10px;
                page-break-inside: avoid;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                font-size: 10px;
                page-break-inside: avoid;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .checklist {{
                background-color: #e8f5e8;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #27ae60;
                margin: 10px 0;
            }}
            .warning {{
                background-color: #fff3cd;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #ffc107;
                margin: 10px 0;
            }}
            .info {{
                background-color: #d1ecf1;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #17a2b8;
                margin: 10px 0;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid #3498db;
            }}
            .author-info {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                font-size: 11px;
            }}
            .toc {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
                page-break-after: avoid;
            }}
            .toc ul {{
                list-style-type: none;
                padding-left: 20px;
            }}
            .toc li {{
                margin: 5px 0;
            }}
            .toc a {{
                text-decoration: none;
                color: #3498db;
            }}
            .progress-tracker {{
                background-color: #e8f5e8;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
                page-break-inside: avoid;
            }}
            .progress-tracker h3 {{
                color: #27ae60;
                margin-top: 0;
            }}
            .progress-tracker ul {{
                list-style-type: none;
                padding-left: 0;
            }}
            .progress-tracker li {{
                margin: 5px 0;
                padding: 5px;
                background-color: #f8f9fa;
                border-radius: 3px;
            }}
            .progress-tracker li:before {{
                content: "☐ ";
                color: #7f8c8d;
            }}
            .progress-tracker li.completed:before {{
                content: "☑ ";
                color: #27ae60;
            }}
            @media print {{
                body {{
                    font-size: 10px;
                }}
                h1 {{
                    font-size: 20px;
                }}
                h2 {{
                    font-size: 16px;
                }}
                h3 {{
                    font-size: 12px;
                }}
                pre, code {{
                    font-size: 8px;
                }}
                table {{
                    font-size: 8px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Contract Review & Risk Analysis System</h1>
            <h2>Comprehensive Methodology & Implementation Guide</h2>
        </div>
        
        <div class="author-info">
            <strong>Author:</strong> Mohammad Babaie<br>
            <strong>Email:</strong> mj.babaie@gmail.com<br>
            <strong>LinkedIn:</strong> <a href="https://www.linkedin.com/in/mohammadbabaie/">https://www.linkedin.com/in/mohammadbabaie/</a><br>
            <strong>GitHub:</strong> <a href="https://github.com/Muh76">https://github.com/Muh76</a>
        </div>
        
        {html_content}
        
        <div class="progress-tracker">
            <h3>📋 Project Progress Tracker</h3>
            <p><strong>Last Updated:</strong> {os.popen('date').read().strip()}</p>
            <p><strong>Next Review:</strong> Weekly</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open('Contract_Review_Risk_Analysis_Methodology.html', 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print("✅ HTML document created successfully: Contract_Review_Risk_Analysis_Methodology.html")
    print("📄 To convert to PDF:")
    print("   1. Open the HTML file in your browser")
    print("   2. Press Ctrl+P (or Cmd+P on Mac)")
    print("   3. Select 'Save as PDF'")
    print("   4. Choose your desired settings and save")

if __name__ == "__main__":
    create_html_from_markdown()
