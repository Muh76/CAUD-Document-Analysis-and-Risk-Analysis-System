#!/usr/bin/env python3
"""
Script to convert markdown methodology document to PDF
"""

import markdown
from weasyprint import HTML, CSS
import os

def create_pdf_from_markdown():
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
            body {{
                font-family: 'Arial', sans-serif;
                line-height: 1.6;
                margin: 40px;
                color: #333;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                border-bottom: 2px solid #bdc3c7;
                padding-bottom: 5px;
            }}
            h3 {{
                color: #7f8c8d;
            }}
            code {{
                background-color: #f8f9fa;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            pre {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                border-left: 4px solid #3498db;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
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
            }}
            .warning {{
                background-color: #fff3cd;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #ffc107;
            }}
            .info {{
                background-color: #d1ecf1;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #17a2b8;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Write HTML to temporary file
    with open('temp_methodology.html', 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    try:
        # Convert HTML to PDF
        HTML('temp_methodology.html').write_pdf('Contract_Review_Risk_Analysis_Methodology.pdf')
        print("✅ PDF created successfully: Contract_Review_Risk_Analysis_Methodology.pdf")
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        print("Creating HTML version instead...")
        os.rename('temp_methodology.html', 'Contract_Review_Risk_Analysis_Methodology.html')
        print("✅ HTML version created: Contract_Review_Risk_Analysis_Methodology.html")
    finally:
        # Clean up temporary file
        if os.path.exists('temp_methodology.html'):
            os.remove('temp_methodology.html')

if __name__ == "__main__":
    create_pdf_from_markdown()
