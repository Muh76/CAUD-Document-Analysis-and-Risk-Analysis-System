#!/usr/bin/env python3
"""
Script to create PDF directly from markdown content using reportlab
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import re

def create_pdf_from_markdown():
    # Read the markdown file
    with open('Contract_Review_Risk_Analysis_Methodology.md', 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Create PDF document
    doc = SimpleDocTemplate("Contract_Review_Risk_Analysis_Methodology.pdf", pagesize=A4)
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=18,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=12,
        textColor=colors.darkblue
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10,
        spaceBefore=10,
        textColor=colors.darkblue
    )
    
    heading3_style = ParagraphStyle(
        'CustomHeading3',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=8,
        spaceBefore=8,
        textColor=colors.darkblue
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    )
    
    code_style = ParagraphStyle(
        'CustomCode',
        parent=styles['Normal'],
        fontSize=8,
        fontName='Courier',
        leftIndent=20,
        rightIndent=20,
        spaceAfter=6,
        backColor=colors.lightgrey
    )
    
    # Add title
    story.append(Paragraph("Contract Review & Risk Analysis System", title_style))
    story.append(Paragraph("Comprehensive Methodology & Implementation Guide", subtitle_style))
    story.append(Spacer(1, 20))
    
    # Add author info
    author_info = """
    <b>Author:</b> Mohammad Babaie<br/>
    <b>Email:</b> mj.babaie@gmail.com<br/>
    <b>LinkedIn:</b> https://www.linkedin.com/in/mohammadbabaie/<br/>
    <b>GitHub:</b> https://github.com/Muh76
    """
    story.append(Paragraph(author_info, normal_style))
    story.append(Spacer(1, 20))
    
    # Process markdown content
    lines = md_content.split('\n')
    in_code_block = False
    code_content = []
    
    for line in lines:
        # Skip the title section since we already added it
        if line.startswith('# Contract Review & Risk Analysis System'):
            continue
        if line.startswith('**Project Overview:**'):
            continue
        if line.startswith('**Author:**'):
            continue
        if line.startswith('---'):
            story.append(Spacer(1, 12))
            continue
        
        # Handle headers
        if line.startswith('## '):
            if in_code_block:
                # End code block
                story.append(Paragraph('<pre>' + '\n'.join(code_content) + '</pre>', code_style))
                code_content = []
                in_code_block = False
            story.append(Paragraph(line[3:], heading1_style))
        elif line.startswith('### '):
            if in_code_block:
                story.append(Paragraph('<pre>' + '\n'.join(code_content) + '</pre>', code_style))
                code_content = []
                in_code_block = False
            story.append(Paragraph(line[4:], heading2_style))
        elif line.startswith('#### '):
            if in_code_block:
                story.append(Paragraph('<pre>' + '\n'.join(code_content) + '</pre>', code_style))
                code_content = []
                in_code_block = False
            story.append(Paragraph(line[5:], heading3_style))
        
        # Handle code blocks
        elif line.startswith('```'):
            if in_code_block:
                # End code block
                story.append(Paragraph('<pre>' + '\n'.join(code_content) + '</pre>', code_style))
                code_content = []
                in_code_block = False
            else:
                # Start code block
                in_code_block = True
        elif in_code_block:
            code_content.append(line)
        
        # Handle regular content
        elif line.strip() and not line.startswith('```'):
            # Convert markdown formatting to HTML
            formatted_line = line
            formatted_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', formatted_line)
            formatted_line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', formatted_line)
            formatted_line = re.sub(r'`(.*?)`', r'<code>\1</code>', formatted_line)
            
            # Handle lists
            if line.strip().startswith('- [ ]'):
                formatted_line = '☐ ' + line.strip()[4:]
            elif line.strip().startswith('- [x]'):
                formatted_line = '☑ ' + line.strip()[4:]
            elif line.strip().startswith('- '):
                formatted_line = '• ' + line.strip()[2:]
            
            story.append(Paragraph(formatted_line, normal_style))
        elif not line.strip():
            story.append(Spacer(1, 6))
    
    # Add final code block if still open
    if in_code_block:
        story.append(Paragraph('<pre>' + '\n'.join(code_content) + '</pre>', code_style))
    
    # Build PDF
    try:
        doc.build(story)
        print("✅ PDF created successfully: Contract_Review_Risk_Analysis_Methodology.pdf")
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        print("Creating a simpler version...")
        create_simple_pdf()

def create_simple_pdf():
    """Create a simpler PDF with basic content"""
    doc = SimpleDocTemplate("Contract_Review_Risk_Analysis_Methodology.pdf", pagesize=A4)
    story = []
    
    styles = getSampleStyleSheet()
    
    # Title
    story.append(Paragraph("Contract Review & Risk Analysis System", styles['Title']))
    story.append(Paragraph("Comprehensive Methodology & Implementation Guide", styles['Heading1']))
    story.append(Spacer(1, 20))
    
    # Author info
    story.append(Paragraph("Author: Mohammad Babaie", styles['Normal']))
    story.append(Paragraph("Email: mj.babaie@gmail.com", styles['Normal']))
    story.append(Paragraph("LinkedIn: https://www.linkedin.com/in/mohammadbabaie/", styles['Normal']))
    story.append(Paragraph("GitHub: https://github.com/Muh76", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Key sections
    sections = [
        ("Project Vision & Market Positioning", "Production-ready legal AI system combining CUAD dataset, risk scoring, RAG, and MLOps for maximum employment impact."),
        ("Technical Architecture", "Multi-tier architecture with FastAPI backend, React frontend, PostgreSQL database, and cloud deployment."),
        ("Implementation Phases", "3-phase approach: MVP (2 weeks), Production features (1 month), Advanced features (2 months)."),
        ("Core Methodologies", "Multi-task learning model, risk scoring algorithm, clause highlighting system, and RAG implementation."),
        ("Risk Scoring & Clause Highlighting", "Prioritize risk scoring over pure classification with actionable intelligence and business impact assessment."),
        ("RAG Implementation", "Retrieval-Augmented Generation for precedent analysis and alternative clause suggestions."),
        ("Compliance & Legal Ops", "Audit trail system, compliance tracking, and legal operations integration."),
        ("MLOps Pipeline", "Continuous training pipeline with model monitoring, versioning, and automated retraining."),
        ("Business Metrics & ROI", "ROI calculator with time savings, cost reduction, and risk mitigation metrics."),
        ("Deployment Strategy", "Docker containerization, cloud deployment, and production-ready infrastructure."),
        ("Testing & Quality Assurance", "Comprehensive testing strategy with unit tests, integration tests, and performance testing."),
        ("Security & Compliance", "Data encryption, access control, audit logging, and GDPR compliance."),
        ("Performance Optimization", "Caching strategies, model optimization, and performance targets."),
        ("Documentation & Portfolio", "Professional documentation structure and portfolio elements for employment."),
        ("Employment Strategy", "Target companies, networking strategy, and interview preparation."),
        ("Success Metrics", "Technical, employment, and business impact metrics for project success.")
    ]
    
    for title, description in sections:
        story.append(Paragraph(title, styles['Heading2']))
        story.append(Paragraph(description, styles['Normal']))
        story.append(Spacer(1, 12))
    
    # Progress tracking
    story.append(Paragraph("Weekly Progress Tracking", styles['Heading2']))
    story.append(Paragraph("Week 1: Development environment setup, CUAD data preprocessing, basic clause extraction model", styles['Normal']))
    story.append(Paragraph("Week 2: Streamlit dashboard, vector database setup, similar clause retrieval, SHAP explanations", styles['Normal']))
    story.append(Paragraph("Week 3-4: FastAPI backend, authentication system, file upload/processing, RESTful endpoints", styles['Normal']))
    story.append(Paragraph("Week 5-6: MLflow tracking, model versioning, performance monitoring, automated retraining", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Success factors
    story.append(Paragraph("Key Success Factors:", styles['Heading2']))
    story.append(Paragraph("1. Focus on Risk: Prioritize risk scoring over pure classification", styles['Normal']))
    story.append(Paragraph("2. RAG Integration: Add precedent analysis and suggestions", styles['Normal']))
    story.append(Paragraph("3. Production Ready: Full MLOps pipeline with monitoring", styles['Normal']))
    story.append(Paragraph("4. Business Focus: ROI metrics and compliance tracking", styles['Normal']))
    story.append(Paragraph("5. Portfolio Quality: Polished demo and documentation", styles['Normal']))
    
    try:
        doc.build(story)
        print("✅ Simple PDF created successfully: Contract_Review_Risk_Analysis_Methodology.pdf")
    except Exception as e:
        print(f"❌ Error creating simple PDF: {e}")

if __name__ == "__main__":
    create_pdf_from_markdown()
