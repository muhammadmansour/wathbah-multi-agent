# -*- coding: utf-8 -*-
"""
Multi-Agent FastAPI System
Generates agents based on both explicit agent_type and message context.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from typing import List, Optional
import aiofiles
import os
import httpx
import PyPDF2
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime

from fastapi import  UploadFile, File, Form, HTTPException



# ==============================
# 1. Model
# ==============================
model = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# ==============================
# 2. FastAPI
# ==============================
app = FastAPI(title="Dynamic Agent API", version="1.0")


class AgentQuery(BaseModel):
    message: str
    agent_type: str | None = None


# ==============================
# 3. Tools
# ==============================
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_API_URL = "https://api.anthropic.com"

# Email configuration
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "moxxmo027@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")  # Gmail App Password

# Validate required environment variables
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")
if not EMAIL_PASSWORD:
    print("⚠️ Warning: EMAIL_PASSWORD not set. Email functionality will be disabled.")

async def upload_to_claude(file: UploadFile) -> str:
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "files-api-2025-04-14",
    }
    # send multipart/form-data
    files = {
        "file": (file.filename, await file.read(), file.content_type)
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{ANTHROPIC_API_URL}/v1/files",
            headers=headers,
            files=files,
        )
    resp.raise_for_status()
    data = resp.json()
    
    # Check if file_id exists in response
    if "file_id" not in data:
        print(f"API Response: {data}")  # Debug: print the actual response
        raise HTTPException(
            status_code=500, 
            detail=f"File upload failed: 'file_id' not found in response. Response: {data}"
        )
    
    return data["file_id"]

async def extract_text_from_pdf(file_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

@tool
def send_email(recipient: str, subject: str, body: str, attachment_path: str = None) -> str:
    """
    Send an email using Gmail's SMTP server with optional file attachment.

    Args:
        recipient (str): Email address of the recipient.
        subject (str): Subject line of the email.
        body (str): Body text of the email.
        attachment_path (str): Optional path to file to attach.

    Returns:
        str: Success or failure message.
    """
    if not EMAIL_PASSWORD:
        return "❌ Email functionality disabled: EMAIL_PASSWORD environment variable not set"
    
    sender = EMAIL_SENDER
    password = EMAIL_PASSWORD

    # Create message container
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient

    # Add body to email
    msg.attach(MIMEText(body, "plain"))

    # Add attachment if provided
    if attachment_path and os.path.exists(attachment_path):
        try:
            with open(attachment_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            
            # Encode file in ASCII characters to send by email
            encoders.encode_base64(part)
            
            # Add header as key/value pair to attachment part
            filename = os.path.basename(attachment_path)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {filename}",
            )
            
            # Attach the part to message
            msg.attach(part)
            
        except Exception as e:
            return f"❌ Failed to attach file {attachment_path}: {str(e)}"

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.send_message(msg)

        attachment_info = f" with attachment {os.path.basename(attachment_path)}" if attachment_path and os.path.exists(attachment_path) else ""
        return f"✅ Email sent to {recipient} with subject '{subject}'{attachment_info}"
    except Exception as e:
        return f"❌ Failed to send email: {str(e)}"

@tool
def analyze_financial_document(file_content: str) -> str:
    """
    Analyze financial statements and provide insights.
    
    Args:
        file_content (str): The text content of the financial document.
        
    Returns:
        str: Analysis of the financial statements.
    """
    # This tool is used by the analyzer agent to process financial documents
    # The actual analysis is performed by the AI model through the agent's reasoning
    # This tool just provides a structured way to reference the file content
    return f"Financial document content available for analysis:\n\n{file_content[:2000]}..."

@tool
def summarize_document(file_content: str) -> str:
    """
    Summarize a document.
    
    Args:
        file_content (str): The text content of the document to summarize.
        
    Returns:
        str: Summary of the document.
    """
    # This tool will be used by the summarizer agent
    return f"Document summary:\n\n{file_content[:500]}...\n\n[This is a placeholder - the actual summary would be generated by the AI model]"

@tool
def detect_document_type(file_content: str) -> str:
    """
    Detect the type of document based on its content.
    
    Args:
        file_content (str): The text content of the document.
        
    Returns:
        str: Type of document detected.
    """
    try:
        content_lower = file_content.lower()
        
        # Financial document indicators
        financial_keywords = [
            "balance sheet", "income statement", "cash flow", "profit and loss",
            "revenue", "expenses", "assets", "liabilities", "equity", "financial statements",
            "net income", "gross profit", "operating income", "retained earnings"
        ]
        
        # Product/service description indicators
        product_keywords = [
            "product description", "service description", "features", "benefits",
            "pricing", "package", "offering", "solution", "platform"
        ]
        
        # Legal document indicators
        legal_keywords = [
            "contract", "agreement", "terms and conditions", "legal", "clause",
            "whereas", "party", "obligations", "jurisdiction"
        ]
        
        # Research/academic indicators
        research_keywords = [
            "abstract", "methodology", "research", "study", "analysis",
            "conclusion", "references", "bibliography", "hypothesis"
        ]
        
        # Count keyword matches
        financial_score = sum(1 for keyword in financial_keywords if keyword in content_lower)
        product_score = sum(1 for keyword in product_keywords if keyword in content_lower)
        legal_score = sum(1 for keyword in legal_keywords if keyword in content_lower)
        research_score = sum(1 for keyword in research_keywords if keyword in content_lower)
        
        # Determine document type based on highest score
        scores = {
            "financial": financial_score,
            "product_description": product_score,
            "legal": legal_score,
            "research": research_score,
            "general": 0  # fallback
        }
        
        detected_type = max(scores, key=scores.get)
        
        # Require minimum threshold
        if scores[detected_type] < 2:
            detected_type = "general"
        
        return f"Document type detected: {detected_type} (confidence: {scores[detected_type]} keywords matched)"
        
    except Exception as e:
        return f"Error detecting document type: {str(e)}"

@tool
def send_email_with_pdf(recipient: str, subject: str, body: str, pdf_path: str) -> str:
    """
    Send an email with a PDF attachment.
    
    Args:
        recipient (str): Email address of the recipient.
        subject (str): Subject line of the email.
        body (str): Body text of the email.
        pdf_path (str): Path to the PDF file to attach.
        
    Returns:
        str: Success or failure message.
    """
    return send_email(recipient, subject, body, pdf_path)

@tool
def search_historical_info(topic: str) -> str:
    """
    Search for historical information about a topic.
    
    Args:
        topic (str): Historical topic or character to search for.
        
    Returns:
        str: Historical information about the topic.
    """
    # This tool would typically connect to historical databases or APIs
    # For now, it provides a structured response format
    return f"Historical research for: {topic}\n\n[This would connect to historical databases to provide accurate, well-sourced information about {topic}]"

@tool
def search_sports_info(topic: str) -> str:
    """
    Search for sports information about a topic.
    
    Args:
        topic (str): Sports topic, player, team, or event to search for.
        
    Returns:
        str: Sports information about the topic.
    """
    # This tool would typically connect to sports databases or APIs
    # For now, it provides a structured response format
    return f"Sports research for: {topic}\n\n[This would connect to sports databases to provide current stats, achievements, and information about {topic}]"

@tool
def calculate_math(expression: str) -> str:
    """
    Calculate mathematical expressions safely.
    
    Args:
        expression (str): Mathematical expression to evaluate (e.g., "2*2", "10+5", "sqrt(16)").
        
    Returns:
        str: Result of the calculation or error message.
    """
    try:
        # Import math functions for safe evaluation
        import math
        import operator
        
        # Define safe operations
        safe_dict = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
            "floor": math.floor,
            "ceil": math.ceil,
            "factorial": math.factorial,
        }
        
        # Add basic operators
        safe_dict.update({
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
            "//": operator.floordiv,
            "%": operator.mod,
            "**": operator.pow,
        })
        
        # Evaluate the expression safely
        result = eval(expression, safe_dict)
        
        # Format the result nicely
        if isinstance(result, float):
            if result.is_integer():
                result = int(result)
            else:
                result = round(result, 6)  # Round to 6 decimal places
        
        return f"Calculation: {expression} = {result}"
        
    except ZeroDivisionError:
        return f"Error: Division by zero in expression '{expression}'"
    except ValueError as e:
        return f"Error: Invalid value in expression '{expression}': {str(e)}"
    except SyntaxError:
        return f"Error: Invalid syntax in expression '{expression}'"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"

@tool
def generate_analysis_pdf(analysis_data: str, filename: str = None) -> str:
    """
    Generate a PDF report from analysis data.
    
    Args:
        analysis_data (str): The analysis content to include in the PDF.
        filename (str): Optional filename for the PDF (without extension).
        
    Returns:
        str: Path to the generated PDF file.
    """
    try:
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"financial_analysis_{timestamp}"
        
        # Ensure filename has .pdf extension
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        
        # Create output directory if it doesn't exist
        output_dir = "generated_reports"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(file_path, pagesize=A4)
        styles = getSampleStyleSheet()
        
        # Define custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        # Build PDF content
        story = []
        
        # Title
        story.append(Paragraph("Financial Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Date
        current_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        story.append(Paragraph(f"Generated on: {current_date}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Analysis content
        story.append(Paragraph("Analysis Results", heading_style))
        
        # Split analysis into paragraphs for better formatting
        analysis_paragraphs = analysis_data.split('\n\n')
        for para in analysis_paragraphs:
            if para.strip():
                story.append(Paragraph(para.strip(), styles['Normal']))
                story.append(Spacer(1, 6))
        
        # Build PDF
        doc.build(story)
        
        return f"PDF report generated successfully: {file_path}"
        
    except Exception as e:
        return f"Error generating PDF: {str(e)}"



async def detect_agent_type(message: str, has_files: bool = False, file_types: list = None) -> str:
    """
    Decide agent type based on message context and file attachments.
    Priority: If files are attached, prefer analyzer/summarizer over mailer.
    """
    print(f"🔍 Detecting agent type...")
    print(f"   Message: '{message}'")
    print(f"   Has files: {has_files}")
    print(f"   File types: {file_types}")
    
    text = message.lower()
    file_types = file_types or []

    # If files are attached, prioritize analysis/summarization
    if has_files:
        print(f"   📁 Files detected, checking for email requests...")
        # Check if user explicitly wants to send email
        if "send mail" in text or "email" in text or "send to" in text:
            print(f"   📧 Email request detected in message")
            return "mailer"
        # For PDF files, check user intent
        elif any(ft.lower() == 'pdf' for ft in file_types):
            print(f"   📄 PDF file detected")
            if "financial" in text or any(keyword in text for keyword in ["analyze", "analysis", "balance sheet", "income statement", "cash flow"]):
                print(f"   📊 Financial analysis keywords detected")
                return "analyzer"
            elif "summarize" in text or "summary" in text:
                print(f"   📝 Summarization request detected")
                return "summarizer"
            elif "analyze" in text or "analysis" in text:
                print(f"   📊 Analysis request detected")
                return "analyzer"
            else:
                print(f"   📊 PDF file - defaulting to analyzer for document analysis")
                return "analyzer"  # Default to analyzer for document analysis
        else:
            agent_type = "summarizer" if "summarize" in text else "analyzer"
            print(f"   📄 Non-PDF file, choosing: {agent_type}")
            return agent_type

    # No files attached - use message content only
    print(f"   📝 No files, analyzing message content...")
    
    # Check for explicit email requests (multilingual patterns)
    explicit_email_patterns = [
        # English
        "send mail", "send email", "email to", "send to", "mail to", "forward to", "share via email",
        # Arabic
        "البريد الإلكتروني", "إرسال", "بريد", "مشاركة",
        # Spanish
        "enviar correo", "enviar email", "correo electrónico",
        # French
        "envoyer mail", "envoyer email", "courrier électronique",
        # German
        "e-mail senden", "email versenden"
    ]
    
    # Check both lowercase and original text for multilingual support
    email_detected = any(pattern in text for pattern in explicit_email_patterns) or \
                    any(pattern in message for pattern in explicit_email_patterns)
    
    if email_detected:
        print(f"   📧 Explicit email request detected")
        return "mailer"
    elif "summarize" in text or "summary" in text:
        print(f"   📝 Summarization request detected")
        return "summarizer"
    elif "analyze" in text or "analysis" in text or "financial" in text:
        print(f"   📊 Analysis request detected")
        return "analyzer"
    elif any(math_indicator in text for math_indicator in ["+", "-", "*", "/", "=", "calculate", "solve", "math", "equation", "formula", "sqrt", "square root", "power", "exponent"]):
        print(f"   🔢 Mathematical query detected")
        return "mathematical"
    else:
        print(f"   🤖 Using LLM classifier for general query...")
        # Let LLM decide for general queries
        classifier_prompt = f"""
        You are a multilingual router. Classify the user's request into one of these agent types:
        - mailer: for sending emails (البريد الإلكتروني, enviar correo, envoyer mail, e-mail senden)
        - summarizer: for summarizing content (ملخص, resumen, résumé, zusammenfassung)
        - analyzer: for analyzing documents (تحليل, análisis, analyse, analyse)
        - mathematical: for math problems (حساب, matemáticas, mathématiques, mathematik)
        - historian: for historical information (تاريخ, historia, histoire, geschichte)
        - sports: for sports information (رياضة, deportes, sports, sport)
        - general: for general questions and information requests

    User query: "{message}"

        Important: 
        - The user may write in English, Arabic, Spanish, French, or German
        - Only choose "mailer" if the user explicitly wants to send an email
        - Choose "mathematical" for any math problems, calculations, or arithmetic
        - If they're asking a question and mentioning email as a secondary action, choose "general"
        - Look for keywords in any language that indicate the user's intent

    Respond with only one word (the English agent type: mailer, summarizer, analyzer, mathematical, historian, sports, or general).
    """
    result = await model.ainvoke(classifier_prompt)
    agent_type = result.content.strip().lower()

    # fallback safety
    if agent_type not in ["general", "mailer", "summarizer", "analyzer", "mathematical", "historian", "sports"]:
        agent_type = "general"

    print(f"   🤖 LLM classified as: {agent_type}")
    return agent_type

async def analyze_context_for_additional_agents(message: str, primary_agent_type: str) -> list[str]:
    """
    Analyze the message context to detect additional agent types needed beyond the primary agent.
    For example, if primary is 'financial' but context mentions 'send mail', return ['mailer'].
    """
    print(f"🔍 Analyzing context for additional agents...")
    print(f"   Primary agent: {primary_agent_type}")
    print(f"   Message: '{message}'")
    
    additional_agents = []
    text = message.lower() if message else ""
    
    # Define keywords for different agent types (multilingual support)
    agent_keywords = {
        "mailer": [
            # English
            "send mail", "send email", "email", "send to", "mail to", "forward to", "share via email", "notify", "distribute", "share with",
            # Arabic
            "البريد الإلكتروني", "إرسال", "بريد", "مشاركة", "إشعار", "توزيع",
            # Spanish
            "enviar correo", "email", "correo electrónico", "enviar a", "compartir por correo",
            # French
            "envoyer mail", "envoyer email", "courrier électronique", "partager par email",
            # German
            "e-mail senden", "email versenden", "per email teilen"
        ],
        "analyzer": [
            # English
            "analyze", "analysis", "examine", "review", "assess", "evaluate", "financial", "balance sheet", "income statement",
            # Arabic
            "تحليل", "فحص", "مراجعة", "تقييم", "مالي", "الميزانية العمومية",
            # Spanish
            "analizar", "análisis", "examinar", "revisar", "evaluar", "financiero",
            # French
            "analyser", "analyse", "examiner", "réviser", "évaluer", "financier",
            # German
            "analysieren", "analyse", "untersuchen", "überprüfen", "bewerten", "finanziell"
        ],
        "summarizer": [
            # English
            "summarize", "summary", "brief", "overview", "condensed", "digest",
            # Arabic
            "ملخص", "تلخيص", "موجز", "نظرة عامة",
            # Spanish
            "resumir", "resumen", "resumen ejecutivo", "sinopsis",
            # French
            "résumer", "résumé", "synthèse", "aperçu",
            # German
            "zusammenfassen", "zusammenfassung", "übersicht"
        ],
        "mathematical": [
            # English
            "calculate", "compute", "solve", "math", "equation", "formula", "+", "-", "*", "/", "=", "sqrt",
            # Arabic
            "حساب", "احسب", "رياضيات", "معادلة", "حل",
            # Spanish
            "calcular", "computar", "resolver", "matemáticas", "ecuación",
            # French
            "calculer", "résoudre", "mathématiques", "équation",
            # German
            "berechnen", "rechnen", "lösen", "mathematik", "gleichung"
        ],
        "historian": [
            # English
            "history", "historical", "past", "ancient", "medieval", "renaissance", "war", "empire", "civilization",
            # Arabic
            "تاريخ", "تاريخي", "الماضي", "قديم", "حرب", "إمبراطورية", "حضارة",
            # Spanish
            "historia", "histórico", "pasado", "antiguo", "guerra", "imperio", "civilización",
            # French
            "histoire", "historique", "passé", "ancien", "guerre", "empire", "civilisation",
            # German
            "geschichte", "historisch", "vergangenheit", "antik", "krieg", "reich", "zivilisation"
        ],
        "sports": [
            # English
            "sports", "athlete", "team", "game", "match", "score", "championship", "league", "player", "football", "basketball", "soccer",
            # Arabic
            "رياضة", "رياضي", "فريق", "لعبة", "مباراة", "نتيجة", "بطولة", "دوري", "لاعب", "كرة القدم", "كرة السلة",
            # Spanish
            "deportes", "atleta", "equipo", "juego", "partido", "puntuación", "campeonato", "liga", "jugador", "fútbol", "baloncesto",
            # French
            "sports", "athlète", "équipe", "jeu", "match", "score", "championnat", "ligue", "joueur", "football", "basketball",
            # German
            "sport", "athlet", "mannschaft", "spiel", "match", "punktzahl", "meisterschaft", "liga", "spieler", "fußball", "basketball"
        ]
    }
    
    # Check for each agent type (except the primary one)
    for agent_type, keywords in agent_keywords.items():
        if agent_type != primary_agent_type:
            # Check both original text and lowercase for multilingual support
            matched_keywords = []
            for keyword in keywords:
                if keyword.lower() in text or keyword in message:  # Check both cases for non-English text
                    matched_keywords.append(keyword)
            
            if matched_keywords:
                print(f"   🎯 Found keywords for {agent_type}: {matched_keywords}")
                additional_agents.append(agent_type)
    
    # Use LLM for more sophisticated context analysis if needed
    if not additional_agents and message:
        print(f"   🤖 Using LLM for advanced context analysis...")
        context_prompt = f"""
        Analyze this user message to identify if they need multiple types of assistance.
        
        Primary agent type: {primary_agent_type}
        User message: "{message}"
        
        Available agent types:
        - mailer: for sending emails
        - analyzer: for document analysis
        - summarizer: for document summarization  
        - mathematical: for calculations
        - historian: for historical information
        - sports: for sports information
        - general: for general questions
        
        Look for secondary tasks or actions mentioned in the message that would require different agent types.
        For example:
        - "Analyze this financial report and send it to John" → primary: analyzer, additional: mailer
        - "Calculate the ROI and email the results" → primary: mathematical, additional: mailer
        - "Summarize this document and share with the team" → primary: summarizer, additional: mailer
        
        Return ONLY the additional agent types needed (comma-separated), or "none" if no additional agents are needed.
        Do not include the primary agent type in your response.
        """
        
        try:
            result = await model.ainvoke(context_prompt)
            response = result.content.strip().lower()
            
            if response != "none" and response:
                # Parse comma-separated agent types
                detected_agents = [agent.strip() for agent in response.split(",") if agent.strip()]
                # Validate agent types
                valid_agents = [agent for agent in detected_agents if agent in agent_keywords.keys()]
                additional_agents.extend(valid_agents)
                print(f"   🤖 LLM detected additional agents: {valid_agents}")
        except Exception as e:
            print(f"   ⚠️ LLM context analysis failed: {e}")
    
    # Remove duplicates while preserving order
    unique_additional_agents = []
    for agent in additional_agents:
        if agent not in unique_additional_agents:
            unique_additional_agents.append(agent)
    
    print(f"   📋 Additional agents needed: {unique_additional_agents}")
    return unique_additional_agents



# ==============================
# 5. Dynamic Agent Factory
# ==============================
def create_dynamic_agent(agent_type: str):
    print(f"🔧 Creating agent of type: {agent_type}")
    
    tools = []

    if agent_type == "mailer":
        tools = [send_email, send_email_with_pdf]
        print(f"📧 Mailer agent created with tools: {[tool.name for tool in tools]}")
    elif agent_type == "summarizer":
        tools = [summarize_document]
        print(f"📝 Summarizer agent created with tools: {[tool.name for tool in tools]}")
    elif agent_type == "analyzer":
        tools = [analyze_financial_document, generate_analysis_pdf, detect_document_type]
        print(f"📊 Analyzer agent created with tools: {[tool.name for tool in tools]}")
    elif agent_type == "mathematical":
        tools = [calculate_math]
        print(f"🔢 Mathematical agent created with tools: {[tool.name for tool in tools]}")
    elif agent_type == "historian":
        tools = [search_historical_info]
        print(f"📚 Historian agent created with tools: {[tool.name for tool in tools]}")
    elif agent_type == "sports":
        tools = [search_sports_info]
        print(f"⚽ Sports agent created with tools: {[tool.name for tool in tools]}")
    else:  # general
        tools = []
        print(f"🤖 General agent created with no tools")

    # Create specific prompts for each agent type
    prompts = {
        "mailer": "You are an email assistant. You can send emails to recipients with optional file attachments. Use send_email for basic emails or send_email_with_pdf for emails with PDF attachments. When a PDF file path is provided, use send_email_with_pdf to actually attach the file to the email. If the user mentions an email address but no specific subject or body, create appropriate content based on the context. Always ask for clarification if you need more information to send a complete email.",
        "summarizer": "You are a document summarization expert. You can analyze and summarize documents, especially PDFs. When you receive file attachments, extract and analyze their content to provide comprehensive summaries.",
        "analyzer": "You are a document analysis expert. FIRST, always use the detect_document_type tool to identify what type of document you're analyzing. Based on the document type detected, provide appropriate analysis: financial analysis for financial documents, content analysis for product descriptions, legal analysis for contracts, etc. You can also generate PDF reports of your analysis using the generate_analysis_pdf tool. IMPORTANT: Never assume document type - always detect it first, then provide analysis appropriate to that document type.",
        "mathematical": "You are a mathematical assistant. You can solve arithmetic problems, algebraic equations, and perform various mathematical calculations. Use the calculate_math tool to compute mathematical expressions. Always show your work and explain the steps when solving complex problems.",
        "historian": "You are a historical research assistant. You can provide detailed information about historical figures, events, periods, and civilizations. Use the search_historical_info tool to access historical databases and provide accurate, well-sourced historical information. Always cite sources and provide context for historical events.",
        "sports": "You are a sports information assistant. You can provide detailed information about athletes, teams, games, statistics, and sports history. Use the search_sports_info tool to access current sports data and provide up-to-date information about players, teams, and sporting events.",
        "general": "You are a general assistant. You can help with various tasks and answer questions. When you receive file attachments, you can reference and discuss their content. If the user asks a question and mentions sending the answer via email, first provide the answer, then offer to send it via email if they'd like."
    }

    agent = create_react_agent(
        model=model,
        tools=tools,
        name=f"{agent_type}_agent",
        prompt=prompts.get(agent_type, prompts["general"]),
    )
    
    print(f"✅ Agent '{agent_type}' created successfully")
    return agent

async def run_multi_agent_workflow(message: str, files: list = None, agent_type: str = None):
    """
    Run a workflow that generates agents based on both explicit agent_type and message context.
    This function creates multiple agents as needed and chains them together.
    """
    print(f"🚀 Starting Multi-Agent Workflow System")
    print(f"   Message: '{message}'")
    print(f"   Explicit agent_type: {agent_type}")
    print(f"   Files: {[f.filename for f in files] if files else 'None'}")
    
    # Process files first
    saved_files = []
    file_contents = []
    if files:
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)

        for file in files:
            file_path = os.path.join(upload_dir, file.filename)
            async with aiofiles.open(file_path, "wb") as out_file:
                content = await file.read()
                await out_file.write(content)
            saved_files.append(file.filename)
            
            # Extract text from PDF files
            if file.filename.lower().endswith('.pdf'):
                pdf_text = await extract_text_from_pdf(file_path)
                file_contents.append(f"File: {file.filename}\nContent:\n{pdf_text}")

    # Step 1: Determine primary agent type
    primary_agent_type = agent_type
    if not primary_agent_type:
        file_types = [f.split('.')[-1] for f in saved_files] if saved_files else []
        primary_agent_type = await detect_agent_type(message, has_files=bool(files), file_types=file_types)
    
    print(f"   🎯 Primary agent type: {primary_agent_type}")
    
    # Step 2: Analyze context for additional agents needed
    additional_agent_types = await analyze_context_for_additional_agents(message, primary_agent_type)
    
    # Step 3: Create execution plan
    agent_chain = [primary_agent_type] + additional_agent_types
    print(f"   📋 Agent execution chain: {agent_chain}")
    
    # Step 4: Execute agents in sequence
    workflow_results = {}
    context_data = {}
    
    # Prepare initial context message
    context_message = message or ""
    if saved_files:
        context_message += f"\n\n[Attached files: {', '.join(saved_files)}]"
        if file_contents:
            context_message += f"\n\n[File Contents:\n{chr(10).join(file_contents)}]"
    
    for i, current_agent_type in enumerate(agent_chain):
        print(f"\n🔧 Step {i+1}: Running {current_agent_type} agent...")
        
        # Create and configure agent
        agent = create_dynamic_agent(current_agent_type)
        
        # Prepare agent-specific context
        agent_context = context_message
        
        # Add specific instructions based on agent type and position in chain
        if current_agent_type == "analyzer":
            agent_context += f"\n\nIMPORTANT: First detect the document type, then provide analysis appropriate to that document type. Focus ONLY on the analysis."
        elif current_agent_type == "mailer" and i > 0:
            # If mailer is not the first agent, include results from previous agents
            previous_results = []
            for prev_agent, prev_result in workflow_results.items():
                if prev_agent != "mailer":
                    previous_results.append(f"{prev_agent.title()} Results:\n{prev_result}")
            
            if previous_results:
                agent_context = f"""
                Please send an email with the following results from previous analysis.
                
                Original user request: "{message}"
                
                {chr(10).join(previous_results)}
                
                PDF Report: {context_data.get('pdf_path', 'No PDF generated')}
                
                If the user mentioned specific recipients, email addresses, or teams, use that information.
                If no specific recipient was mentioned, ask for clarification or use a default recipient.
                
                Include the analysis results in the email body and attach any PDF reports if available.
                
                IMPORTANT: Use the appropriate email tool (send_email or send_email_with_pdf) based on whether PDF is available.
                """
        elif current_agent_type == "summarizer" and context_data.get("analysis_result"):
            # If summarizer comes after analyzer, summarize the analysis
            agent_context += f"\n\nPrevious analysis results to summarize:\n{context_data.get('analysis_result')}"
        
        print(f"   📝 Context message length: {len(agent_context)} characters")
        
        # Execute agent
        try:
            result = await agent.ainvoke({"messages": [("user", agent_context.strip())]})
            answer = extract_final_answer(result)
            workflow_results[current_agent_type] = answer
            
            print(f"   ✅ {current_agent_type} completed, response length: {len(answer)} characters")
            
            # Store specific data for next agents
            if current_agent_type == "analyzer":
                context_data["analysis_result"] = answer
                # Generate PDF if needed for subsequent mailer
                if "mailer" in agent_chain[i+1:]:
                    print(f"   📄 Generating PDF for future email...")
                    pdf_result = generate_analysis_pdf(answer)
                    if "PDF report generated successfully:" in pdf_result:
                        pdf_path = pdf_result.split("PDF report generated successfully: ")[1].strip()
                        context_data["pdf_path"] = pdf_path
                        context_data["pdf_result"] = pdf_result
                        print(f"   📄 PDF generated: {pdf_path}")
                    
        except Exception as e:
            print(f"   ❌ {current_agent_type} agent failed: {e}")
            workflow_results[current_agent_type] = f"Error: {str(e)}"
    
    # Step 5: Prepare final response
    print(f"\n📊 Workflow Summary:")
    print(f"   Agents executed: {list(workflow_results.keys())}")
    
    # Create comprehensive response
    response_parts = []
    for agent_type, result in workflow_results.items():
        response_parts.append(f"=== {agent_type.title()} Results ===\n{result}")
    
    final_response = "\n\n".join(response_parts)
    
    return {
        "workflow_type": "multi_agent",
        "agent_chain": agent_chain,
        "agents_executed": list(workflow_results.keys()),
        "primary_agent": primary_agent_type,
        "additional_agents": additional_agent_types,
        "individual_results": workflow_results,
        "pdf_generated": context_data.get("pdf_result"),
        "pdf_path": context_data.get("pdf_path"),
        "response": final_response,
        "uploaded_files": saved_files,
    }

async def run_agent_workflow(message: str, files: list = None, agent_type: str = None):
    """
    Legacy wrapper that now uses the new multi-agent system.
    Maintains backward compatibility while providing enhanced functionality.
    """
    return await run_multi_agent_workflow(message, files, agent_type)


# ==============================
# 6. Extract Final Response
# ==============================
def extract_final_answer(result):
    messages = result["messages"]
    ai_messages = [
        m.content
        for m in messages
        if m.__class__.__name__ == "AIMessage" and isinstance(m.content, str)
    ]
    return ai_messages[-1].strip() if ai_messages else "No AI response found."


# ==============================
# 7. Endpoint
# ==============================

@app.post("/ask")
async def ask_with_files(
    message: str | None = Form(None),
    agent_type: str | None = Form(None),
    files: list[UploadFile] | None = File(None)
):
    if not message and not files:
        raise HTTPException(status_code=400, detail="Must provide message, file, or both")

    # Use the new workflow function
    return await run_agent_workflow(message, files, agent_type)

@app.post("/multi-agent")
async def multi_agent_endpoint(
    message: str | None = Form(None),
    agent_type: str | None = Form(None),
    files: list[UploadFile] | None = File(None)
):
    """
    Explicit multi-agent endpoint that generates agents based on both agent_type and message context.
    
    Example scenarios:
    - agent_type: "financial", message: "analyze this report and send to john@company.com" 
      → Creates financial agent + mailer agent
    - agent_type: "analyzer", message: "examine this document and email the summary"
      → Creates analyzer agent + mailer agent  
    - message: "calculate the ROI and share the results via email"
      → Creates mathematical agent + mailer agent
    """
    if not message and not files:
        raise HTTPException(status_code=400, detail="Must provide message, file, or both")

    print(f"\n🚀 === MULTI-AGENT ENDPOINT CALLED ===")
    print(f"📝 Message: '{message}'")
    print(f"🎯 Explicit agent_type: {agent_type}")
    print(f"📁 Files: {[f.filename for f in files] if files else 'None'}")
    
    # Use the multi-agent workflow directly for full visibility
    result = await run_multi_agent_workflow(message, files, agent_type)
    
    # Add additional metadata for debugging/monitoring
    result["endpoint_used"] = "multi-agent"
    result["request_details"] = {
        "explicit_agent_type": agent_type,
        "message_provided": bool(message),
        "files_uploaded": len(files) if files else 0,
        "file_names": [f.filename for f in files] if files else []
    }
    
    print(f"✅ Multi-agent workflow completed")
    print(f"   Primary agent: {result.get('primary_agent')}")
    print(f"   Additional agents: {result.get('additional_agents')}")
    print(f"   Total agents executed: {len(result.get('agents_executed', []))}")
    
    return result

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Multi-Agent FastAPI Server...")
    print("📋 Available endpoints:")
    print("   • POST /ask - General agent endpoint")
    print("   • POST /multi-agent - Explicit multi-agent endpoint")
    print("   • GET /docs - API documentation")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)