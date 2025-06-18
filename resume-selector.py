import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF
from io import BytesIO
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
import re

# Load environment variables
load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_API_BASE")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Page configuration
st.set_page_config(page_title="AutoRankCV", layout="wide", page_icon="📄")

# Header
st.markdown("""
    <h1 style='text-align: center;'>📄 AutoRank CV</h1>
    <h4 style='text-align: center; color: grey;'>AI-powered Resume Ranking with Parameter-Based Scoring</h4>
    <hr style='margin-top: 0;'>
""", unsafe_allow_html=True)

# Resume Count Selector
st.markdown("### Number of Resumes")
st.write("Select the number of resumes you want to upload for analysis. The maximum is 100.")
max_files = st.number_input("Select how many resumes to upload", min_value=1, max_value=100, value=10, step=1)

# File uploader
st.markdown("### Upload Resumes (PDF only)")
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > max_files:
        st.error(f"You selected {max_files}, but uploaded {len(uploaded_files)}. Please reduce your uploads.")
    else:
        st.success(f"✅ Uploaded {len(uploaded_files)} resume(s).")

# Job Description
st.markdown("### Job Description")
job_description = st.text_area("Paste the job description here", height=300)

# Analyze Button (smaller and right aligned)
btn_col = st.columns([10, 1])[1]
analyze_button = btn_col.button("Analyze", use_container_width=True)

# LangChain Setup
@st.cache_resource
def init_chain():
    llm = AzureChatOpenAI(
        openai_api_base=api_base,
        openai_api_version=api_version,
        openai_api_key=api_key,
        deployment_name=deployment_name,
        model_name="gpt-4o",
        temperature=0.3,
        max_tokens=1600,
        top_p=0.95,
    )
    prompt = PromptTemplate(
        input_variables=["resumes", "job_description"],
        template="""
You are a senior recruiter.

A company is hiring for this role:
---
{job_description}
---

You're given several resumes labeled Candidate 1, Candidate 2, etc.

For each candidate, evaluate on the following criteria:
1. Technical Skills
2. Relevant Experience
3. Education Alignment
4. Communication and Presentation
5. Overall Fit for the Role

For each parameter, give a score out of 10 with reasoning. Then compute a final average score and rank all candidates from best to worst.

Provide:
- Detailed analysis for each candidate.
- A final summary table:
| Rank | Candidate | Tech Skills | Experience | Education | Communication | Overall Fit | Final Score |

Now begin:
{resumes}
"""
    )
    return LLMChain(llm=llm, prompt=prompt)

llm_chain = init_chain()

# PDF Text Extractor
def extract_text_from_pdf(file):
    text = ""
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        text += f"\n[Error reading PDF: {str(e)}]"
    return text.strip()

# Markdown cleaner
def clean_markdown(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)     # remove bold
    text = re.sub(r"#+\s*", "", text)                # remove headings
    text = re.sub(r"- ", "• ", text)                 # bullet points
    return text.strip()

# PDF Report Generator (final version)
def generate_full_pdf_report(full_text, summary_lines):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=60)
    elements = []
    styles = getSampleStyleSheet()
    normal = ParagraphStyle(name="Normal", fontSize=10, leading=14, spaceAfter=6)
    header = ParagraphStyle(name="Header", parent=styles["Heading2"], spaceAfter=12)

    # Clean and split candidate sections (before summary table)
    cleaned_text = re.split(r"\n\s*### Final Summary Table", full_text)[0]
    # Remove any pipe table lines from the body
    cleaned_text = "\n".join([line for line in cleaned_text.splitlines() if not line.strip().startswith("|")])
    sections = cleaned_text.split("---")

    for section in sections:
        cleaned = clean_markdown(section.strip())
        if cleaned:
            for para in cleaned.split('\n'):
                if para.strip():
                    elements.append(Paragraph(para.strip(), normal))
            elements.append(Spacer(1, 0.2 * inch))

    # Add Summary Table once
    # Clean each cell to remove markdown like **bold**
    rows = [
        [re.sub(r"\*\*(.*?)\*\*", r"\1", cell.strip()) for cell in line.strip('| ').split('|')]
        for line in summary_lines
        if '|' in line and '---' not in line
    ]
    if rows:
        elements.append(Paragraph("Final Summary Table", header))
        table = Table(rows, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# Candidate Name Formatter for web
def bold_candidate_names(text):
    return re.sub(r"(Candidate\s+\d+:)", r"<strong style='font-size:1.1em;'>\1</strong>", text)

# Main logic
if analyze_button:
    if not uploaded_files or len(uploaded_files) > max_files:
        st.warning("⚠️ Please upload the number of resumes you selected.")
    elif not job_description.strip():
        st.warning("⚠️ Please enter the job description.")
    else:
        with st.spinner("🔍 Analyzing resumes..."):
            resume_texts = []
            for idx, file in enumerate(uploaded_files):
                text = extract_text_from_pdf(file)
                resume_texts.append(f"\n\nCandidate {idx+1} Resume:\n{text[:3000]}")

            combined = "\n".join(resume_texts)
            response = llm_chain.invoke({
                "resumes": combined,
                "job_description": job_description
            })

            result_text = response['text']
            st.success("✅ Analysis Complete!")

            # Display in app
            st.markdown("### 📄 Detailed Evaluation")
            st.markdown(bold_candidate_names(result_text), unsafe_allow_html=True)

            # Extract table lines
            table_lines = [line for line in result_text.splitlines() if line.strip().startswith('|')]

            # Generate PDF
            pdf_file = generate_full_pdf_report(result_text, table_lines)

            st.download_button(
                label="📥 Download Full Report as PDF",
                data=pdf_file,
                file_name="resume_ranking_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
