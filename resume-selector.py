import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF for PDF text extraction
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load environment variables
load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_API_BASE")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Page configuration
st.set_page_config(page_title="AutoRankCV", layout="wide", page_icon="📄")

st.markdown(
    "<h1 style='text-align: center;'>📄 AutoRank CV</h1>"
    "<h5 style='text-align: center; color: grey;'>AI-powered Resume Ranking using GPT-4o & LangChain</h5>",
    unsafe_allow_html=True
)
st.markdown("---")

# Upload resumes and job description input
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📂 Upload Resumes")
    uploaded_files = st.file_uploader("Upload up to 10 PDF resumes", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        st.success(f"{len(uploaded_files)} resume(s) uploaded.")
    else:
        st.info("Awaiting resume uploads.")

with col2:
    st.subheader("🧾 Job Description")
    job_description = st.text_area(
        "Paste the job description here:",
        height=300,
        placeholder="E.g. We're looking for a Machine Learning Engineer with experience in Python, ML frameworks, cloud platforms..."
    )

# Analyze button - smaller and right-aligned
_, col_btn, _ = st.columns([7, 1.3, 1])
analyze_button = col_btn.button("Analyze", use_container_width=True)

# LLM Setup
@st.cache_resource
def init_chain():
    llm = AzureChatOpenAI(
        openai_api_base=api_base,
        openai_api_version=api_version,
        openai_api_key=api_key,
        deployment_name=deployment_name,
        model_name="gpt-4o",
        temperature=0.3,
        max_tokens=1200,
        top_p=0.95,
    )

    prompt = PromptTemplate(
        input_variables=["resumes", "job_description"],
        template="""
You are a professional HR specialist.

A company is hiring for the following job:
---
{job_description}
---

You have up to 10 resumes. Each resume is in plain text format and labeled as Candidate 1, Candidate 2, etc.

Your task:
1. Analyze and compare all submitted resumes against the job description.
2. Rank them from best fit to least fit (1 = best match).
3. Provide a brief reason (1–2 sentences) for each candidate’s ranking.

Output format:
Ranked List:
1. Candidate X: [Reason]
2. Candidate Y: [Reason]
...

Begin analysis below:
{resumes}
"""
    )
    return LLMChain(llm=llm, prompt=prompt)

llm_chain = init_chain()

# Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        text += f"\n[Error reading PDF: {str(e)}]"
    return text.strip()

# Generate PDF file from content
def generate_pdf(content):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40

    lines = content.split('\n')
    for line in lines:
        if y < 40:
            p.showPage()
            y = height - 40
        p.drawString(40, y, line.strip())
        y -= 15

    p.save()
    buffer.seek(0)
    return buffer

# Run Analysis
if analyze_button:
    if not uploaded_files or len(uploaded_files) > 10:
        st.warning("Please upload between 1 and 10 PDF resumes.")
    elif not job_description.strip():
        st.warning("Please enter a job description.")
    else:
        with st.spinner("🔍 Analyzing resumes..."):
            resume_texts = []
            for idx, file in enumerate(uploaded_files):
                text = extract_text_from_pdf(file)
                resume_texts.append(f"\n\nCandidate {idx+1} Resume:\n{text[:3000]}")  # Optional truncate
            combined_resumes = "\n".join(resume_texts)

            response = llm_chain.invoke({
                "resumes": combined_resumes,
                "job_description": job_description
            })

            st.success("✅ Ranking Complete!")
            st.markdown("---")
            st.subheader("📊 Ranked Candidates")
            st.markdown(response['text'])

            # Generate and offer PDF download
            pdf_data = generate_pdf(response['text'])
            st.download_button(
                label="📥 Download Rankings as PDF",
                data=pdf_data,
                file_name="ranked_resumes.pdf",
                mime="application/pdf",
                use_container_width=True
            )
