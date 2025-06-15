import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

# Load environment variables
load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_API_BASE")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# UI config
st.set_page_config(page_title="AutoRankCV", layout="wide", page_icon="📄")
st.markdown("""
    <h1 style='text-align: center;'>📄 AutoRank CV</h1>
    <h5 style='text-align: center; color: grey;'>AI-powered Resume Ranking with Parameter-Based Scoring</h5>
""", unsafe_allow_html=True)
st.markdown("---")

# Resume number selector
max_files = st.number_input("How many resumes do you want to upload? (up to 100)", min_value=1, max_value=100, value=10, step=1)

# Upload and JD
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_files = st.file_uploader("Upload PDF Resumes", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        if len(uploaded_files) > max_files:
            st.error(f"You selected {max_files} but uploaded {len(uploaded_files)}. Please upload only the number you selected.")
        else:
            st.success(f"Uploaded {len(uploaded_files)} resume(s).")

with col2:
    job_description = st.text_area("Job Description", height=300)

# Analyze button
_, btn_col, _ = st.columns([7, 1.3, 1])
analyze_button = btn_col.button("Analyze", use_container_width=True)

# Initialize LangChain
@st.cache_resource
def init_chain():
    llm = AzureChatOpenAI(
        openai_api_base=api_base,
        openai_api_version=api_version,
        openai_api_key=api_key,
        deployment_name=deployment_name,
        model_name="gpt-4o",
        temperature=0.3,
        max_tokens=16000,
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
1. **Technical Skills**
2. **Relevant Experience**
3. **Education Alignment**
4. **Communication and Presentation**
5. **Overall Fit for the Role**

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

def extract_text_from_pdf(file):
    text = ""
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        text += f"\n[Error reading PDF: {str(e)}]"
    return text.strip()

def generate_pdf_table(summary_lines):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40

    # Parse headers and rows from markdown-like table
    rows = [line.strip('| ').split('|') for line in summary_lines if '|' in line and '---' not in line]
    if not rows:
        c.drawString(40, y, "No table data found.")
        c.save()
        buffer.seek(0)
        return buffer

    table = Table(rows)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 8)
    ]))

    table.wrapOn(c, width, height)
    table.drawOn(c, 30, y - len(rows)*15)
    c.save()
    buffer.seek(0)
    return buffer

# Analysis
if analyze_button:
    if not uploaded_files or len(uploaded_files) > max_files:
        st.warning("Please upload the exact number of resumes you selected.")
    elif not job_description.strip():
        st.warning("Please provide a job description.")
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
            st.subheader("📊 Candidate Evaluation")
            st.markdown(result_text)

            # Extract table for PDF
            table_lines = [line for line in result_text.splitlines() if line.strip().startswith('|')]
            pdf_file = generate_pdf_table(table_lines)

            st.download_button(
                label="📥 Download Ranking Table (PDF)",
                data=pdf_file,
                file_name="ranked_resume_summary.pdf",
                mime="application/pdf",
                use_container_width=True
            )
