import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF for PDF text extraction

# Load environment variables
load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_API_BASE")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Page config
st.set_page_config(page_title="AutoRankCV", layout="wide", page_icon="📄")
st.title("📄 AutoRank CV - Resume Selector using GPT-4o and LangChain")

# Upload resumes
st.header("📂 Upload Resumes (PDFs only)")
uploaded_files = st.file_uploader("Upload up to 10 resume PDFs", type=["pdf"], accept_multiple_files=True)

# Job description input
st.header("🧾 Job Description")
job_description = st.text_area("Paste the job description here", height=300)

# Button
analyze_button = st.button("🚀 Analyze & Rank Resumes")

# LLM setup
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

# Helper: Extract PDF text
def extract_text_from_pdf(file):
    text = ""
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        text += f"\n[Error reading PDF: {str(e)}]"
    return text.strip()

# Handle logic
if analyze_button:
    if not uploaded_files or len(uploaded_files) > 10:
        st.warning("Please upload between 1 and 10 PDF resumes.")
    elif not job_description.strip():
        st.warning("Please provide a job description.")
    else:
        with st.spinner("Analyzing resumes with GPT-4o..."):

            resume_texts = []
            for idx, file in enumerate(uploaded_files):
                text = extract_text_from_pdf(file)
                resume_texts.append(f"\n\nCandidate {idx+1} Resume:\n{text[:3000]}")  # truncate if too long

            combined_resumes = "\n".join(resume_texts)

            response = llm_chain.invoke({
                "resumes": combined_resumes,
                "job_description": job_description
            })

            st.success("✅ Ranking Complete!")
            st.subheader("📊 Ranked Resumes")
            st.markdown(response['text'])
            st.download_button(
                label="Download Rankings",
                data=response['text'],
                file_name="ranked_resumes.txt",
                mime="text/plain"
            )