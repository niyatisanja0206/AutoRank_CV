# 📄 AutoRank CV

AutoRank CV is an AI-powered resume ranking application built with **Streamlit**, **LangChain**, and **Azure OpenAI GPT-4o**. It allows recruiters and hiring managers to upload multiple resumes (up to 100), input a job description, and receive a ranked list of candidates based on AI analysis. The tool also evaluates resumes across key parameters and provides a downloadable PDF report.


## 🚀 Features

- ✅ Upload **up to 100** resumes in PDF format
- ✅ Paste a job description for a target role
- ✅ Intelligent analysis using **GPT-4o** with **Chain-of-Thought reasoning**
- ✅ Resume evaluation across multiple criteria:
  - Skills Match
  - Experience Relevance
  - Educational Fit
  - Project/Research Alignment
  - Overall Presentation
- ✅ Final ranking with parameter-wise score breakdown
- ✅ **Downloadable PDF report** with formatted table of scores


## 📷 Screenshot

![AutoRank CV Screenshot](docs/screenshot.png) <!-- Add your own screenshot -->


## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) – UI/UX layer
- [LangChain](https://www.langchain.com/) – Prompt orchestration
- [Azure OpenAI](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/) – GPT-4o model for intelligent ranking
- [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) – PDF text extraction
- [ReportLab](https://www.reportlab.com/) – PDF generation for final report


## 🧑‍💻 Getting Started

### Prerequisites

- Python 3.9+
- Azure OpenAI access (GPT-4o deployment)

### Clone the Repo

```bash
git clone https://github.com/yourusername/autorank-cv.git
cd autorank-cv
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Set Environment Variables
Create a .env file with the following:
```bash
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_API_BASE=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-05-01-preview
```

## 🧪 Run the App

```bash
streamlit run app.py
```

## 📌 TODO / Future Improvements

Export to Excel or CSV

Add filtering/sorting by parameter

Display graphs

User authentication for recruiters

Resume parsing enhancements (e.g., Named Entity Recognition)



