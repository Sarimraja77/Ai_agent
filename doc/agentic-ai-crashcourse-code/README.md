# Agentic AI Crash Course

Welcome to the Agentic AI Crash Course. This tutorial folder contains resources and examples to help you learn the basics of building agentic AI systems.

## Contents

| Folder | Topic |
|--------|-------|
| `1_simple_llm_calling` | Simple LLM API calls |
| `2_health_analysis` | Health data analysis agent |
| `3_vector_db` | Vector databases with ChromaDB |
| `4_rag_basics` | Retrieval-Augmented Generation (RAG) |
| `5_single_agent` | Single agent workflows |
| `6_memory` | Agent memory patterns |
| `7_multimodal` | Multimodal (vision + text) agents |
| `8_guardrails` | Safety and guardrails |
| `9_eval` | Agent evaluation |
| `10_project_shopping_agent` | Project: Shopping agent |
| `11_project_telecom_chatbot` | Project: Telecom RAG chatbot |

## Setup

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### 1. Clone the repository

```bash
git clone <repo-url>
cd agentic-ai-crash-course
```

### 2. Create and activate a virtual environment

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -e .
```

### 3. Configure API keys

Copy `.env.sample` to `.env` and fill in your API keys:

```bash
cp .env.sample .env
```

Then edit `.env`:

```
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
```

- **Google API key**: Get from [Google AI Studio](https://aistudio.google.com/)
- **Groq API key**: Get from [Groq Console](https://console.groq.com/)
- **LangSmith API key**: Get from [LangSmith](https://smith.langchain.com/) (optional, for tracing)

### 4. Run the notebooks

Launch Jupyter and open any notebook from the numbered folders:

```bash
jupyter notebook
```

Or run Streamlit apps (where applicable):

```bash
streamlit run <folder>/app.py
```

## Getting Started

1. Start with `1_simple_llm_calling` to verify your setup.
2. Work through the numbered folders in order.
3. Explore the project folders (`10_*`, `11_*`) to see complete end-to-end examples.

## Notes

This course is designed for quick learning and practical application of agentic AI techniques.

Codebasics Inc all rights reserved