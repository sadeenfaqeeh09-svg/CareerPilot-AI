# 🚀 CareerPilot AI

> **AI-Powered Career Advisor & Mock Interview Trainer**  
> Built with Python · Gradio · RAG · LLMs · SQLite

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange?logo=gradio)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📖 Project Idea

CareerPilot AI was built for an **AI/RAG Bootcamp** to demonstrate real-world use of:

- **Large Language Models** (OpenAI GPT-4o-mini or Google Gemini 1.5 Flash)
- **RAG (Retrieval-Augmented Generation)** for grounded CV analysis
- **Gradio Blocks** for a multi-tab, production-quality UI
- **SQLite** for persistent progress tracking without a backend

The goal: give students and job-seekers a tool they'd actually use — a career advisor and interview trainer that feels realistic, not generic.

---

## ✨ Features

### 1. 🎯 Career Advisor
- Input your major, skills, interests, target job, and experience level
- Get back: best career paths, skill roadmap, portfolio projects, learning plan, suggested job titles, and a 30-day action plan

### 2. 📄 CV + Job Description Analyzer (RAG)
- Upload CV as PDF or TXT
- Paste any job description
- The RAG pipeline retrieves the **most relevant** CV sections and compares them to the JD
- Returns: match percentage, missing skills, strong skills, CV improvement suggestions, and likely interview questions

### 3. 🎤 Mock Interview Trainer
- **4 interview types:** HR, Technical, Behavioral, Stress
- One question at a time — adapts based on your answers
- Each answer gets: score (1–10), detailed feedback, ideal rewrite, and next harder question
- Feels like a real interview, not a quiz

### 4. 📋 Feedback Report
- After completing an interview, generate a full debrief:
  - Overall score with letter grade
  - Key strengths
  - Repeated weaknesses
  - Communication assessment
  - 2-week improvement plan
  - Resources to study

### 5. 📊 SQLite Progress Tracker
- Every completed interview auto-saves to a local SQLite database
- View history table: date, type, role, score, weakness, improvement tip
- See aggregate stats: average score, best score, most practiced type

---

## 🏗️ Architecture

```
careerpilot/
├── app.py              ← Gradio UI + business logic
├── llm_utils.py        ← LLM abstraction (OpenAI / Gemini)
├── rag_utils.py        ← RAG pipeline (extract → chunk → embed → retrieve)
├── database.py         ← SQLite CRUD operations
├── requirements.txt    ← Python dependencies
├── .env.example        ← API key template
└── README.md           ← This file
```

### Data Flow

```
User uploads CV (PDF/TXT)
        │
        ▼
  extract_text_from_file()         ← PyPDF2 or plain read
        │
        ▼
  chunk_text()                     ← 300-word overlapping chunks
        │
        ▼
  create_embeddings()              ← sentence-transformers (all-MiniLM-L6-v2)
        │                             384-dim vectors, runs locally
        ▼
  search_relevant_chunks(query)    ← cosine similarity, top-k retrieval
        │
        ▼
  LLM prompt + retrieved context   ← grounded, accurate analysis
        │
        ▼
  Gradio UI displays result
```

---

## 🔧 How RAG Works Here

Standard LLM limitation: if you paste a 5-page CV into a prompt, you waste tokens and the model loses focus.

**CareerPilot's RAG solution:**

1. **Chunk** — The CV is split into overlapping 300-word chunks (50-word overlap prevents sentences from being cut off at boundaries)
2. **Embed** — Each chunk is converted to a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2` — a small, free, local model
3. **Index** — Vectors are stored in a numpy array (in-memory, no server needed)
4. **Retrieve** — The job description is embedded, and cosine similarity is computed against all CV chunks; top-5 most similar are retrieved
5. **Augment** — Only the retrieved chunks are injected into the LLM prompt, giving it focused, relevant context

This makes the analysis more accurate and much cheaper in tokens.

---

## 🎨 How Gradio is Used

CareerPilot uses **Gradio Blocks** (not the simple ChatInterface) to build a multi-tab, stateful application.

Key Gradio patterns used:

| Pattern | Where Used |
|---------|-----------|
| `gr.Blocks()` with `gr.Tabs()` | Multi-tab layout |
| `gr.State()` | Store CV embeddings and interview history across function calls |
| `gr.File(type="filepath")` | CV upload (returns temp file path) |
| `gr.Chatbot()` | Mock interview conversation display |
| `gr.DataFrame()` | Progress history table |
| `.click()` and `.submit()` | Button and Enter-key triggers |
| `gr.themes.Soft()` + custom CSS | Polished visual styling |

---

## 📦 Installation

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
# 1. Clone or download the project
git clone https://github.com/your-repo/careerpilot-ai
cd careerpilot-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your API key
cp .env.example .env
# Edit .env and add your OpenAI or Gemini key

# 4. Run the app
python app.py
```

Then open your browser at: **http://localhost:7860**

### API Keys

**Option A — OpenAI (recommended):**
```
OPENAI_API_KEY=sk-your-key-here
```
Get yours at: https://platform.openai.com/api-keys

**Option B — Google Gemini (free tier available):**
```
GEMINI_API_KEY=your-gemini-key
```
Get yours at: https://aistudio.google.com/app/apikey

> If OPENAI_API_KEY is set, it takes priority. If only GEMINI_API_KEY is set, Gemini is used. If neither is set, the app shows a clear error message inside the UI.

---

## 🎬 Demo Flow (Suggested Walkthrough)

1. **Open the app** at http://localhost:7860
2. **Go to Career Advisor tab** → Enter: Major = "Computer Science", Skills = "Python, SQL", Interests = "AI/ML", Target = "Data Scientist", Level = "Junior" → Click "Get Career Advice"
3. **Go to CV Analyzer tab** → Upload a CV PDF → Paste a job description → Click "Analyze My CV" and watch the RAG pipeline in action
4. **Go to Mock Interview tab** → Select "Technical Interview" → Enter "Data Scientist" → Click "Start Interview" → Answer 5+ questions
5. **Go to Feedback Report tab** → Click "Generate Feedback Report"
6. **Go to Progress History tab** → Click "Refresh History" → See your session recorded

---

## 🛠️ Extending the Project (Future Improvements)

| Idea | How |
|------|-----|
| 📧 Email report | Add SMTP or Gmail API to send the feedback report |
| 🗣️ Voice interview | Integrate Whisper (STT) + TTS for audio input/output |
| 📈 Score chart | Add Plotly/Matplotlib chart in Progress History tab |
| 🔐 User accounts | Replace SQLite sessions with user-keyed tables |
| 📝 Resume builder | Add a tab to generate a tailored CV from scratch |
| 🌍 Multi-language | Detect language and respond accordingly |
| 💾 Cloud database | Replace SQLite with Supabase or Firebase |
| 🤝 Multi-agent | Use LangGraph/CrewAI for multi-agent interview panels |

---

## 📁 File Reference

| File | Purpose |
|------|---------|
| `app.py` | Main entry point, all Gradio UI, section logic |
| `llm_utils.py` | LLM abstraction — routes to OpenAI or Gemini |
| `rag_utils.py` | Full RAG pipeline: extract → chunk → embed → retrieve |
| `database.py` | SQLite init, save, query, stats |
| `requirements.txt` | All pip dependencies |
| `.env.example` | Template for API keys |
| `careerpilot_progress.db` | Auto-created SQLite file (gitignore this) |

---

## 🧑‍💻 For Bootcamp Students

### Key concepts to explain in your presentation:

1. **RAG vs Fine-tuning** — RAG is cheaper, updatable, and doesn't hallucinate CV facts. Fine-tuning would bake the CV into model weights permanently.

2. **Why sentence-transformers?** — We need embeddings for RAG. OpenAI's embedding API costs money per call. `all-MiniLM-L6-v2` is free, runs locally, and is excellent for semantic similarity.

3. **Gradio State** — Since Gradio is stateless between function calls (like HTTP), we use `gr.State` to persist objects (CV embeddings, interview history) within a user session.

4. **Prompt engineering** — Look at the prompts in `app.py`. Each one has a specific format instruction to get structured, consistent output from the LLM.

5. **SQLite** — No backend server needed. SQLite is a file-based database perfect for local demos and single-user apps.

---

## 📄 License

MIT — Free to use, modify, and share for educational purposes.

---

*Made with ❤️ for the AI/RAG Bootcamp | Python + Gradio + LLMs*

---

## Sadeen J. Faqih
## Nadeen A. Jaber

