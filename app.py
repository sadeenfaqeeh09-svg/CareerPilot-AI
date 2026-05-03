"""
app.py — CareerPilot AI — Main Application Entry Point
=======================================================
CareerPilot AI is a career advisor and mock interview trainer
powered by LLMs and RAG (Retrieval-Augmented Generation).

Architecture:
  ┌─────────────┐    ┌─────────────────┐    ┌──────────────┐
  │  Gradio UI  │───▶│  app.py Logic   │───▶│  LLM API     │
  │  (Blocks)   │    │  (state mgmt)   │    │  (OpenAI /   │
  └─────────────┘    └────────┬────────┘    │   Gemini)    │
                              │             └──────────────┘
                     ┌────────▼────────┐
                     │  rag_utils.py   │  ← CV embeddings
                     │  database.py    │  ← SQLite progress
                     │  llm_utils.py   │  ← LLM abstraction
                     └─────────────────┘

Run: python app.py
"""

import re
import gradio as gr
import pandas as pd
from datetime import datetime

# ── Import our custom modules ──
from llm_utils import get_llm_response, check_api_availability, get_provider_label
from rag_utils import (
    extract_text_from_file,
    chunk_text,
    create_embeddings,
    search_relevant_chunks,
    build_rag_context,
)
from database import init_db, save_session, get_all_sessions, get_summary_stats, clear_all_sessions

# ── Initialize the database on startup ──
init_db()


# ═══════════════════════════════════════════════════════════════
# SECTION 1: CAREER ADVISOR
# ═══════════════════════════════════════════════════════════════

def get_career_advice(major, skills, interests, target_job, experience):
    """
    Generate a personalized career roadmap using the LLM.

    Args:
        major (str): Field of study or educational background
        skills (str): Current technical/soft skills
        interests (str): What the user enjoys doing
        target_job (str): Desired job title or field
        experience (str): Experience level (Student / Junior / Mid / Senior)

    Returns:
        str: Formatted career advice with sections for paths, roadmap, projects, etc.
    """
    # Validate inputs
    missing = [name for name, val in [
        ("Major/Background", major),
        ("Current Skills", skills),
        ("Interests", interests),
        ("Target Job", target_job)
    ] if not val or not val.strip()]

    if missing:
        return f"⚠️ **Please fill in:** {', '.join(missing)}\n\nAll fields are needed for personalized advice."

    # Build a structured prompt for the LLM
    prompt = f"""You are CareerPilot AI, a world-class career advisor with deep knowledge of tech, business, and creative industries.

A user has shared their profile. Provide highly specific, actionable, and encouraging career guidance.

═══ USER PROFILE ═══
• Major / Background: {major}
• Current Skills: {skills}
• Interests & Passions: {interests}
• Target Job / Dream Role: {target_job}
• Experience Level: {experience}

═══ YOUR RESPONSE FORMAT ═══

## 🎯 Best Career Paths
List 3-4 realistic career paths that match this profile. For each, explain WHY it fits.

## 🛠️ Skill Roadmap
Break down skills to learn into:
- **Immediate (0–3 months):** Quick wins
- **Short-term (3–6 months):** Core competencies
- **Long-term (6–12 months):** Advanced mastery

## 💡 Projects to Build
Suggest 4-5 specific portfolio projects that would impress recruiters for {target_job}. Be concrete.

## 📚 Learning Plan
Recommend specific:
- Online courses (with platform names)
- Books
- Communities to join
- YouTube channels or podcasts

## 💼 Suggested Job Titles to Apply For
List 6-8 realistic job titles (from entry-level to target), with brief descriptions.

## 🚀 First 30-Day Action Plan
Give a concrete week-by-week plan to get started immediately.

Be specific to their background. Avoid generic advice."""

    return get_llm_response([{"role": "user", "content": prompt}])


# ═══════════════════════════════════════════════════════════════
# SECTION 2: CV + JOB DESCRIPTION ANALYZER (RAG)
# ═══════════════════════════════════════════════════════════════

def analyze_cv(cv_file, job_description, cv_state):
    """
    RAG-powered CV analysis:
      1. Extract text from uploaded CV
      2. Chunk and embed the text
      3. Retrieve most relevant sections for the job description
      4. Ask LLM to compare CV vs JD with grounded context

    Args:
        cv_file: Gradio file object (gives us a temp file path)
        job_description (str): Pasted job description text
        cv_state (dict): Gradio State — stores CV data across interactions

    Returns:
        tuple: (analysis_markdown, updated_cv_state)
    """
    # ── Input validation ──
    if cv_file is None:
        return "⚠️ **No CV uploaded.** Please upload your CV as a PDF or TXT file.", cv_state

    if not job_description or len(job_description.strip()) < 30:
        return "⚠️ **Job description too short.** Please paste the full job description (30+ characters).", cv_state

    # ── Step 1: Extract text from CV ──
    cv_text = extract_text_from_file(cv_file)

    if cv_text.startswith("Error"):
        return f"❌ **CV Extraction Failed:** {cv_text}", cv_state

    if len(cv_text.strip()) < 50:
        return "❌ **CV appears empty.** The file was read but contains very little text. Try a different format.", cv_state

    # ── Step 2: Chunk + Embed (RAG preparation) ──
    chunks = chunk_text(cv_text, chunk_size=300, overlap=50)
    embeddings = create_embeddings(chunks)

    # Save to state so other tabs can reuse CV data
    updated_state = {
        "text": cv_text,
        "chunks": chunks,
        "embeddings": embeddings,
        "filename": getattr(cv_file, 'name', 'uploaded_cv') if hasattr(cv_file, 'name') else str(cv_file)
    }

    # ── Step 3: Retrieve relevant CV sections via RAG ──
    # We query using the job description to find matching parts of the CV
    rag_context = build_rag_context(
        query=job_description,
        chunks=chunks,
        embeddings=embeddings,
        top_k=5
    )

    # ── Step 4: LLM-powered comparison ──
    prompt = f"""You are an expert ATS (Applicant Tracking System) analyst and career coach.

⚠️ IMPORTANT: Base your analysis ONLY on the actual CV content below. 
Do NOT invent skills or experiences that aren't present. 
If something is missing, say it's missing.

═══ CV RELEVANT SECTIONS (Retrieved via RAG) ═══
{rag_context}

═══ FULL CV TEXT (first 2500 chars) ═══
{cv_text[:2500]}

═══ JOB DESCRIPTION ═══
{job_description[:2000]}

═══ ANALYSIS FORMAT ═══

## 📊 Match Score
Estimate the match percentage (0–100%) with a one-sentence justification.
Format: **Match: XX%** — [reason]

## ✅ Strong Skills Found in CV
List skills from the CV that match the job requirements. Reference specific CV content.

## ❌ Missing Skills & Gaps
List required skills/experience NOT found in the CV. Be honest.

## 📝 CV Improvement Suggestions
Give 4–6 specific, actionable suggestions to improve the CV for THIS role.
(e.g., "Add a Projects section showcasing X", "Quantify your impact in role Y")

## 🎤 Likely Interview Questions
List 6–7 questions the interviewer will likely ask, based on both the CV and JD.

## ⚡ Quick Wins
2–3 things the candidate can do THIS WEEK to strengthen their application.

Note: This analysis is grounded in your uploaded CV content."""

    result = get_llm_response([{"role": "user", "content": prompt}])
    return result, updated_state


# ═══════════════════════════════════════════════════════════════
# SECTION 3: MOCK INTERVIEW
# ═══════════════════════════════════════════════════════════════

INTERVIEWER_PROMPTS = {
    "HR Interview": """You are a professional HR interviewer. Your style:
- Warm but professional
- Focus on: motivation, culture fit, career goals, salary expectations, work style
- Ask questions like: "Why this company?", "Where do you see yourself in 5 years?"
- Probe soft skills: communication, teamwork, adaptability""",

    "Technical Interview": """You are a senior technical interviewer. Your style:
- Precise and analytical
- Focus on: technical depth, problem-solving, system design, algorithms, domain knowledge
- Ask coding questions, design challenges, "explain how X works" questions
- Follow up on vague answers: "Can you give me more detail on that?"
- Get progressively harder each round""",

    "Behavioral Interview": """You are a behavioral interview specialist. Your style:
- Structured and methodical
- Use STAR-format questions (Situation, Task, Action, Result)
- Focus on: past experiences, conflict resolution, leadership, failures & learning
- Ask: "Tell me about a time when...", "Give me an example of..."
- If answer lacks STAR structure, ask: "What was the specific outcome?"  """,

    "Stress Interview": """You are a stress test interviewer. Your style:
- Challenging but not hostile
- Interrupt occasionally: "Let me stop you there — what if your approach failed?"
- Ask unexpected: "Sell me this pen", "Estimate the number of piano tuners in Chicago"
- Question reasoning: "Why should I believe that?", "What's the weakness in your answer?"
- Test composure, critical thinking, quick adaptation"""
}


def start_interview(interview_type, target_role, interview_state):
    """
    Initialize a new interview session and ask the first question.

    Args:
        interview_type (str): Type selected by user
        target_role (str): Job title they're preparing for
        interview_state (dict): Gradio State for interview data

    Returns:
        tuple: (chat_history, status_message, updated_interview_state)
    """
    if not target_role or not target_role.strip():
        return [], "⚠️ Please enter a **Target Role** before starting.", interview_state

    # Reset interview state for fresh session
    new_state = {
        "type": interview_type,
        "target_role": target_role.strip(),
        "messages": [],        # Full conversation for LLM context
        "scores": [],          # Numeric scores per answer
        "weaknesses": [],      # Weakness notes for final report
        "question_count": 0,
        "started": True
    }

    system_prompt = f"""{INTERVIEWER_PROMPTS[interview_type]}

You are interviewing for the position: {target_role}
Rules:
- Ask ONE question at a time
- Never answer your own questions
- When evaluating answers, use this EXACT format:
  📊 **Score: X/10** — [one-line reason]
  🔍 **Feedback:** [2-3 sentences on what was good/weak]
  ✨ **Ideal Answer:** [Rewritten model answer]
  ❓ **Next Question:** [Your next question]
- Start with a greeting and first question only"""

    # First turn: just ask the opening question
    opening_prompt = (
        f"Start the {interview_type} for a {target_role} candidate. "
        "Greet them professionally and ask your first question. Nothing else."
    )

    new_state["messages"].append({"role": "user", "content": opening_prompt})
    bot_response = get_llm_response(new_state["messages"], system_prompt=system_prompt)
    new_state["messages"].append({"role": "assistant", "content": bot_response})
    new_state["question_count"] = 1

    # Gradio Chatbot format: list of [user_msg, bot_msg] pairs
    chat_history = [[None, bot_response]]

    status = f"✅ **{interview_type}** started for **{target_role}**. Answer the question above!"
    return chat_history, status, new_state


def send_interview_answer(user_answer, chat_history, interview_state):
    """
    Process the user's answer and return evaluation + next question.

    Flow per turn:
      1. Receive user answer
      2. Build evaluation prompt (with full history for context)
      3. Get LLM to score + give feedback + ask next question
      4. Extract score and weakness for the final report
      5. Update state and chat history

    Args:
        user_answer (str): The candidate's answer
        chat_history (list): Current chatbot display history
        interview_state (dict): Current interview state

    Returns:
        tuple: (updated_chat_history, cleared_input, updated_state)
    """
    # Guard: check if interview was started
    if not interview_state or not interview_state.get("started"):
        msg = "⚠️ Please **Start Interview** first using the button above."
        chat_history = chat_history or []
        chat_history.append([user_answer, msg])
        return chat_history, "", interview_state

    if not user_answer or not user_answer.strip():
        return chat_history, "", interview_state

    # Record user's answer in messages
    interview_state["messages"].append({
        "role": "user",
        "content": user_answer
    })

    system_prompt = f"""{INTERVIEWER_PROMPTS[interview_state['type']]}
You are interviewing for: {interview_state['target_role']}
This is question {interview_state['question_count'] + 1}.

ALWAYS format your response exactly like this:
📊 **Score: X/10** — [one-line justification]
🔍 **Feedback:** [what was strong and what was weak — 2-3 sentences]
✨ **Ideal Answer:** [A model answer they should aim for]
❓ **Next Question:** [Your next question — make it progressively harder]"""

    # Get LLM evaluation
    bot_response = get_llm_response(
        interview_state["messages"],
        system_prompt=system_prompt
    )

    # Record bot response
    interview_state["messages"].append({
        "role": "assistant",
        "content": bot_response
    })

    # ── Extract score for progress tracking ──
    score = _extract_score(bot_response)
    if score is not None:
        interview_state["scores"].append(score)

    # ── Extract weakness snippet ──
    weakness = _extract_weakness(bot_response)
    if weakness:
        interview_state["weaknesses"].append(weakness)

    interview_state["question_count"] += 1

    # Update display
    chat_history.append([user_answer, bot_response])

    return chat_history, "", interview_state


def _extract_score(text: str):
    """Parse 'Score: X/10' from LLM response."""
    match = re.search(r'Score:\s*(\d+)/10', text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        return max(1, min(10, score))   # Clamp to 1–10
    return None


def _extract_weakness(text: str):
    """Extract the feedback section as a short weakness note."""
    match = re.search(r'(?:Feedback:|🔍)[^\n]*\n(.*?)(?:\n|✨|❓)', text, re.DOTALL)
    if match:
        return match.group(1).strip()[:200]
    return None


# ═══════════════════════════════════════════════════════════════
# SECTION 4: FEEDBACK REPORT
# ═══════════════════════════════════════════════════════════════

def generate_feedback_report(interview_state):
    """
    Generate a comprehensive post-interview feedback report.
    Also saves the session to SQLite for progress tracking.

    Args:
        interview_state (dict): Completed interview state with scores/weaknesses

    Returns:
        str: Full markdown report
    """
    # Guard: need at least 2 scored answers
    if not interview_state or not interview_state.get("scores"):
        return (
            "⚠️ **No interview data found.**\n\n"
            "Please complete at least **2 questions** in the Mock Interview tab first,\n"
            "then come back to generate your report."
        )

    scores = interview_state["scores"]
    avg_score = sum(scores) / len(scores)
    interview_type = interview_state.get("type", "Unknown")
    target_role = interview_state.get("target_role", "Unknown Role")
    weaknesses = interview_state.get("weaknesses", [])

    # Grade mapping
    grade = (
        "A+ 🏆" if avg_score >= 9 else
        "A  ✅" if avg_score >= 8 else
        "B+ 🙂" if avg_score >= 7 else
        "B  📚" if avg_score >= 6 else
        "C  ⚠️" if avg_score >= 5 else
        "D  🔴"
    )

    # Score bar visualization (text-based)
    filled = int(avg_score)
    score_bar = "█" * filled + "░" * (10 - filled)

    # Build conversation summary for context
    history_summary = ""
    messages = interview_state.get("messages", [])
    # Skip the very first message (our setup prompt) and take last 12
    for msg in messages[1:][-12:]:
        role = "🎤 Candidate" if msg["role"] == "user" else "🤖 Interviewer"
        snippet = msg["content"][:400].replace('\n', ' ')
        history_summary += f"{role}: {snippet}...\n\n"

    prompt = f"""You are a professional interview coach writing a post-interview debrief report.

═══ SESSION DATA ═══
• Interview Type: {interview_type}
• Target Role: {target_role}
• Questions Answered: {len(scores)}
• Scores Per Question: {scores}
• Average Score: {avg_score:.1f}/10 (Grade: {grade})
• Detected Weaknesses: {weaknesses[:4] if weaknesses else ["Not enough data"]}

═══ CONVERSATION SUMMARY ═══
{history_summary[:2000]}

═══ WRITE A DETAILED REPORT ═══

## 🏆 Overall Performance
- Grade: {grade}
- Score: {avg_score:.1f}/10 [{score_bar}]
- Write 2-3 sentences summarizing overall performance honestly.

## 💪 Key Strengths
List 3-4 specific strengths observed during the interview with brief examples.

## ⚠️ Repeated Weaknesses
Identify 2-3 recurring patterns of weakness. Be constructive and specific.

## 🗣️ Communication Assessment
Evaluate: clarity, structure (STAR), confidence, conciseness, use of examples.

## 📋 2-Week Improvement Plan
Give a day-by-day action plan for improvement. Be specific:
- Days 1-3: ...
- Days 4-7: ...
- Days 8-14: ...

## 🎯 Before Your Next Interview
List 3-5 concrete, specific actions to take before interviewing for {target_role}.

## 📌 Resources to Study
Recommend 3-4 specific resources (books, courses, videos) relevant to this interview type and role."""

    report = get_llm_response([{"role": "user", "content": prompt}])

    # ── Save to SQLite ──
    tip = f"Practice STAR method; study {target_role} domain knowledge" if avg_score < 7 else "Push for advanced scenarios"
    weakness_note = "; ".join(weaknesses[:2]) if weaknesses else "See report for details"

    save_session(
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        interview_type=interview_type,
        target_role=target_role,
        score=round(avg_score, 1),
        weakness=weakness_note[:300],
        improvement_tip=tip[:300]
    )

    return report


# ═══════════════════════════════════════════════════════════════
# SECTION 5: PROGRESS HISTORY
# ═══════════════════════════════════════════════════════════════

def load_progress_history():
    """
    Load all sessions from SQLite and return as a pandas DataFrame.

    Returns:
        tuple: (DataFrame or None, summary_markdown)
    """
    sessions = get_all_sessions()
    stats = get_summary_stats()

    # Build stats summary
    summary = f"""### 📊 Overall Statistics
| Metric | Value |
|--------|-------|
| Total Sessions | {stats['total_sessions']} |
| Average Score | {stats['avg_score']}/10 |
| Best Score | {stats['best_score']}/10 |
| Most Practiced | {stats['most_practiced_type']} |
"""

    if not sessions:
        return None, summary + "\n\n📭 **No sessions yet.** Complete a mock interview to see your progress here."

    # Build DataFrame
    df = pd.DataFrame(sessions, columns=[
        "ID", "Date", "Interview Type", "Target Role",
        "Score /10", "Weakness", "Improvement Tip"
    ])

    # Add emoji grade column
    df["Grade"] = df["Score /10"].apply(lambda s:
        "A+ 🏆" if s >= 9 else "A ✅" if s >= 8 else "B+ 🙂" if s >= 7 else
        "B 📚" if s >= 6 else "C ⚠️" if s >= 5 else "D 🔴"
    )

    # Reorder columns for display
    df = df[["ID", "Date", "Interview Type", "Target Role", "Score /10", "Grade", "Weakness", "Improvement Tip"]]

    return df, summary


def clear_history_confirm():
    """Clear all progress history from database."""
    deleted = clear_all_sessions()
    return None, f"🗑️ **Cleared {deleted} session(s).** Your history has been reset."


# ═══════════════════════════════════════════════════════════════
# GRADIO UI — Build the interface
# ═══════════════════════════════════════════════════════════════

# Custom CSS for a polished look
CUSTOM_CSS = """
/* Main container */
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    font-family: 'Inter', sans-serif !important;
}

/* Tab styling */
.tab-nav button {
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 10px 18px !important;
}

/* Card-like sections */
.gr-group {
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    padding: 16px !important;
}

/* Status message */
.status-msg {
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 14px;
}

/* Score color */
strong { color: #2563eb; }

/* Chatbot bubble */
.message.bot { background: #f0f9ff !important; }
.message.user { background: #f0fdf4 !important; }
"""

# ── Check API on startup ──
_provider, _api_status = check_api_availability()
_provider_label = get_provider_label()


def build_ui():
    """Assemble the complete Gradio Blocks UI."""

    with gr.Blocks(title="CareerPilot AI 🚀") as app:

        # ── Shared State ──
        # cv_state: holds extracted CV text, chunks, and embeddings
        cv_state = gr.State({})

        # interview_state: holds ongoing interview conversation and scores
        interview_state = gr.State({})

        # ══════════════════════════════════════════
        # HEADER
        # ══════════════════════════════════════════
        gr.Markdown(f"""
# 🚀 CareerPilot AI
### Your AI-Powered Career Advisor & Interview Trainer

{_api_status}  |  **Model:** {_provider_label}

---
""")

        # ══════════════════════════════════════════
        # TABS
        # ══════════════════════════════════════════
        with gr.Tabs():

            # ─────────────────────────────────────
            # TAB 1: HOME
            # ─────────────────────────────────────
            with gr.TabItem("🏠 Home"):
                gr.Markdown("""
## Welcome to CareerPilot AI! 👋

CareerPilot AI helps you **land your dream job** with AI-powered career guidance and interview practice.

---

### 🗺️ What Can CareerPilot Do?

| Feature | Tab | What It Does |
|---------|-----|--------------|
| 🎯 Career Advisor | `Career Advisor` | Get a personalized roadmap, skill plan & job titles |
| 📄 CV Analyzer | `CV Analyzer` | Upload your CV + paste a job description → get match % |
| 🎤 Mock Interview | `Mock Interview` | Practice HR, Technical, Behavioral, or Stress interviews |
| 📋 Feedback Report | `Feedback Report` | Get a detailed performance report after your interview |
| 📊 Progress History | `Progress History` | Track your scores over time with SQLite |

---

### 🔧 Tech Stack (For Students)
- **Gradio Blocks** — Multi-tab UI with state management
- **RAG Pipeline** — sentence-transformers + cosine similarity → grounded CV analysis
- **LLM** — OpenAI GPT-4o-mini or Google Gemini 1.5 Flash
- **SQLite** — Persistent progress tracking (no server needed)
- **Python** — Fully local, runs on any machine

---

### 🚦 Quick Start
1. Set your API key in `.env` (copy `.env.example`)
2. Go to **Career Advisor** → fill in your profile
3. Upload your CV in **CV Analyzer** → paste a job description
4. Start a **Mock Interview** → pick your type and role
5. After 5+ questions, view your **Feedback Report**

---
*Built for AI/RAG Bootcamp | Open source & educational*
""")

            # ─────────────────────────────────────
            # TAB 2: CAREER ADVISOR
            # ─────────────────────────────────────
            with gr.TabItem("🎯 Career Advisor"):
                gr.Markdown("### 🎯 Personalized Career Roadmap\nFill in your profile and get AI-powered career advice.")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📋 Your Profile")

                        ca_major = gr.Textbox(
                            label="Major / Educational Background",
                            placeholder="e.g. Computer Science, Business Administration, Self-taught...",
                            lines=2
                        )
                        ca_skills = gr.Textbox(
                            label="Current Skills",
                            placeholder="e.g. Python, SQL, Excel, data analysis, communication...",
                            lines=3
                        )
                        ca_interests = gr.Textbox(
                            label="Interests & Passions",
                            placeholder="e.g. AI, startups, healthcare, building products...",
                            lines=2
                        )
                        ca_target = gr.Textbox(
                            label="Target Job / Dream Role",
                            placeholder="e.g. Machine Learning Engineer, Product Manager...",
                            lines=2
                        )
                        ca_experience = gr.Dropdown(
                            label="Experience Level",
                            choices=["Student / No experience", "Junior (0–2 years)", "Mid-level (2–5 years)", "Senior (5+ years)"],
                            value="Student / No experience"
                        )

                        ca_btn = gr.Button("🚀 Get Career Advice", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        gr.Markdown("#### 💡 Your Personalized Advice")
                        ca_output = gr.Markdown(
                            value="*Your career roadmap will appear here...*",
                            label="Career Advice"
                        )

                # Connect button
                ca_btn.click(
                    fn=get_career_advice,
                    inputs=[ca_major, ca_skills, ca_interests, ca_target, ca_experience],
                    outputs=ca_output
                )

            # ─────────────────────────────────────
            # TAB 3: CV ANALYZER (RAG)
            # ─────────────────────────────────────
            with gr.TabItem("📄 CV Analyzer"):
                gr.Markdown("""### 📄 CV + Job Description Analyzer (RAG)
Upload your CV and paste a job description. The AI will compare them using **RAG** — 
retrieving the most relevant sections of your CV before analyzing.""")

                with gr.Row():
                    with gr.Column(scale=1):
                        cv_file = gr.File(
                            label="📎 Upload Your CV",
                            file_types=[".pdf", ".txt", ".docx"],
                            type="filepath"
                        )
                        cv_jd = gr.Textbox(
                            label="📋 Paste Job Description",
                            placeholder="Paste the full job description here...\nThe more complete, the better the analysis.",
                            lines=15
                        )
                        cv_btn = gr.Button("🔍 Analyze My CV", variant="primary", size="lg")

                        gr.Markdown("""
> **How RAG works here:**
> 1. Your CV is split into ~300-word chunks
> 2. Each chunk is converted to a vector embedding
> 3. The job description is used as a search query
> 4. Top 5 most relevant CV sections are retrieved
> 5. LLM analyzes retrieved sections vs JD
>
> This ensures the LLM focuses on the most relevant parts of your CV!
""")

                    with gr.Column(scale=2):
                        cv_output = gr.Markdown(
                            value="*Upload your CV and paste a job description to see the analysis...*"
                        )

                cv_btn.click(
                    fn=analyze_cv,
                    inputs=[cv_file, cv_jd, cv_state],
                    outputs=[cv_output, cv_state]
                )

            # ─────────────────────────────────────
            # TAB 4: MOCK INTERVIEW
            # ─────────────────────────────────────
            with gr.TabItem("🎤 Mock Interview"):
                gr.Markdown("""### 🎤 AI Mock Interview Trainer
Select your interview type and target role, then start practicing.
The AI will ask one question at a time, evaluate your answers, and adapt based on your responses.""")

                with gr.Row():
                    iv_type = gr.Dropdown(
                        label="Interview Type",
                        choices=["HR Interview", "Technical Interview", "Behavioral Interview", "Stress Interview"],
                        value="HR Interview",
                        scale=2
                    )
                    iv_role = gr.Textbox(
                        label="Target Role",
                        placeholder="e.g. Data Scientist, Software Engineer, Product Manager...",
                        scale=3
                    )
                    iv_start_btn = gr.Button("▶️ Start Interview", variant="primary", scale=1)

                iv_status = gr.Markdown("*Press 'Start Interview' to begin...*")

                # The main chat interface
                iv_chatbot = gr.Chatbot(
                    label="Interview Conversation",
                    height=480,
                )

                with gr.Row():
                    iv_input = gr.Textbox(
                        label="Your Answer",
                        placeholder="Type your answer here and press Enter or click Send...",
                        lines=3,
                        scale=5
                    )
                    iv_send_btn = gr.Button("📤 Send", variant="primary", scale=1)

                gr.Markdown("""
> 💡 **Tips:**
> - Answer naturally as you would in a real interview
> - The AI will score each answer (1–10), give feedback, rewrite an ideal answer, and ask the next question
> - After 5+ questions, go to **Feedback Report** for your full assessment
""")

                # Connect Start button
                iv_start_btn.click(
                    fn=start_interview,
                    inputs=[iv_type, iv_role, interview_state],
                    outputs=[iv_chatbot, iv_status, interview_state]
                )

                # Connect Send button
                iv_send_btn.click(
                    fn=send_interview_answer,
                    inputs=[iv_input, iv_chatbot, interview_state],
                    outputs=[iv_chatbot, iv_input, interview_state]
                )

                # Also allow pressing Enter in the text box
                iv_input.submit(
                    fn=send_interview_answer,
                    inputs=[iv_input, iv_chatbot, interview_state],
                    outputs=[iv_chatbot, iv_input, interview_state]
                )

            # ─────────────────────────────────────
            # TAB 5: FEEDBACK REPORT
            # ─────────────────────────────────────
            with gr.TabItem("📋 Feedback Report"):
                gr.Markdown("""### 📋 Interview Feedback Report
After completing your mock interview (5+ questions), generate a comprehensive feedback report.
Your session will also be **automatically saved** to your progress history.""")

                with gr.Row():
                    fb_btn = gr.Button("📊 Generate Feedback Report", variant="primary", size="lg")

                gr.Markdown("> ⚠️ **Note:** You must complete at least 2 answered questions in Mock Interview first.")

                fb_output = gr.Markdown(
                    value="*Your feedback report will appear here after completing a mock interview...*"
                )

                fb_btn.click(
                    fn=generate_feedback_report,
                    inputs=[interview_state],
                    outputs=fb_output
                )

            # ─────────────────────────────────────
            # TAB 6: PROGRESS HISTORY
            # ─────────────────────────────────────
            with gr.TabItem("📊 Progress History"):
                gr.Markdown("""### 📊 Interview Progress History
Track your improvement over time. Each completed interview report is saved here.""")

                with gr.Row():
                    ph_refresh_btn = gr.Button("🔄 Refresh History", variant="primary")
                    ph_clear_btn = gr.Button("🗑️ Clear All History", variant="stop")

                ph_stats = gr.Markdown("*Click Refresh to load your history...*")

                ph_table = gr.DataFrame(
                    label="Interview Sessions",
                    interactive=False,
                    wrap=True
                )

                # Connect buttons
                ph_refresh_btn.click(
                    fn=load_progress_history,
                    inputs=[],
                    outputs=[ph_table, ph_stats]
                )

                ph_clear_btn.click(
                    fn=clear_history_confirm,
                    inputs=[],
                    outputs=[ph_table, ph_stats]
                )

                gr.Markdown("""
---
> 💡 **How progress is saved:**
> Each time you click "Generate Feedback Report," your session (date, type, role, score, and weakness) 
> is saved to a local SQLite database (`careerpilot_progress.db`).
> This file persists between app restarts.
""")

    return app


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  CareerPilot AI — Starting Up")
    print("=" * 60)

    # Check API availability
    provider, status = check_api_availability()
    print(f"  LLM Provider: {status}")

    if not provider:
        print("\n  ⚠️  WARNING: No API key found!")
        print("  Copy .env.example to .env and add your API key.")
        print("  The app will launch but LLM features won't work.\n")

    # Build and launch the app
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="cyan",
            neutral_hue="slate",
        ),
        css=CUSTOM_CSS,
    )