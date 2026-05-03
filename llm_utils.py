"""
llm_utils.py — LLM Abstraction Layer for CareerPilot AI
========================================================
Supports two LLM providers:
  1. OpenAI (GPT-4o-mini) — used if OPENAI_API_KEY is set
  2. Google Gemini (gemini-1.5-flash) — used if GEMINI_API_KEY is set

If neither key is set, all calls return a friendly error message.

Usage:
    from llm_utils import get_llm_response, check_api_availability

    response = get_llm_response(
        messages=[{"role": "user", "content": "Hello!"}],
        system_prompt="You are a helpful assistant."
    )
"""

import os
from dotenv import load_dotenv

# Load keys from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()


def check_api_availability():
    """
    Check which LLM provider is available.

    Returns:
        tuple: (provider_name: str | None, status_message: str)
    """
    if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"):
        return "openai", "✅ Using OpenAI (GPT-4o-mini)"
    elif GEMINI_API_KEY and len(GEMINI_API_KEY) > 10:
        return "gemini", "✅ Using Google Gemini (gemini-1.5-flash)"
    else:
        return None, (
            "❌ **No API key found!**\n\n"
            "Please create a `.env` file with either:\n"
            "```\nOPENAI_API_KEY=sk-your-key-here\n```\n"
            "or\n"
            "```\nGEMINI_API_KEY=your-gemini-key\n```\n\n"
            "See `.env.example` for the template."
        )


def get_llm_response(messages, system_prompt=None):
    """
    Send messages to the available LLM and return the response text.

    Args:
        messages (list): List of dicts with "role" and "content" keys.
                         Roles: "user" or "assistant"
        system_prompt (str, optional): System-level instructions for the model.

    Returns:
        str: The model's text response, or an error message.

    Example:
        messages = [{"role": "user", "content": "What is machine learning?"}]
        response = get_llm_response(messages, system_prompt="You are a teacher.")
    """
    provider, status = check_api_availability()

    if provider is None:
        return status  # Return the error message directly

    try:
        if provider == "openai":
            return _call_openai(messages, system_prompt)
        elif provider == "gemini":
            return _call_gemini(messages, system_prompt)
    except Exception as e:
        # Surface the error clearly so the student can debug
        return f"❌ **LLM Error ({provider}):** {str(e)}\n\nCheck your API key and internet connection."


# ──────────────────────────────────────────────
# Private helpers — one per provider
# ──────────────────────────────────────────────

def _call_openai(messages, system_prompt=None):
    """
    Call OpenAI Chat Completions API.
    Model: gpt-4o-mini (cheap, fast, capable)
    """
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Build the full message list, prepending system prompt if given
    formatted = []
    if system_prompt:
        formatted.append({"role": "system", "content": system_prompt})
    formatted.extend(messages)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=formatted,
        temperature=0.7,        # Balanced creativity
        max_tokens=1500,        # Enough for detailed answers
    )

    return response.choices[0].message.content


def _call_gemini(messages, system_prompt=None):
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)



    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_prompt or "You are CareerPilot AI, a professional career advisor and interview trainer.",
        generation_config={"temperature": 0.7, "max_output_tokens": 1500}
    )

    chat = model.start_chat()

    last_message = messages[-1]["content"]
    response = chat.send_message(last_message)

    return response.text

    """
    Call Google Gemini API.
    Model: gemini-1.5-flash (free tier available, fast)
    """
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)

    # Configure system instruction
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system_prompt or "You are CareerPilot AI, a professional career advisor and interview trainer.",
        generation_config={"temperature": 0.7, "max_output_tokens": 1500}
    )

    # Convert messages to Gemini chat history format
    # Gemini uses "user" and "model" roles (not "assistant")
    history = []
    for msg in messages[:-1]:   # All except last (we send that as new message)
        role = "user" if msg["role"] == "user" else "model"
        history.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })

    chat = model.start_chat(history=history)

    # Send the final message
    last_message = messages[-1]["content"]
    response = chat.send_message(last_message)

    return response.text


def get_provider_label():
    """Returns a short display label for the active provider."""
    provider, _ = check_api_availability()
    labels = {
        "openai": "🤖 GPT-4o-mini",
        "gemini": "✨ Gemini 1.5 Flash",
        None: "❌ No Provider"
    }
    return labels.get(provider, "Unknown")
