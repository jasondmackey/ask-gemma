"""Gradio web interface for ask-gemma.

Injects current date, time, and location into every session's system prompt,
then streams responses from Gemma via Ollama.

Usage:
    python3 app.py
Then open http://localhost:7860 in your browser.
"""
import datetime
import sys
import gradio as gr
from zoneinfo import ZoneInfo
from ollama import chat as ollama_chat

# Allow running from any directory
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ask_gemma import get_location_info, get_system_prompt

MODEL = "gemma4:31b"


def _session_header() -> str:
    location, timezone = get_location_info()
    tz = ZoneInfo(timezone)
    now = datetime.datetime.now(tz=tz)
    return (
        f"**{now.strftime('%A, %B %d, %Y')}** &nbsp;·&nbsp; "
        f"**{now.strftime('%I:%M %p %Z')}** &nbsp;·&nbsp; "
        f"📍 {location}"
    )


def respond(message: str, history: list) -> str:
    """Stream a reply from Gemma, maintaining full conversation history."""
    # Build message list: system prompt + prior turns + new user message
    messages = [{"role": "system", "content": get_system_prompt()}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})

    stream = ollama_chat(model=MODEL, messages=messages, stream=True)
    partial = ""
    for chunk in stream:
        partial += chunk.message.content or ""
        yield partial


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Ask Gemma", fill_height=True) as demo:
    gr.Markdown("# 🦙 Ask Gemma")
    session_info = gr.Markdown(_session_header())

    gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(
            height=520,
            show_label=False,
            placeholder="Ask me anything — I know where and when you are.",
        ),
        textbox=gr.Textbox(
            placeholder="Type your message…",
            container=False,
            autofocus=True,
            submit_btn="Send",
        ),
        examples=[
            "What day of the week is it?",
            "How many days until the end of the month?",
            "What city am I in?",
            "What's the weather typically like here this time of year?",
        ],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
