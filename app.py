"""Gradio web interface for ask-gemma.

Injects current date, time, and location into every session's system prompt,
then streams responses from Gemma via Ollama.
Supports attaching text files and PDFs — their contents are injected into the prompt.

Usage:
    python3 app.py
Then open http://localhost:7860 in your browser.
"""
import datetime
import os
import sys
import gradio as gr
from zoneinfo import ZoneInfo
from ollama import chat as ollama_chat

# Allow running from any directory
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ask_gemma import get_location_info, get_system_prompt, get_weather, _get

MODEL = "gemma4:31b"

# ---------------------------------------------------------------------------
# Browser GPS geolocation
# ---------------------------------------------------------------------------

_GEOLOCATION_JS = """
async () => {
    return new Promise(resolve => {
        if (!navigator.geolocation) { resolve([null, null]); return; }
        navigator.geolocation.getCurrentPosition(
            p => resolve([p.coords.latitude, p.coords.longitude]),
            () => resolve([null, null]),
            { timeout: 10000, enableHighAccuracy: true }
        );
    });
}
"""


def _reverse_geocode(lat, lon) -> str | None:
    """Convert GPS coords to a city string via Nominatim (OpenStreetMap)."""
    if lat is None or lon is None:
        return None
    try:
        data = _get(
            f"https://nominatim.openstreetmap.org/reverse"
            f"?lat={lat}&lon={lon}&format=json&zoom=14"
        )
        addr = data.get("address", {})
        city = (
            addr.get("city") or addr.get("town") or addr.get("village")
            or addr.get("suburb") or addr.get("neighbourhood") or ""
        )
        state   = addr.get("state", "")
        country = addr.get("country_code", "").upper()
        return ", ".join(p for p in (city, state, country) if p) or None
    except Exception:
        return None

# File extensions treated as plain text
TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".json", ".yaml", ".yml", ".toml", ".csv", ".html",
    ".htm", ".css", ".xml", ".sh", ".bash", ".zsh",
    ".go", ".rs", ".java", ".c", ".cpp", ".h", ".rb",
    ".swift", ".kt", ".r", ".sql", ".env", ".cfg", ".ini",
    ".log", ".diff", ".patch", ".prc",
}


def _session_header(loc_info: dict | None = None) -> str:
    location_override = (loc_info or {}).get("name")
    latlon_override   = None
    if loc_info and loc_info.get("lat") is not None:
        latlon_override = (str(loc_info["lat"]), str(loc_info["lon"]))

    location, timezone = get_location_info()
    if location_override:
        location = location_override
    tz = ZoneInfo(timezone)
    now = datetime.datetime.now(tz=tz)
    weather = get_weather(latlon=latlon_override)
    weather_part = f" &nbsp;·&nbsp; 🌤️ {weather}" if weather else ""
    return (
        f"**{now.strftime('%A, %B %d, %Y')}** &nbsp;·&nbsp; "
        f"**{now.strftime('%I:%M %p %Z')}** &nbsp;·&nbsp; "
        f"📍 {location}{weather_part}"
    )


def _on_geolocation(lat, lon):
    """Called on page load with browser GPS coords. Returns (location_state, header)."""
    name = _reverse_geocode(lat, lon)
    loc_info = {"name": name, "lat": lat, "lon": lon} if name else None
    return loc_info, _session_header(loc_info)


def _read_file(file) -> str:
    """Extract text from an uploaded file (plain text or PDF)."""
    if isinstance(file, dict):
        path = file.get("path", "")
        name = file.get("orig_name") or os.path.basename(path)
    else:
        path = str(file)
        name = os.path.basename(path)

    ext = os.path.splitext(name)[1].lower()

    try:
        if ext == ".pdf":
            import pypdf
            reader = pypdf.PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return f"[File: {name}]\n{text.strip()}"
        elif ext in TEXT_EXTENSIONS or ext == "":
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return f"[File: {name}]\n{fh.read().strip()}"
        else:
            return f"[File: {name} — unsupported type '{ext}', skipped]"
    except Exception as exc:
        return f"[File: {name} — could not read: {exc}]"


def _to_text(content) -> str:
    """Normalize any Gradio message format to a plain string, reading attached files."""
    if isinstance(content, dict):
        parts = []
        text = (content.get("text") or "").strip()
        if text:
            parts.append(text)
        for f in content.get("files", []):
            parts.append(_read_file(f))
        return "\n\n".join(parts)
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def _format_response(thinking: str, response: str) -> str:
    """Wrap thinking in a collapsible block above the response."""
    if not thinking:
        return response
    label = "💭 Thinking…" if not response else "💭 Thoughts"
    return f"<details><summary>{label}</summary>\n\n{thinking}\n\n</details>\n\n{response}"


def respond(message, history: list, loc_info: dict | None = None):
    """Stream a reply from Gemma with visible chain-of-thought thinking."""
    location_override = (loc_info or {}).get("name")
    latlon_override   = None
    if loc_info and loc_info.get("lat") is not None:
        latlon_override = (str(loc_info["lat"]), str(loc_info["lon"]))
    messages = [{"role": "system", "content": get_system_prompt(
        location_override=location_override,
        latlon_override=latlon_override,
    )}]
    for turn in history:
        messages.append({"role": turn["role"], "content": _to_text(turn["content"])})
    messages.append({"role": "user", "content": _to_text(message)})

    stream = ollama_chat(model=MODEL, messages=messages, stream=True, think=True)
    thinking = ""
    response = ""
    for chunk in stream:
        thinking += chunk.message.thinking or ""
        response += chunk.message.content or ""
        yield _format_response(thinking, response)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Ask Gemma", fill_height=True) as demo:
    gr.Markdown("# 🦙 Ask Gemma")
    session_info  = gr.Markdown(_session_header())
    location_state = gr.State(None)  # set on load from browser GPS
    _lat_state     = gr.State(None)  # temporary holders for JS → Python routing
    _lon_state     = gr.State(None)
    gr.Markdown(
        "_Attach `.txt`, `.md`, `.py`, `.json`, `.pdf` and more — "
        "file contents are sent to the model as context._",
    )

    # On page load: JS gets GPS coords, Gradio routes them via the State placeholders
    # into _on_geolocation(lat, lon) which returns the resolved location + header.
    demo.load(
        fn=_on_geolocation,
        inputs=[_lat_state, _lon_state],
        outputs=[location_state, session_info],
        js=_GEOLOCATION_JS,
    )

    # Refresh header every 60 s (passes current location state so it stays accurate)
    gr.Timer(value=60).tick(
        fn=_session_header,
        inputs=[location_state],
        outputs=session_info,
    )

    gr.ChatInterface(
        fn=respond,
        additional_inputs=[location_state],
        chatbot=gr.Chatbot(
            height=500,
            show_label=False,
            placeholder="Ask me anything — I know where and when you are.",
            render_markdown=True,
        ),
        textbox=gr.MultimodalTextbox(
            placeholder="Type your message or attach files…",
            container=False,
            autofocus=True,
            file_count="multiple",
            file_types=[".txt", ".md", ".py", ".js", ".ts", ".json",
                        ".yaml", ".yml", ".toml", ".csv", ".html", ".css",
                        ".xml", ".sh", ".go", ".rs", ".java", ".c", ".cpp",
                        ".rb", ".swift", ".sql", ".log", ".pdf", ".prc"],
            submit_btn="Send",
        ),
        # With additional_inputs, examples must be [[multimodal_msg, state_val], ...]
        examples=[
            [{"text": "What day of the week is it?",                    "files": []}, None],
            [{"text": "How many days until the end of the month?",      "files": []}, None],
            [{"text": "What city am I in?",                              "files": []}, None],
            [{"text": "What's the weather like here today?",             "files": []}, None],
        ],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
