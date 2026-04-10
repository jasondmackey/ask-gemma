import datetime
import urllib.request
import json
from zoneinfo import ZoneInfo
from ollama import chat

_FALLBACK_TIMEZONE = "America/Denver"
_FALLBACK_LOCATION = "Superior, Colorado, USA"

def get_location_info():
    """Fetch city, region, country, and timezone via ipinfo.io."""
    try:
        with urllib.request.urlopen("https://ipinfo.io/json", timeout=5) as resp:
            data = json.loads(resp.read())
        city     = data.get("city", "")
        region   = data.get("region", "")
        country  = data.get("country", "")
        timezone = data.get("timezone") or _FALLBACK_TIMEZONE
        location = ", ".join(part for part in (city, region, country) if part)
        return location or _FALLBACK_LOCATION, timezone
    except Exception:
        return _FALLBACK_LOCATION, _FALLBACK_TIMEZONE

def get_system_prompt():
    location, timezone = get_location_info()
    tz = ZoneInfo(timezone)
    now = datetime.datetime.now(tz=tz)
    tz_name = now.strftime("%Z")  # e.g. "MDT" or "MST" automatically
    return (
        f"Today is {now.strftime('%A, %B %d, %Y')}. "
        f"Current time is {now.strftime('%I:%M %p')} {tz_name}. "
        f"Location: {location}.\n\n"
        "You are a helpful assistant. Always use the above date, time, and location "
        "when the user asks anything time- or location-related."
    )

def ask(user_message: str) -> str:
    response = chat(
        model="gemma4:31b",           # or whatever your model tag is in ollama list
        messages=[
            {"role": "system",  "content": get_system_prompt()},
            {"role": "user",    "content": user_message},
        ]
    )
    return response.message.content

def chat_loop():
    """Interactive chat with Gemma. Time, date, and location are injected at session start."""
    location, timezone = get_location_info()
    tz = ZoneInfo(timezone)
    now = datetime.datetime.now(tz=tz)
    print(f"[session] {now.strftime('%A, %B %d, %Y  %I:%M %p %Z')} — {location}")
    print("Type 'exit' or press Ctrl+C to quit.\n")

    messages = [{"role": "system", "content": get_system_prompt()}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})
        response = chat(model="gemma4:31b", messages=messages)
        reply = response.message.content
        messages.append({"role": "assistant", "content": reply})
        print(f"Gemma: {reply}\n")

if __name__ == "__main__":
    chat_loop()
