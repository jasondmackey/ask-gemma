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

if __name__ == "__main__":
    # Verify location + timezone resolution before querying the model
    location, timezone = get_location_info()
    tz = ZoneInfo(timezone)
    now = datetime.datetime.now(tz=tz)
    print(f"[context] Location: {location} | Timezone: {timezone} | Local time: {now.strftime('%I:%M %p %Z')}")
    print()

    questions = [
        "What day of the week is it and what time is it?",
        "How many days until the end of the month?",
        "What city and country am I in right now?",
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {ask(q)}")
        print()
