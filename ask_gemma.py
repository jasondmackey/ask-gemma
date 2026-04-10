import datetime
import json
import ssl
import urllib.parse
import urllib.request
import certifi
from zoneinfo import ZoneInfo
from ollama import chat

# Use certifi's CA bundle — required on macOS Python from python.org
_SSL = ssl.create_default_context(cafile=certifi.where())


def _get(url: str) -> dict:
    """Fetch a URL and return parsed JSON, using certifi for SSL."""
    req = urllib.request.Request(url, headers={"User-Agent": "ask-gemma/1.0"})
    with urllib.request.urlopen(req, timeout=5, context=_SSL) as resp:
        return json.loads(resp.read())

_FALLBACK_TIMEZONE = "America/Denver"
_FALLBACK_LOCATION = "Superior, Colorado, USA"

# WMO weather interpretation codes -> human-readable description
_WMO = {
    0: "clear sky",
    1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "icy fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    71: "light snow", 73: "moderate snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "moderate showers", 82: "heavy showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}

_geo_cache: dict | None = None


def _fetch_geo() -> dict:
    """Fetch and cache ipinfo.io data for the current session."""
    global _geo_cache
    if _geo_cache is not None:
        return _geo_cache
    try:
        _geo_cache = _get("https://ipinfo.io/json")
    except Exception:
        _geo_cache = {}
    return _geo_cache


def _get_latlon() -> tuple[str, str] | None:
    """Return (lat, lon) strings, falling back to geocoding the city name."""
    data = _fetch_geo()
    loc  = data.get("loc", "")
    if loc and "," in loc:
        return tuple(loc.split(",", 1))
    # Fallback: geocode city via Open-Meteo's free geocoding API
    city = data.get("city", "") or _FALLBACK_LOCATION.split(",")[0].strip()
    try:
        geo = _get(f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city)}&count=1&language=en&format=json")
        results = geo.get("results") or []
        if results:
            return str(results[0]["latitude"]), str(results[0]["longitude"])
    except Exception:
        pass
    return None


def get_location_info():
    """Return (location_string, timezone) from cached geo data."""
    data = _fetch_geo()
    city     = data.get("city", "")
    region   = data.get("region", "")
    country  = data.get("country", "")
    timezone = data.get("timezone") or _FALLBACK_TIMEZONE
    location = ", ".join(part for part in (city, region, country) if part)
    return location or _FALLBACK_LOCATION, timezone


def get_weather(latlon: tuple[str, str] | None = None) -> str:
    """Fetch current conditions + 2-day forecast from Open-Meteo (no API key)."""
    coords = latlon or _get_latlon()
    if not coords:
        return ""
    lat, lon = coords
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,weather_code,wind_speed_10m"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code"
        f"&temperature_unit=fahrenheit&wind_speed_unit=mph"
        f"&precipitation_unit=inch&timezone=auto&forecast_days=2"
    )
    try:
        w = _get(url)
        cur   = w.get("current", {})
        daily = w.get("daily", {})
        temp  = cur.get("temperature_2m", "?")
        cond  = _WMO.get(cur.get("weather_code"), "unknown")
        wind  = cur.get("wind_speed_10m", "?")
        hi0   = daily.get("temperature_2m_max", ["?"])[0]
        lo0   = daily.get("temperature_2m_min", ["?"])[0]
        hi1   = daily.get("temperature_2m_max", ["?", "?"])[1]
        lo1   = daily.get("temperature_2m_min", ["?", "?"])[1]
        cond1 = _WMO.get(
            (daily.get("weather_code") or [None, None])[1], "unknown"
        )
        pop1  = daily.get("precipitation_probability_max", ["?", "?"])[1]
        return (
            f"Current weather: {temp}°F, {cond}, wind {wind} mph. "
            f"Today: high {hi0}°F, low {lo0}°F. "
            f"Tomorrow: {cond1}, high {hi1}°F, low {lo1}°F, "
            f"{pop1}% chance of precipitation."
        )
    except Exception:
        return ""


def get_system_prompt(
    location_override: str | None = None,
    latlon_override: tuple[str, str] | None = None,
) -> str:
    _, timezone = get_location_info()  # always use IP-based timezone
    location = location_override or get_location_info()[0]
    tz = ZoneInfo(timezone)
    now = datetime.datetime.now(tz=tz)
    tz_name = now.strftime("%Z")  # e.g. "MDT" or "MST" automatically
    weather = get_weather(latlon=latlon_override)
    weather_line = f" {weather}" if weather else ""
    return (
        f"Today is {now.strftime('%A, %B %d, %Y')}. "
        f"Current time is {now.strftime('%I:%M %p')} {tz_name}. "
        f"Location: {location}.{weather_line}\n\n"
        "You are a helpful assistant. Always use the above date, time, location, "
        "and weather when the user asks anything time-, location-, or weather-related."
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
