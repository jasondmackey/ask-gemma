# ask-gemma

A minimal Python script that queries a local [Ollama](https://ollama.com) model (Gemma 4 31B) with awareness of your current **date, time, and location** automatically injected into every prompt.

## How it works

1. At startup, the script calls `ipinfo.io` to resolve your public IP to a city, region, country, and IANA timezone.
2. The local time is computed using that timezone (correctly handling DST — e.g. MDT vs MST).
3. A system prompt containing the date, time, and location is prepended to every Ollama request.
4. If the geolocation request fails (offline, timeout), it silently falls back to hardcoded defaults.

## Requirements

- Python 3.9+ (uses `zoneinfo`, part of the standard library)
- [Ollama](https://ollama.com) running locally with `gemma4:31b` pulled
- The `ollama` Python package

## Setup

```bash
# Install Ollama (if not already installed)
# https://ollama.com/download

# Pull the model
ollama pull gemma4:31b

# Install the Python client
pip3 install ollama
```

## Usage

```bash
python3 ask_gemma.py
```

To ask your own questions, edit the `questions` list at the bottom of `ask_gemma.py`, or import and call `ask()` from another script:

```python
from ask_gemma import ask

print(ask("What's the weather like here this time of year?"))
```

## Configuration

Two fallback constants at the top of the script are used when geolocation is unavailable:

```python
_FALLBACK_TIMEZONE = "America/Denver"
_FALLBACK_LOCATION = "Superior, Colorado, USA"
```

Update these to match your default location.

## Example output

```
[context] Location: Superior, Colorado, USA | Timezone: America/Denver | Local time: 01:27 PM MDT

Q: What day of the week is it and what time is it?
A: It is Friday, and the time is 01:27 PM MDT.

Q: How many days until the end of the month?
A: There are 20 days remaining until the end of the month (April 30, 2026).

Q: What city and country am I in right now?
A: You are in Superior, USA.
```
