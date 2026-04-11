# ask-gemma

A minimal Python script that queries a local [Ollama](https://ollama.com) model (Gemma 4 31B) with awareness of your current **date, time, and location** automatically injected into every prompt.

## How it works

1. At startup, the script calls `ipinfo.io` to resolve your public IP to a city, region, country, and IANA timezone.
2. The local time is computed using that timezone (correctly handling DST — e.g. MDT vs MST).
3. A system prompt containing the date, time, and location is prepended to every Ollama request.
4. If the geolocation request fails (offline, timeout), it silently falls back to hardcoded defaults.

## Requirements

- Python 3.10+ (uses `zoneinfo` and `dict | None` syntax)
- [Ollama](https://ollama.com) running locally with `gemma4:31b` pulled
- The `ollama` and `certifi` Python packages

## Setup

```bash
# Install Ollama (if not already installed)
# https://ollama.com/download

# Pull the model
ollama pull gemma4:31b

# Install Python dependencies
pip3 install ollama certifi
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

---

## location_time.py

A standalone utility to display your current location, local time, and UTC time.
Uses `ip-api.com` (primary) and `ipinfo.io` (fallback) for geolocation — no API key required.

### Features

- Dual geolocation providers with automatic fallback
- 24-hour result cache (skip with `--no-cache`)
- Retry logic with configurable timeout
- `LOCATION_OVERRIDE` env var to correct ISP-level city resolution
- ISO 8601 timestamp, UTC offset, and coordinates

### Usage

```bash
# Basic run
python3 location_time.py

# Skip cache (force fresh lookup)
python3 location_time.py --no-cache

# Verbose logging to stderr
python3 location_time.py --verbose

# Override city (IP geolocation resolves to ISP node, e.g. Denver not Superior)
export LOCATION_OVERRIDE="Superior, Colorado, USA"
python3 location_time.py

# Custom timeout
python3 location_time.py --timeout 5
```

### Environment variables

`LOCATION_OVERRIDE` — Override the displayed city. IP geolocation resolves to your ISP's
routing node rather than your actual city. Add to `~/.zshrc` to make permanent:

```bash
export LOCATION_OVERRIDE="Superior, Colorado, USA"
```

### Example output

```
Fetching location and time information...

📍 Location Information
   City:        Superior, Colorado, USA
   ⚠️  Location overridden via $LOCATION_OVERRIDE
   Timezone:    America/Denver  (via ip-api)
   ISP:         Comcast Cable Communications, LLC
   Coordinates: 39.8831, -105.1122

⏰ Time Information
   Local Time:  2026-04-11 15:06:53 MDT
   UTC Time:    2026-04-11 21:06:53 UTC
   Offset:      UTC-06:00
   Timezone:    America/Denver
   ISO Format:  2026-04-11T15:06:53.029835-06:00
```

---

## ask_gemma.py — Example output

```
[context] Location: Superior, Colorado, USA | Timezone: America/Denver | Local time: 01:27 PM MDT

Q: What day of the week is it and what time is it?
A: It is Friday, and the time is 01:27 PM MDT.

Q: How many days until the end of the month?
A: There are 20 days remaining until the end of the month (April 30, 2026).

Q: What city and country am I in right now?
A: You are in Superior, USA.
```
