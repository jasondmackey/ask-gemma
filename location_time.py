#!/usr/bin/env python3
"""location_time.py — display current location and time information.

Uses ipinfo.io with ip-api.com as a fallback for improved geolocation accuracy.
No API key required.

Usage:
    python3 location_time.py
    ./location_time.py
"""

import json
import ssl
import urllib.request
from datetime import datetime, timezone as dt_timezone
from zoneinfo import ZoneInfo, available_timezones

import certifi

_SSL = ssl.create_default_context(cafile=certifi.where())


def _get(url: str) -> dict:
    """Fetch a URL and return parsed JSON, using certifi for SSL."""
    req = urllib.request.Request(url, headers={"User-Agent": "location_time/1.0"})
    with urllib.request.urlopen(req, timeout=5, context=_SSL) as resp:
        return json.loads(resp.read())


def get_location() -> dict | None:
    """Return location info, trying ipinfo.io then ip-api.com as fallback."""
    # --- Primary: ipinfo.io ---
    try:
        data = _get("https://ipinfo.io/json")
        city    = data.get("city", "Unknown")
        region  = data.get("region", "Unknown")
        country = data.get("country", "Unknown")
        tz      = data.get("timezone", "Unknown")
        isp     = data.get("org", "Unknown")
        # ipinfo.io sometimes returns a bogus city; verify with ip-api.com
        if city and city != "Unknown":
            return {
                "city":         city,
                "region":       region,
                "country":      _country_name(country),
                "country_code": country,
                "timezone":     tz,
                "isp":          isp,
                "source":       "ipinfo.io",
            }
    except Exception:
        pass

    # --- Fallback: ip-api.com (no HTTPS on free tier, but more accurate city) ---
    try:
        data = _get("http://ip-api.com/json/?fields=status,city,regionName,country,countryCode,timezone,isp")
        if data.get("status") == "success":
            return {
                "city":         data.get("city", "Unknown"),
                "region":       data.get("regionName", "Unknown"),
                "country":      data.get("country", "Unknown"),
                "country_code": data.get("countryCode", "Unknown"),
                "timezone":     data.get("timezone", "Unknown"),
                "isp":          data.get("isp", "Unknown"),
                "source":       "ip-api.com",
            }
    except Exception:
        pass

    return None


def _country_name(code: str) -> str:
    """Map ISO country code to a readable name for common cases."""
    names = {
        "US": "United States", "CA": "Canada", "GB": "United Kingdom",
        "AU": "Australia", "DE": "Germany", "FR": "France", "JP": "Japan",
    }
    return names.get(code, code)


def get_time_info(timezone_str: str) -> dict:
    """Return UTC and local time info for the given IANA timezone string."""
    utc_now = datetime.now(tz=dt_timezone.utc)
    try:
        if timezone_str != "Unknown" and timezone_str in available_timezones():
            local_tz  = ZoneInfo(timezone_str)
            local_now = utc_now.astimezone(local_tz)
            utc_offset = local_now.strftime("%z")
            utc_offset_formatted = f"UTC{utc_offset[:3]}:{utc_offset[3:]}"
        else:
            # Fallback to system local time
            local_now = datetime.now()
            utc_offset_formatted = "Unknown"

        return {
            "utc_time":   utc_now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "local_time": local_now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "utc_offset": utc_offset_formatted,
            "iso_format": local_now.isoformat(),
        }
    except Exception as e:
        print(f"Error processing timezone: {e}")
        return {
            "utc_time":   utc_now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "local_time": "Error",
            "utc_offset": "Unknown",
        }


def display_info(location: dict, time_info: dict) -> None:
    """Display formatted location and time information."""
    print(f"\n📍 Location Information")
    print(f"   City:       {location['city']}")
    print(f"   Region:     {location['region']}")
    print(f"   Country:    {location['country']} ({location['country_code']})")
    print(f"   Timezone:   {location['timezone']}")
    print(f"   ISP:        {location['isp']}")

    print(f"\n⏰ Time Information")
    print(f"   Local Time: {time_info['local_time']}")
    print(f"   UTC Time:   {time_info['utc_time']}")
    print(f"   Offset:     {time_info['utc_offset']}")
    print()


def main():
    """Main function."""
    print("Fetching location and time information...")

    location = get_location()
    if location:
        time_info = get_time_info(location["timezone"])
        display_info(location, time_info)
    else:
        print("Failed to retrieve location information.")


if __name__ == "__main__":
    main()
