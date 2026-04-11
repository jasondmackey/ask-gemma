#!/usr/bin/env python3
"""Enhanced Current Location and Time Display Script.

Displays local and UTC time information for current location with
multiple API providers, caching, retry logic, and fallback options.

Requires Python 3.10+.

Usage:
    python3 location_time.py
    python3 location_time.py --verbose
    python3 location_time.py --no-cache
    LOCATION_OVERRIDE="Superior, Colorado, USA" python3 location_time.py

Environment variables:
    LOCATION_OVERRIDE   Override the city displayed (IP geolocation is ISP-level).
                        e.g.  export LOCATION_OVERRIDE="Superior, Colorado, USA"
"""

import argparse
import json
import logging
import os
import ssl
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

import certifi

if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ required for this script")

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

_SSL = ssl.create_default_context(cafile=certifi.where())

# Optional location override — IP geolocation resolves to ISP routing node
# (e.g. Denver) rather than your actual city (e.g. Superior, Colorado).
# e.g.  export LOCATION_OVERRIDE="Superior, Colorado, USA"
LOCATION_OVERRIDE = os.environ.get("LOCATION_OVERRIDE", "").strip()

DEFAULT_CONFIG: dict[str, Any] = {
    "timeout": 10,
    "retry_attempts": 3,
    "retry_delay": 2,
    "cache_file": "location_cache.json",
    "cache_expiry_hours": 24,
    "api_providers": [
        {
            "name": "ip-api",
            # NOTE: ip-api.com free tier is HTTP only
            "url": "http://ip-api.com/json/?fields=status,city,regionName,country,countryCode,timezone,isp,lat,lon",
            "enabled": True,
            "fields": {
                "city": "city", "region": "regionName",
                "country": "country", "country_code": "countryCode",
                "timezone": "timezone", "isp": "isp",
                "lat": "lat", "lon": "lon",
            },
            "success_key": "status",
            "success_value": "success",
        },
        {
            "name": "ipinfo",
            "url": "https://ipinfo.io/json",
            "enabled": True,
            "fields": {
                "city": "city", "region": "region",
                "country": "country", "country_code": "country",
                "timezone": "timezone", "isp": "org",
                "lat": None, "lon": None,  # ipinfo returns "loc": "lat,lon"
            },
            "success_key": "ip",
            "success_value": None,
        },
    ],
}


def _fetch(url: str, timeout: int) -> dict:
    """Fetch a URL and return parsed JSON."""
    req = urllib.request.Request(url, headers={"User-Agent": "location_time/2.0"})
    ctx = _SSL if url.startswith("https") else None
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read())


class LocationService:
    """Service to handle location detection with fallback providers."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        cache_file = config.get("cache_file")
        self.cache_file = Path(cache_file) if cache_file else None

    def _load_cached_location(self) -> Optional[dict[str, Any]]:
        """Load location from cache if still valid."""
        if not self.cache_file or not self.cache_file.exists():
            return None
        try:
            with open(self.cache_file) as f:
                cached_data = json.load(f)
            cache_time = datetime.fromisoformat(cached_data["cache_time"])
            expiry_time = cache_time + timedelta(hours=self.config["cache_expiry_hours"])
            if datetime.now(timezone.utc) < expiry_time:
                logger.info("Using cached location data")
                return cached_data["location_data"]
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to load cache: {e}")
        return None

    def _save_location_to_cache(self, location_data: dict[str, Any]) -> None:
        """Save location data to cache."""
        if not self.cache_file:
            return
        try:
            cache_data = {
                "cache_time": datetime.now(timezone.utc).isoformat(),
                "location_data": location_data,
            }
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
            logger.info("Location data cached successfully")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _try_provider(self, provider: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Try to get location from a specific provider."""
        if not provider.get("enabled", True):
            return None
        url    = provider["url"]
        fields = provider["fields"]
        try:
            data = _fetch(url, self.config["timeout"])
            # Validate response
            success_key = provider.get("success_key")
            success_val = provider.get("success_value")
            if success_key:
                val = data.get(success_key)
                if success_val is not None and val != success_val:
                    logger.warning(f"Provider {provider['name']}: status={val!r}")
                    return None
                if success_val is None and not val:
                    logger.warning(f"Provider {provider['name']}: missing {success_key!r}")
                    return None
            # Parse lat/lon — ipinfo returns "loc": "lat,lon"
            lat, lon = 0.0, 0.0
            if fields["lat"] is None:
                loc_str = data.get("loc", "")
                if "," in loc_str:
                    lat, lon = map(float, loc_str.split(",", 1))
            else:
                lat = float(data.get(fields["lat"]) or 0)
                lon = float(data.get(fields["lon"]) or 0)
            return {
                "city":         data.get(fields["city"], "Unknown"),
                "region":       data.get(fields["region"], "Unknown"),
                "country":      data.get(fields["country"], "Unknown"),
                "country_code": data.get(fields["country_code"], "Unknown"),
                "timezone":     data.get(fields["timezone"], "Unknown"),
                "isp":          data.get(fields["isp"], "Unknown"),
                "lat":          lat,
                "lon":          lon,
                "provider":     provider["name"],
            }
        except Exception as e:
            logger.warning(f"Provider {provider['name']} failed: {e}")
            return None

    def get_location(self) -> Optional[dict[str, Any]]:
        """Get current location with retry and fallback providers."""
        cached = self._load_cached_location()
        if cached:
            return cached
        for provider in self.config["api_providers"]:
            for attempt in range(self.config["retry_attempts"]):
                if attempt > 0:
                    logger.info(f"Retry {attempt + 1} for {provider['name']}")
                    time.sleep(self.config["retry_delay"])
                location = self._try_provider(provider)
                if location:
                    self._save_location_to_cache(location)
                    return location
        logger.error("All location providers failed")
        return None


class TimeService:
    """Service to handle time-related operations."""

    @staticmethod
    def format_offset(dt: datetime) -> str:
        """Format UTC offset as UTC±HH:MM."""
        offset = dt.utcoffset()
        if offset is None:
            return "Unknown"
        total_seconds = int(offset.total_seconds())
        sign = "+" if total_seconds >= 0 else "-"
        total_seconds = abs(total_seconds)
        hours, rem = divmod(total_seconds, 3600)
        minutes = rem // 60
        return f"UTC{sign}{hours:02d}:{minutes:02d}"

    @staticmethod
    def get_time_info(timezone_str: str) -> dict[str, str]:
        """Get current time in local timezone and UTC."""
        utc_now = datetime.now(timezone.utc)
        try:
            if timezone_str != "Unknown" and timezone_str in available_timezones():
                local_now = utc_now.astimezone(ZoneInfo(timezone_str))
            else:
                local_now = datetime.now().astimezone()
        except (ZoneInfoNotFoundError, ValueError) as e:
            logger.warning(f"Timezone error: {e}. Falling back to system time.")
            local_now = datetime.now().astimezone()
        return {
            "utc_time":      utc_now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "local_time":    local_now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "utc_offset":    TimeService.format_offset(local_now),
            "iso_format":    local_now.isoformat(),
            "timezone_name": str(local_now.tzinfo) if local_now.tzinfo else "System",
        }


class DisplayService:
    """Service to handle display formatting."""

    @staticmethod
    def display_info(location: dict[str, Any], time_info: dict[str, str]) -> None:
        """Display formatted location and time information."""
        city = LOCATION_OVERRIDE if LOCATION_OVERRIDE else location["city"]
        print(f"\n📍 Location Information")
        print(f"   City:        {city}")
        if not LOCATION_OVERRIDE:
            if location.get("region", "Unknown") != "Unknown":
                print(f"   Region:      {location['region']}")
            country = location.get("country", "Unknown")
            code    = location.get("country_code", "")
            if country != "Unknown":
                country_str = f"{country} ({code})" if code and code != country else country
                print(f"   Country:     {country_str}")
        else:
            print(f"   ⚠️  Location overridden via $LOCATION_OVERRIDE")
        print(f"   Timezone:    {location['timezone']}  (via {location.get('provider', 'unknown')})")
        print(f"   ISP:         {location['isp']}")
        if location.get("lat") or location.get("lon"):
            print(f"   Coordinates: {location['lat']}, {location['lon']}")
        print(f"\n⏰ Time Information")
        print(f"   Local Time:  {time_info['local_time']}")
        print(f"   UTC Time:    {time_info['utc_time']}")
        print(f"   Offset:      {time_info['utc_offset']}")
        print(f"   Timezone:    {time_info['timezone_name']}")
        print(f"   ISO Format:  {time_info['iso_format']}")
        print()

    @staticmethod
    def display_error(message: str) -> None:
        """Display error and fall back to system time."""
        print(f"\n❌ Error: {message}")
        print("Using system time information only...\n")
        time_info = TimeService.get_time_info("Unknown")
        print(f"⏰ System Time Information")
        print(f"   Local Time:  {time_info['local_time']}")
        print(f"   UTC Time:    {time_info['utc_time']}")
        print(f"   Offset:      {time_info['utc_offset']}")
        print(f"   Timezone:    {time_info['timezone_name']}")
        print()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display current location and time information")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache usage")
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_CONFIG["timeout"],
        help=f"Request timeout in seconds (default: {DEFAULT_CONFIG['timeout']})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    config = DEFAULT_CONFIG.copy()
    config["api_providers"] = DEFAULT_CONFIG["api_providers"][:]
    config["timeout"] = args.timeout
    if args.no_cache:
        config["cache_file"] = None

    print("Fetching location and time information...")
    location = LocationService(config).get_location()
    if location:
        DisplayService.display_info(location, TimeService.get_time_info(location["timezone"]))
    else:
        DisplayService.display_error("Failed to retrieve location information")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
