"""
Weather Agent
-------------
Fetches real-time weather from OpenWeatherMap and evaluates suitability
for agricultural use.  Falls back to mock data when the API key is absent.
"""
import os
from typing import Any

import requests

BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


class LocationNotFoundError(Exception):
    """Raised when OpenWeatherMap cannot resolve the given location."""


class WeatherAgent:
    """Fetches and interprets current weather conditions for a given location."""

    def get_weather(self, location: str) -> dict[str, Any]:
        """
        Return structured weather data for *location* ('lat,lon').

        Falls back to mock data when the API key is missing.
        Raises LocationNotFoundError when the API key is present but the
        location cannot be resolved.
        """
        api_key = os.getenv("OPENWEATHER_API_KEY", "")
        if not api_key:
            return {
                **self._mock_weather(location),
                "warning": "No OPENWEATHER_API_KEY set – weather data is simulated.",
            }

        try:
            return self._fetch_live(location, api_key)
        except LocationNotFoundError:
            raise
        except Exception as exc:
            return {**self._mock_weather(location), "warning": str(exc)}

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _fetch_live(self, location: str, api_key: str) -> dict[str, Any]:
        """Call OpenWeatherMap and normalise the response."""
        params: dict[str, Any] = {
            "appid": api_key,
            "units": "metric",
        }

        # Support "lat,lon" shorthand
        if "," in location:
            lat, lon = location.split(",", 1)
            params["lat"] = lat.strip()
            params["lon"] = lon.strip()
        else:
            params["q"] = location

        resp = requests.get(BASE_URL, params=params, timeout=10)
        if resp.status_code == 404:
            raise LocationNotFoundError(
                f"Location '{location}' was not found. "
                "Please enter valid coordinates as 'latitude,longitude' (e.g. '28.6139,77.2090')."
            )
        resp.raise_for_status()
        data = resp.json()

        temperature = data["main"]["temp"]
        humidity    = data["main"]["humidity"]
        description = data["weather"][0]["description"]
        wind_speed  = data["wind"]["speed"]
        rainfall_1h = data.get("rain", {}).get("1h", 0.0)

        return {
            "location":    data.get("name", location),
            "temperature": temperature,
            "humidity":    humidity,
            "description": description,
            "wind_speed":  wind_speed,
            "rainfall_1h": rainfall_1h,
            "suitability": self._evaluate_suitability(temperature, humidity, rainfall_1h),
            "source":      "openweathermap",
        }

    @staticmethod
    def _mock_weather(location: str) -> dict[str, Any]:
        """Return plausible default weather when the API key is absent."""
        return {
            "location":    location,
            "temperature": 25.0,
            "humidity":    70.0,
            "description": "partly cloudy (mock data)",
            "wind_speed":  3.5,
            "rainfall_1h": 0.0,
            "suitability": "moderate – suitable for most crops",
            "source":      "mock",
        }

    @staticmethod
    def _evaluate_suitability(temp: float, humidity: float, rainfall: float) -> str:
        """Translate raw weather numbers into a plain-language suitability label."""
        issues = []

        if temp > 40:
            issues.append("extreme heat stress risk")
        elif temp < 5:
            issues.append("frost / cold stress risk")

        if humidity < 20:
            issues.append("very low humidity – moisture stress likely")
        elif humidity > 95:
            issues.append("very high humidity – fungal disease risk")

        if rainfall > 50:
            issues.append("heavy rainfall – flooding / waterlogging risk")

        if not issues:
            if 15 <= temp <= 35 and 40 <= humidity <= 85:
                return "excellent – ideal growing conditions"
            return "good – generally suitable for most crops"

        return "poor – " + "; ".join(issues)
