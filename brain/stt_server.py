#!/opt/voice/bin/python
"""PersonaPlex brain — speech-to-text service (runs on CT 101's host, not in docker).

POST /stt?rate=24000   body: raw s16le mono PCM   ->  {"text": "...", "ms": 312, "audio_s": 2.4}
GET  /health           -> {"ok": true, "model": "small.en"}
GET  /search?q=..&n=4  -> {"results": [{"title","body","href"}], "ms": 800}   (DuckDuckGo via ddgs,
                          the same library Open WebUI uses; for the brain's live-info questions).
                          Also mixes in news results, which carry the concrete facts (scores, dates).
GET  /weather?place=.. -> {"results": [{"title","body"}]}   Open-Meteo (free, no key): now + 7 days, °F

faster-whisper on CPU (int8), reusing the /opt/voice venv and its cached models.
One transcription at a time (a lock), which matches the bridge's one-conversation rule.
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from faster_whisper import WhisperModel

MODEL = os.environ.get("STT_MODEL", "small.en")
BEAM = int(os.environ.get("STT_BEAM", "1"))
THREADS = int(os.environ.get("STT_THREADS", "4"))
PORT = int(os.environ.get("STT_PORT", "8996"))
WEATHER_HOME = os.environ.get("WEATHER_HOME", "")   # default place for /weather, e.g. "Austin, Texas"

model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=THREADS)
list(model.transcribe(np.zeros(16000, np.float32), language="en")[0])  # warm-up
lock = threading.Lock()


WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain",
       65: "heavy rain", 71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
       81: "rain showers", 82: "violent rain showers", 95: "thunderstorms", 96: "thunderstorms with hail",
       99: "thunderstorms with hail"}


def _get(url, params):
    import urllib.request
    from urllib.parse import urlencode
    with urllib.request.urlopen(url + "?" + urlencode(params), timeout=8) as r:
        return json.loads(r.read())


def weather(place):
    name = place.split(",")[0].strip()
    g = _get("https://geocoding-api.open-meteo.com/v1/search", {"name": name, "count": 5, "language": "en"})
    hits = g.get("results") or []
    if not hits:
        return []
    want = place.lower()
    loc = next((h for h in hits if (h.get("admin1") or "").lower() in want), hits[0])
    f = _get("https://api.open-meteo.com/v1/forecast", {
        "latitude": loc["latitude"], "longitude": loc["longitude"], "timezone": "auto",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "forecast_days": 7,
        "current": "temperature_2m,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"})
    where = f"{loc['name']}, {loc.get('admin1', '')}"
    c = f.get("current", {})
    out = [{"title": f"Now in {where}", "body": f"{round(c.get('temperature_2m', 0))}°F, "
            f"{WMO.get(c.get('weather_code'), 'unknown')}, wind {round(c.get('wind_speed_10m', 0))} mph"}]
    d = f.get("daily", {})
    for i, day in enumerate(d.get("time", [])):
        out.append({"title": f"{where} {day}", "body": f"high {round(d['temperature_2m_max'][i])}°F, "
                    f"low {round(d['temperature_2m_min'][i])}°F, {WMO.get(d['weather_code'][i], 'mixed')}, "
                    f"{d['precipitation_probability_max'][i]}% chance of rain"})
    return out


def to16k(a, rate):
    if rate == 16000:
        return a
    n = int(len(a) * 16000 / rate)
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/health":
            return self._json(200, {"ok": True, "model": MODEL})
        if u.path == "/search":
            qs = parse_qs(u.query)
            q = (qs.get("q") or [""])[0].strip()[:200]
            n = max(1, min(8, int((qs.get("n") or ["4"])[0])))
            t0 = time.monotonic()
            try:
                from ddgs import DDGS
                res = []
                if q:
                    d = DDGS()
                    try:
                        res += [{"title": r.get("title", ""), "body": f"({r.get('date', '')[:10]}) {r.get('body', '')}",
                                 "href": r.get("url", "")} for r in d.news(q, max_results=n)]
                    except Exception:
                        pass
                    res += [{"title": r.get("title", ""), "body": r.get("body", ""), "href": r.get("href", "")}
                            for r in d.text(q, max_results=n)]
                return self._json(200, {"results": res, "ms": round((time.monotonic() - t0) * 1000)})
            except Exception as e:
                return self._json(502, {"results": [], "error": str(e)[:200]})
        if u.path == "/weather":
            place = ((parse_qs(u.query).get("place") or [""])[0]).strip()[:80] or WEATHER_HOME
            if not place:
                return self._json(400, {"results": [], "error": "no place given and WEATHER_HOME not set"})
            t0 = time.monotonic()
            try:
                return self._json(200, {"results": weather(place), "ms": round((time.monotonic() - t0) * 1000)})
            except Exception as e:
                return self._json(502, {"results": [], "error": str(e)[:200]})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/stt":
            return self._json(404, {"error": "not found"})
        rate = int(parse_qs(u.query).get("rate", ["24000"])[0])
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
        t0 = time.monotonic()
        with lock:
            segs, _ = model.transcribe(to16k(a, rate), language="en", beam_size=BEAM,
                                       vad_filter=False, condition_on_previous_text=False)
            text = "".join(s.text for s in segs).strip()
        self._json(200, {"text": text, "ms": round((time.monotonic() - t0) * 1000),
                         "audio_s": round(len(a) / rate, 2)})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"stt on :{PORT} model={MODEL} threads={THREADS}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
