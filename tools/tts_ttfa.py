"""Time-to-first-audio for an OpenAI-style /v1/audio/speech endpoint (streamed PCM).

usage: python3 tts_ttfa.py http://host:8880/v1/audio/speech [voice]
"""
import json, statistics, sys, time, urllib.request

URL = sys.argv[1]
VOICE = sys.argv[2] if len(sys.argv) > 2 else "af_heart"
TEXTS = {
    "short (5 words)": "It arrives at four oh five.",
    "reply (2 sentences)": "It looks like there is only a one percent chance of rain tomorrow in your area. "
                           "You can expect it to be overcast with a high of ninety eight degrees.",
}


def once(text):
    body = json.dumps({"model": "kokoro", "input": text, "voice": VOICE,
                       "response_format": "pcm", "stream": True}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=60) as r:
        first, n = None, 0
        while True:
            b = r.read1(4096)
            if not b:
                break
            if first is None:
                first = time.monotonic() - t0
            n += len(b)
    return first, time.monotonic() - t0, n / 2 / 24000


once("warm up")
once("warm up again, a little longer this time.")
for name, text in TEXTS.items():
    firsts, totals = [], []
    for _ in range(5):
        f, t, dur = once(text)
        firsts.append(f); totals.append(t)
    print(f"{name:20s} first audio median {statistics.median(firsts)*1000:6.0f} ms "
          f"(min {min(firsts)*1000:.0f}, max {max(firsts)*1000:.0f})   "
          f"all audio {statistics.median(totals)*1000:6.0f} ms for {dur:.1f} s of speech")
