"""PersonaPlex "brain" — gives the full-duplex voice a smarter mind on the R9700.

PersonaPlex keeps doing what it's good at (listening while talking, backchannels,
interruptions, its own voice). This module listens alongside it:

  your audio ──► endpointing (energy VAD) ──► STT (faster-whisper, CT 101 :8996)
            ──► Qwen3.6-35B on the R9700 (LiteLLM router) ──► "SAY <answer>" on the
                patched personaplex's stdin ──► the answer is forced into PersonaPlex's
                inner-monologue text stream, paced by the model, so it SPEAKS it in its
                own voice.

The LLM answers only when the voice needs help (questions, facts, reasoning); for small
talk it replies PASS and PersonaPlex carries on by itself. For anything current (weather,
sports, news, prices) it replies SEARCH: <query>; we look it up (DuckDuckGo via the STT
service's /search) while the voice says "Let me check", then ask the LLM again with results.

Settings (brain.json next to this file, or BRAIN_<NAME> env vars): {"home": "City, State"}
is the default place for weather questions and tells the LLM where the person lives.

rev 7 (2026-09-19): VOICE lines are marked unreliable (PersonaPlex invents topics and the
brain was building on them); hard 2-sentence cap; web search; 500 ms pre-roll so the first
word of an utterance isn't clipped.
"""
import asyncio
import collections
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import aiohttp

STT_URL = "http://192.168.1.174:8996/stt?rate=24000"
LLM_URL = "http://192.168.1.176:4000/v1/chat/completions"
LLM_MODEL = "claude-r9700-qwen3.6-35b"          # reasoning disabled: fast first token
SEARCH_URL = "http://192.168.1.174:8996/search"
WEATHER_URL = "http://192.168.1.174:8996/weather"
KEY_FILE = Path(__file__).resolve().parent / ".litellm-key"
SETTINGS_FILE = Path(__file__).resolve().parent / "brain.json"   # local settings, not in git


def _setting(name, default=""):
    """BRAIN_<NAME> env var, else brain.json next to this file, else the default."""
    env = os.environ.get("BRAIN_" + name.upper())
    if env is not None:
        return env
    try:
        return json.loads(SETTINGS_FILE.read_text()).get(name, default)
    except (OSError, ValueError):
        return default

FRAME_S = 0.02                     # the browser sends 20 ms chunks
START_FRAMES = 3                   # 60 ms above threshold = speech started
END_FRAMES = 35                    # 700 ms below threshold = utterance over
MIN_UTT_S = 0.35
MAX_UTT_S = 30.0
MAX_TURNS = 16
PREROLL_FRAMES = 25                # 500 ms kept from before speech was detected
MAX_SENTENCES = 2
HOLD_FRAMES = 100                  # 8 s of forced silence after "Let me check." while we look it up;
                                   # the answer's CLEAR ends it early. Without it PersonaPlex keeps
                                   # talking and invents its own "lookup" results (braintest_C).
HOME = _setting("home")             # e.g. "Austin, Texas": default place for weather; "" = ask
TURN_HOLD_FRAMES = 50              # when you finish speaking, keep PersonaPlex quiet (forced silence,
                                   # 4 s max) until the brain has decided: PASS releases it, an answer
                                   # replaces it. Stops the instant made-up answers (Steve, 09-19).
WEATHER_WORDS = re.compile(r"\b(weather|rain|raining|forecast|temperature|snow|storm|storms|sunny|"
                           r"humid|humidity|hot|cold|freeze|freezing|degrees)\b", re.I)
TAIL_HOLD_FRAMES = 25              # 2 s of silence after a brain answer, so PersonaPlex hands the turn
                                   # back instead of tacking on its own made-up follow-up (braintest_G)
PAUSE_FRAMES = 16                  # 80 ms each; used only when the voice is mid-sentence

SYSTEM = """You are the thinking brain behind a real-time spoken voice assistant. Today is {today}.
{home_line}A small speech model (the VOICE) does the actual talking. It sounds natural but it knows very
little and constantly makes things up: it invents topics, facts and names nobody mentioned.

You get a transcript. USER lines are what the person said (speech-to-text, so expect small
errors and misheard names; use common sense, e.g. "Alice Cowboy" in a sports context is probably
"Dallas Cowboys"). VOICE lines are what the speech model said: UNRELIABLE, never treat them as
facts or as the topic of conversation. Only the USER decides the topic. BRAIN lines are what you
told the voice to say.

Look at the person's LAST utterance and reply with exactly one of:
1. PASS  - greetings, small talk, chit-chat about their day or mood, thanks, acknowledgements,
   or the utterance looks cut off, garbled or unfinished. The voice handles chat fine on its own.
   When in doubt, PASS.
2. WEATHER: <city, state>  - weather, temperature, rain or a forecast (a season-long outlook is
   a SEARCH instead).
   SEARCH: <short web search query>  - they need other current or live information: sports,
   news, scores, prices, schedules, anything that changes over time or happened recently.
3. Otherwise, what the voice should say next: a question, facts, numbers, reasoning, advice,
   a decision. At most two short sentences, under 35 words, warm and plain spoken English.
   No lists, markdown, emojis or stage directions. Write numbers and times as spoken words (two percent, ninety-eight degrees, four oh five), never digits.
   If the person is confused by or objects to something the VOICE said, apologize in a few
   words and steer back to what THEY asked, in one sentence."""



def available():
    return KEY_FILE.exists() and KEY_FILE.stat().st_size > 0


def _key():
    return KEY_FILE.read_text().strip()


def _cap(text, n=MAX_SENTENCES):
    """Keep at most n sentences: long answers make the voice ramble and drift."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return " ".join(parts[:n]).strip()


def _words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


class Brain:
    def __init__(self, ws, stdin, log):
        self.ws, self.stdin, self.log = ws, stdin, log
        self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20))
        self.hist = []                 # (role, text): USER / VOICE / BRAIN
        self.voice_buf = ""            # PersonaPlex text since the last user turn
        self.pre = collections.deque(maxlen=PREROLL_FRAMES)
        self.cur = bytearray()
        self.in_speech = False
        self.above = 0
        self.below = 0
        self.noise = 0.004
        self.turn = 0
        self.voice_t = 0.0             # when PersonaPlex last produced a word
        self.tasks = set()

    # --- inputs from the bridge ----------------------------------------------------
    def on_voice_text(self, text):
        self.voice_buf += text
        if text.strip():
            self.voice_t = time.monotonic()

    def feed(self, pcm):
        a = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
        if not len(a):
            return
        rms = float(np.sqrt(np.mean(a * a)))
        thr = max(0.012, self.noise * 3.0)
        if not self.in_speech:
            self.pre.append(bytes(pcm))
            if rms > thr:
                self.above += 1
            else:
                self.above = 0
                self.noise = 0.97 * self.noise + 0.03 * rms
            if self.above >= START_FRAMES:
                self.in_speech, self.below = True, 0
                self.cur = bytearray(b"".join(self.pre))
                self._speech_started()
        else:
            self.cur += pcm
            self.below = self.below + 1 if rms < thr else 0
            dur = len(self.cur) / 2 / 24000
            if self.below >= END_FRAMES or dur > MAX_UTT_S:
                self.in_speech, self.above = False, 0
                seg = bytes(self.cur)
                self.cur = bytearray()
                self.pre.clear()
                if len(seg) / 2 / 24000 >= MIN_UTT_S:
                    if TURN_HOLD_FRAMES:
                        self._send_model("CLEAR")
                        self._send_model(f"PAUSE {TURN_HOLD_FRAMES}")
                    t = asyncio.create_task(self._handle(seg, self.turn, time.monotonic()))
                    self.tasks.add(t)
                    t.add_done_callback(self.tasks.discard)

    # --- actions -------------------------------------------------------------------
    def _send_model(self, line):
        try:
            self.stdin.write((line + "\n").encode())
        except (BrokenPipeError, ConnectionResetError, RuntimeError):
            pass

    def _speech_started(self):
        # barge-in: the person is talking again, so drop any answer still queued
        self.turn += 1
        self._send_model("CLEAR")

    async def _event(self, **kw):
        if not self.ws.closed:
            await self.ws.send_json({"type": "brain", **kw})

    async def _stt(self, seg):
        async with self.http.post(STT_URL, data=seg) as r:
            j = await r.json()
        return j.get("text", "").strip(), j.get("ms")

    def _transcript(self):
        out = []
        for r, t in self.hist[-MAX_TURNS:]:
            if r == "VOICE":
                out.append(f"VOICE (unreliable): {t[-240:]}")
            else:
                out.append(f"{r}: {t}")
        return "\n".join(out)

    async def _chat(self, user, max_tokens=160):
        home_line = f"The person lives in {HOME}, unless they say otherwise.\n" if HOME else ""
        sysmsg = SYSTEM.format(today=datetime.now().strftime("%A, %B %-d, %Y"), home_line=home_line)
        body = {"model": LLM_MODEL, "temperature": 0.0, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": sysmsg},
                             {"role": "user", "content": user}]}
        async with self.http.post(LLM_URL, json=body,
                                  headers={"Authorization": f"Bearer {_key()}"}) as r:
            j = await r.json()
        out = (j.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
        return re.sub(r"\s+", " ", out.replace("*", "")).strip()

    async def _llm(self):
        return await self._chat(f"Transcript:\n{self._transcript()}\n\n"
                                "What should the voice say next? (PASS, SEARCH: <query>, or the words)")

    async def _search(self, q):
        async with self.http.get(SEARCH_URL, params={"q": q, "n": "3"}) as r:
            j = await r.json()
        return j.get("results") or []

    async def _weather(self, place):
        async with self.http.get(WEATHER_URL, params={"place": place}) as r:
            j = await r.json()
        return j.get("results") or []

    async def _llm_with_results(self, q, results):
        lines = "\n".join(f"- {x.get('title','')}: {x.get('body','')}" for x in results)[:2500]
        return await self._chat(
            f"Transcript:\n{self._transcript()}\n\nYou searched the web for: {q}\nResults:\n"
            f"{lines or '(no results)'}\n\nNow write exactly what the voice should say: at most two "
            "short spoken sentences answering the person's last question from these results. Only state "
            "facts that appear in the results; never add numbers or predictions that aren't there. If the "
            "results don't answer it, say so briefly. Do not reply PASS, SEARCH or WEATHER.")

    def _release(self, turn):
        """End the turn hold so PersonaPlex answers on its own (only if nobody spoke since)."""
        if turn == self.turn:
            self._send_model("CLEAR")

    def _say(self, text, clear=True):
        if clear:
            self._send_model("CLEAR")
            # If the voice is mid-sentence, let it finish that clause first: measured
            # 2026-09-19 (exp2), a 16-frame (1.3 s) pause gave the cleanest hand-over.
            if time.monotonic() - self.voice_t < 0.5:
                self._send_model(f"PAUSE {PAUSE_FRAMES}")
        self._send_model("SAY " + text)

    async def _handle(self, seg, turn, t_end):
        try:
            text, stt_ms = await self._stt(seg)
            if not text:
                self._release(turn)
                return
            # echo guard: ignore our own voice leaking back into the mic
            recent = set(_words(self.voice_buf[-400:]))
            w = _words(text)
            if len(w) >= 3 and recent and sum(x in recent for x in w) / len(w) > 0.7:
                self._release(turn)
                await self._event(heard=text, echo=True)
                return
            if self.voice_buf.strip():
                self.hist.append(("VOICE", self.voice_buf.strip()))
                self.voice_buf = ""
            self.hist.append(("USER", text))
            t_llm = time.monotonic()
            answer = await self._llm()
            searched = None
            m = re.match(r"(?i)^(search|weather)\s*:\s*(.+)", answer or "")
            if m and turn == self.turn:
                kind, searched = m.group(1).lower(), m.group(2).strip().strip('"')
                if kind == "search" and WEATHER_WORDS.search(searched) and not re.search(
                        r"(?i)season|outlook|winter|summer|year", searched):
                    kind = "weather"                       # the LLM often picks SEARCH for weather
                if kind == "weather":
                    # the place the person named, else the LLM's place, else the home setting
                    pm = re.search(r"\b(?:in|for|at) ([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)", text)
                    if pm:
                        searched = pm.group(1)
                    elif not searched[:1].isupper() or "weather" in searched.lower():
                        searched = HOME
                self._say("Let me check.")
                self._send_model(f"PAUSE {HOLD_FRAMES}")
                try:
                    results = await (self._weather(searched) if kind == "weather" else self._search(searched))
                except Exception as e:
                    self.log(f"brain search error: {e!r}")
                    results = []
                answer = await self._llm_with_results(searched, results)
            llm_ms = round((time.monotonic() - t_llm) * 1000)
            answer = _cap(answer)
            stale = turn != self.turn
            passed = not answer or answer.upper().startswith(("PASS", "SEARCH", "WEATHER"))
            if not passed and not stale:
                self._say(answer)
                self._send_model(f"PAUSE {TAIL_HOLD_FRAMES}")   # end of answer: hand the turn back
                self.hist.append(("BRAIN", answer))
            elif passed:
                self._release(turn)
            total = round((time.monotonic() - t_end) * 1000)
            self.log(f"brain: heard={text!r}{f' search={searched!r}' if searched else ''}"
                     f" -> {'PASS' if passed else answer!r}"
                     f"{' (stale)' if stale else ''} stt={stt_ms}ms llm={llm_ms}ms total={total}ms")
            await self._event(heard=text, said=None if passed else answer, stale=stale, search=searched,
                              stt_ms=stt_ms, llm_ms=llm_ms, total_ms=total)
        except Exception as e:                     # never take the conversation down
            self._release(turn)
            self.log(f"brain error: {e!r}")
            await self._event(error=str(e)[:200])

    async def close(self):
        for t in list(self.tasks):
            t.cancel()
        await self.http.close()
