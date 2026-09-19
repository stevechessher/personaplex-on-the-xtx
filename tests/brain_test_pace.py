#!/usr/bin/env python3
"""End-to-end test of brain mode, no human needed.

Turn-taking and correction test. Kokoro speaks into the bridge WebSocket in real time:
  1. a time question with a 1.1 s pause in the middle -> one turn, "eight fifteen"
  2. "what can you do?"                                -> the fixed abilities answer
  3. "that's not what I meant"                         -> an apology and a question
  4. "wait a little longer on my pauses"               -> pace goes to 1.4x
Records the reply audio, the text stream and the brain events with timestamps.
"""
import asyncio, io, json, sys, time, wave
import numpy as np
import aiohttp

WS = "ws://127.0.0.1:8998/ws?brain=1&model=q8&voice=NATF2&seed=0&prompt=" + \
     "You%20are%20a%20friendly%2C%20concise%20assistant."
KOKORO = "http://192.168.1.174:8880/v1/audio/speech"
RATE, CHUNK = 24000, 480
import os
UTTS = [
    ("If I leave the house at seven forty and the drive takes", 1.1, "thirty five minutes, when do I get there?"),
    "Can you tell me what you're able to do?",
    "No, I'm not talking about a website. That's not what I meant.",
    "Can you wait a little longer on my pauses?",
]
OUT = "/opt/personaplex/brain/test"


async def kokoro(s, text):
    async with s.post(KOKORO, json={"model": "kokoro", "input": text, "voice": "af_heart",
                                    "response_format": "wav"}) as r:
        r.raise_for_status()
        raw = await r.read()
    with wave.open(io.BytesIO(raw)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        sr = w.getframerate()
    if sr != RATE:
        x = np.linspace(0, len(pcm) - 1, int(len(pcm) * RATE / sr))
        pcm = np.interp(x, np.arange(len(pcm)), pcm).astype(np.int16)
    return pcm


async def main():
    async with aiohttp.ClientSession() as s:
        qs = []
        for u in UTTS:
            if isinstance(u, tuple):
                a1, gap, a2 = u
                qs.append(np.concatenate([await kokoro(s, a1), np.zeros(int(RATE * gap), np.int16), await kokoro(s, a2)]))
            else:
                qs.append(await kokoro(s, u))
        audio, events, texts = [], [], []
        ready = asyncio.Event()
        t0 = time.monotonic()
        async with s.ws_connect(WS, max_msg_size=1 << 20) as ws:
            async def rx():
                async for m in ws:
                    now = time.monotonic() - t0
                    if m.type == aiohttp.WSMsgType.BINARY:
                        audio.append(np.frombuffer(m.data, np.int16))
                    else:
                        j = json.loads(m.data)
                        if j["type"] == "ready":
                            ready.set()
                        elif j["type"] == "text":
                            texts.append((now, j["data"]))
                        elif j["type"] == "brain":
                            events.append((now, j))
                            print(f"[{now:6.2f}s] BRAIN {json.dumps(j)}", flush=True)
                        else:
                            print(f"[{now:6.2f}s] {j}", flush=True)
            rt = asyncio.create_task(rx())
            await asyncio.wait_for(ready.wait(), 90)
            audio.clear()                       # keep only the conversation, not load-time silence
            t_conv = time.monotonic() - t0

            async def send(pcm):
                t_next = time.monotonic()
                for i in range(0, len(pcm), CHUNK):
                    await ws.send_bytes(pcm[i:i + CHUNK].tobytes())
                    t_next += CHUNK / RATE
                    await asyncio.sleep(max(0, t_next - time.monotonic()))

            silence = lambda sec: np.zeros(int(RATE * sec), np.int16)
            await send(silence(4))              # let the greeting happen
            marks = []
            for q in qs:
                await send(q)
                marks.append(time.monotonic() - t0)
                print(f"[{marks[-1]:6.2f}s] --- finished saying: utterance {len(marks)}", flush=True)
                await send(silence(15))
            await ws.send_str("stop")
            await ws.close()
            rt.cancel()

    print("\nTEXT STREAM (what PersonaPlex's inner monologue said, timestamped):")
    line, lt = "", None
    for t, x in texts:
        if lt is None:
            lt = t
        line += x
        if len(line) > 90:
            print(f"  [{lt:6.2f}s] {line.strip()}"); line, lt = "", None
    if line.strip():
        print(f"  [{lt:6.2f}s] {line.strip()}")
    print("\nutterance end marks:", [round(m, 2) for m in marks], "conversation started at", round(t_conv, 2))
    out = np.concatenate(audio) if audio else np.zeros(1, np.int16)
    with wave.open(OUT + "_reply.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(out.tobytes())
    inp = np.concatenate([np.zeros(RATE * 4, np.int16)] + [np.concatenate([q, np.zeros(RATE * 15, np.int16)]) for q in qs])
    n = max(len(out), len(inp))
    mix = np.zeros(n, np.float32)
    mix[:len(inp)] += inp * 0.6
    mix[:len(out)] += out * 0.8
    with wave.open(OUT + "_conversation.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes(np.clip(mix, -32768, 32767).astype(np.int16).tobytes())
    print("saved", OUT + "_reply.wav", "and", OUT + "_conversation.wav")


asyncio.run(main())
