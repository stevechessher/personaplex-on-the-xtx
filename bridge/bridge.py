#!/usr/bin/env python3
"""PersonaPlex network bridge.

Runs the UNMODIFIED moshi.cpp `personaplex` binary and moves audio between a
browser WebSocket and two PulseAudio null sinks:

  browser mic --ws--> pacat --> pp_in  --(monitor)--> personaplex (SDL capture)
  browser spk <--ws-- parec <-- pp_out.monitor <---- personaplex (SDL playback)

One conversation at a time (one GPU model instance). Each WebSocket connection
spawns a fresh personaplex process and kills it on disconnect.

Wire format: binary frames = mono s16le PCM @ 24 kHz, both directions.
Text frames (server->client) = JSON: {"type": "status"|"ready"|"text"|"brain"|"error"|"ended", ...}

Brain mode (?brain=1): runs the PATCHED personaplex (/opt/personaplex/moshi-inject, text injection
via stdin) and brain.py, which transcribes you, asks Qwen3.6-35B on the R9700, and injects the
answer so PersonaPlex speaks it in its own voice. Without ?brain=1 nothing changes.
"""
import asyncio
import json
import os
import re
import resource
import signal
import time
from pathlib import Path

from aiohttp import web, WSMsgType

import brain

ROOT = Path("/opt/personaplex")
BIN = ROOT / "moshi-bin-linux-x64-v0.8.0-beta" / "personaplex"
INJECT_BIN = ROOT / "moshi-inject" / "personaplex"     # patched: --inject-stdin (brain mode)
MODELS = ROOT / "models"
VOICES = ROOT / "voices-f32"          # bf16->f32 converted voices (see voice_bf16_to_f32.py)
HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("BRIDGE_PORT", "8998"))
PULSE = os.environ.get("PULSE_SERVER", "unix:/tmp/pulse/native")
RATE = 24000
CHUNK_BYTES = 960                      # 20 ms of s16le mono @ 24 kHz
MAX_PROMPT = 2000
# model variants: extra personaplex args; the FIRST available one is the default (q8 = Steve's daily pick).
# Each lives in its own dir with a personaplex-config.json naming its GGUF (same layout as Codes4Fun q4),
# so no -q/-g flags and no original safetensors needed (deleted 2026-09-19).
#   q8   = nvidia/personaplex-7b-v1 quantized locally to q8_0 (9.3 GB VRAM, ~30 fps, ~7 s load)
#   q4   = Codes4Fun q4_k GGUF, the tool default (5.4 GB VRAM, ~34 fps, ~2.4 s load)
#   full = original bf16 weights, GGUF written by bf16gguf.py (16.8 GB VRAM, ~22.6 fps, ~12 s load);
#          verified bit-identical F32 / cos>=0.99996 BF16 vs q8, identical seeded output to the safetensors
M = MODELS / "local"
MODEL_ARGS = {
    "q8": ["-m", "local/personaplex-7b-v1-q8_0-GGUF"],
    "q4": [],
    "full": ["-m", "local/personaplex-7b-v1-bf16-GGUF"],
}
MODEL_FILES = {"q8": M / "personaplex-7b-v1-q8_0-GGUF" / "model-q8_0.gguf",
               "q4": MODELS / "Codes4Fun" / "personaplex-7b-v1-q4_k-GGUF" / "model-q4_k.gguf",
               "full": M / "personaplex-7b-v1-bf16-GGUF" / "model-bf16.gguf"}


def models():
    # offered only if the file exists AND is non-empty (a 0-byte GGUF segfaults moshi.cpp)
    return [m for m in MODEL_ARGS
            if m not in MODEL_FILES or (MODEL_FILES[m].exists() and MODEL_FILES[m].stat().st_size > 0)]

session_lock = asyncio.Lock()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def voices():
    return sorted(p.stem for p in VOICES.glob("*.gguf"))


async def index(_req):
    return web.FileResponse(HERE / "index.html")


async def list_voices(_req):
    return web.json_response({"voices": voices(), "models": models()})


def brain_ok():
    return INJECT_BIN.exists() and brain.available()


async def health(_req):
    return web.json_response({"ok": True, "busy": session_lock.locked(),
                              "binary": BIN.exists(), "voices": len(voices()),
                              "models": models(), "brain": brain_ok()})


def _no_core():
    # a crashing 7B model writes multi-GB core files; we never read them
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


async def spawn(*cmd, stdin=None, stdout=None, stderr=None, env=None):
    return await asyncio.create_subprocess_exec(
        *cmd, stdin=stdin, stdout=stdout, stderr=stderr, env=env,
        start_new_session=True, preexec_fn=_no_core)


async def stop(proc, name):
    if proc is None or proc.returncode is not None:
        return
    for sig, wait in ((signal.SIGINT, 2.0), (signal.SIGTERM, 2.0), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), wait)
            log(f"stopped {name} with {sig.name}")
            return
        except asyncio.TimeoutError:
            continue


async def ws_handler(req):
    ws = web.WebSocketResponse(max_msg_size=1 << 20, heartbeat=20)
    await ws.prepare(req)

    voice = req.query.get("voice", "NATF0")
    if voice not in voices():
        await ws.send_json({"type": "error", "message": f"unknown voice {voice!r}"})
        await ws.close()
        return ws
    avail = models()
    mdl = req.query.get("model") or (avail[0] if avail else "q4")
    if mdl not in avail:
        await ws.send_json({"type": "error", "message": f"unknown model {mdl!r}"})
        await ws.close()
        return ws
    prompt = (req.query.get("prompt") or "").strip()[:MAX_PROMPT]
    seed = req.query.get("seed")
    use_brain = req.query.get("brain") == "1" and brain_ok()
    try:
        pace = min(2.5, max(0.5, float(req.query.get("pace", "1"))))
    except ValueError:
        pace = 1.0

    if session_lock.locked():
        await ws.send_json({"type": "error", "message": "busy: another conversation is running"})
        await ws.close()
        return ws

    async with session_lock:
        env = dict(os.environ, PULSE_SERVER=PULSE, SDL_AUDIODRIVER="pulseaudio")
        cmd = ["stdbuf", "-o0", str(INJECT_BIN if use_brain else BIN), "-d", "Vulkan0",
               "-r", str(MODELS), *MODEL_ARGS[mdl], "-v", str(VOICES / f"{voice}.gguf")]
        if use_brain:
            cmd += ["--inject-stdin"]
        if prompt:
            cmd += ["-p", prompt]
        if seed and re.fullmatch(r"\d{1,9}", seed):
            cmd += ["-s", seed]

        peer = req.headers.get("X-Forwarded-For", req.remote)
        log(f"session start peer={peer} model={mdl} voice={voice} brain={use_brain} prompt={len(prompt)} chars")
        await ws.send_json({"type": "status", "message": "loading model"})

        pcm = ["--format=s16le", f"--rate={RATE}", "--channels=1", "--latency-msec=20", "--raw"]
        play = await spawn("pacat", "--playback", "--device=pp_in", *pcm,
                           stdin=asyncio.subprocess.PIPE, env=env)
        rec = await spawn("pacat", "--record", "--device=pp_out.monitor", *pcm,
                          stdout=asyncio.subprocess.PIPE, env=env)
        errlog = open("/tmp/personaplex-stderr.log", "ab")
        model = await spawn(*cmd, stdout=asyncio.subprocess.PIPE, stderr=errlog, env=env,
                            stdin=asyncio.subprocess.PIPE if use_brain else None)
        br = brain.Brain(ws, model.stdin, log, pace=pace) if use_brain else None
        t0 = time.monotonic()
        tasks = []

        async def pump_text():
            buf = ""
            ready = False
            while True:
                chunk = await model.stdout.read(256)
                if not chunk:
                    break
                buf += chunk.decode("utf-8", "replace")
                if not ready:
                    if "ready\n" in buf:
                        ready = True
                        _, buf = buf.split("ready\n", 1)
                        log(f"model ready in {time.monotonic() - t0:.2f}s")
                        await ws.send_json({"type": "ready",
                                            "load_s": round(time.monotonic() - t0, 2)})
                    else:
                        continue
                if buf:
                    if br:
                        br.on_voice_text(buf)
                    await ws.send_json({"type": "text", "data": buf, "t": time.monotonic()})
                    buf = ""
            rc = await model.wait()
            log(f"model exited rc={rc}")
            if not ws.closed:
                await ws.send_json({"type": "ended", "rc": rc})
                await ws.close()

        async def pump_audio_out():
            while True:
                data = await rec.stdout.readexactly(CHUNK_BYTES)
                if ws.closed:
                    break
                await ws.send_bytes(data)

        tasks.append(asyncio.create_task(pump_text()))
        tasks.append(asyncio.create_task(pump_audio_out()))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    play.stdin.write(msg.data)
                    await play.stdin.drain()
                    if br:
                        br.feed(msg.data)
                elif msg.type == WSMsgType.TEXT and msg.data == "stop":
                    break
                elif msg.type == WSMsgType.TEXT and br and msg.data.startswith("{"):
                    try:
                        j = json.loads(msg.data)
                        if j.get("type") == "pace":
                            br.set_pace(j.get("value", 1.0))
                    except (ValueError, TypeError):
                        pass
                elif msg.type == WSMsgType.ERROR:
                    break
        except (ConnectionResetError, BrokenPipeError) as e:
            log(f"stream error: {e!r}")
        finally:
            for t in tasks:
                t.cancel()
            if br:
                await br.close()
            await stop(model, "personaplex")
            await stop(play, "pacat-play")
            await stop(rec, "pacat-rec")
            errlog.close()
            log(f"session end after {time.monotonic() - t0:.1f}s")
            if not ws.closed:
                await ws.close()
    return ws


def main():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/voices", list_voices)
    app.router.add_get("/health", health)
    app.router.add_get("/ws", ws_handler)
    log(f"bridge on :{PORT}  binary={BIN.exists()}  voices={len(voices())}")
    web.run_app(app, host="0.0.0.0", port=PORT, access_log=None, print=None)


if __name__ == "__main__":
    main()
