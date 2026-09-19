# PersonaPlex on the RX 7900 XTX

This repo runs NVIDIA's [PersonaPlex-7B](https://huggingface.co/nvidia/personaplex-7b-v1), a full-duplex speech-to-speech model, in real time on an **AMD Radeon RX 7900 XTX** (Vulkan, no ROCm). It uses [moshi.cpp](https://github.com/Codes4Fun/moshi.cpp) with a small patch that lets a bigger LLM **whisper answers into PersonaPlex's inner monologue**, so PersonaPlex speaks them in its own voice while staying full duplex.

It's a homelab project: Proxmox, an LXC container with the XTX passed through, and a second GPU (R9700) serving Qwen3.6-35B through LiteLLM.

## What's here

| path | what it is |
|---|---|
| `moshi-patch/inject.patch` | A patch to moshi.cpp v0.8.0-beta. It adds a forced-text queue to `moshi_lmgen_step` and `SAY` / `PAUSE` / `CLEAR` commands on stdin (`--inject-stdin`). |
| `moshi-patch/build.sh` | Builds only `libmoshi` and `personaplex` in a throwaway `ubuntu:24.04` container, against the release's own ggml libs. |
| `bridge/bridge.py`, `bridge/index.html` | A WebSocket bridge and browser page: talk to PersonaPlex from a phone or laptop. You can pick the model (q8 / q4 / bf16) and the voice, and switch the brain on or off. |
| `bridge/brain.py` | The **brain**: VAD → speech-to-text → LLM (PASS / answer / web search / weather) → `SAY` into PersonaPlex. |
| `brain/stt_server.py` | A small HTTP service with faster-whisper STT (CPU), DuckDuckGo search (`ddgs`) and Open-Meteo weather. |
| `tools/bf16gguf.py`, `tools/verify_bf16.py` | A streaming bf16 GGUF writer. It turns a 77 s load into 12 s without the ~27 GB of RAM moshi.cpp's own `-g` writer needs. |
| `tools/gguftypes.py`, `tools/voice_bf16_to_f32.py` | A dependency-free GGUF tensor-type lister, and a converter for voice embeddings (RADV has no bf16→f32 copy). |
| `tools/builder` | "Builder mode" for the Proxmox host: frees RAM and VRAM for experiments, and restores everything afterwards. |
| `tests/brain_test.py`, `tests/stt_reply.py` | An end-to-end test. Kokoro TTS speaks questions into the bridge in real time, and the reply audio is transcribed. |

## Settings

Copy `bridge/brain.example.json` to `bridge/brain.json` (git-ignored) and set your city. It's the default place for weather questions and tells the LLM where you live. Or set the `BRAIN_HOME` env var. The STT/search service takes `WEATHER_HOME` for the same purpose. The router key goes in `bridge/.litellm-key` (git-ignored). The service URLs at the top of `brain.py` point at my LAN; change them for yours.

## Measured (RX 7900 XTX, Vulkan/RADV)

| model | fps (12.5 = real time) | VRAM | load |
|---|---|---|---|
| q4_k (Codes4Fun) | 33.8–34.2 | 5.4 GB | 2.4 s |
| q8_0 (local conversion) | 29.8–30.5 | 9.3 GB | 5–7 s |
| bf16 (full size) | 22.5–22.8 | 16.8 GB | 12 s with our GGUF (77 s from safetensors) |

With the brain on, the answer is queued about **1.9 s** after you stop talking for plain questions, and **3.7–4 s** for web or weather lookups. That's after a 0.7 s end-of-speech wait, and speech-to-text takes about 1.3 s of it.

## How the injection works

Moshi-family models produce a text token every 80 ms frame (the inner monologue), and the audio codebooks are conditioned on it. The patch overrides that sampled text token from a queue:

- **Paced mode (default):** a queued word replaces a word the model was about to say anyway. While the model is silent, it gets a nudge after 6 frames. Forcing a token every frame (strict mode) garbles the speech.
- `PAUSE n` forces silence for n frames. The brain uses it to hold PersonaPlex quiet while it thinks, so PersonaPlex doesn't blurt out its own made-up answer first.
- `CLEAR` drops anything still queued. It's used when you start talking again (barge-in).

## Lessons learned

- PersonaPlex answers **instantly and confidently, and it's often wrong**. A stricter role prompt made this worse. Holding it silent until the LLM has decided (PASS lets it go) fixed it.
- Mark PersonaPlex's own lines as **unreliable** in the LLM's transcript. Otherwise the LLM builds on its made-up topics.
- Have the LLM **write numbers as words** ("ninety-eight degrees"). With digits, the audio sometimes said a different number from the forced text (2 % → "20 %", 98 → "78").
- A filler ("Let me check.") without a forced silence afterwards teaches PersonaPlex to invent its own "lookup results".

## Status

This is an experiment, not a product. Weights aren't included: PersonaPlex is gated on Hugging Face under NVIDIA's license. `moshi-patch/inject.patch` modifies [moshi.cpp](https://github.com/Codes4Fun/moshi.cpp) (MIT, © Codes4Fun).
