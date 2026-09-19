# PersonaPlex on the RX 7900 XTX

This repo runs NVIDIA's [PersonaPlex-7B](https://huggingface.co/nvidia/personaplex-7b-v1), a full-duplex speech-to-speech model, in real time on an **AMD Radeon RX 7900 XTX** (Vulkan, no ROCm). It uses [moshi.cpp](https://github.com/Codes4Fun/moshi.cpp) with a small patch that lets a bigger LLM **whisper answers into PersonaPlex's inner monologue**, so PersonaPlex speaks them in its own voice while staying full duplex.

It's a homelab project: Proxmox, an LXC container with the XTX passed through, and a second GPU (R9700) serving Qwen3.6-35B through LiteLLM.

## What's here

| path | what it is |
|---|---|
| `moshi-patch/inject.patch` | A patch to moshi.cpp v0.8.0-beta. It adds a forced-text queue to `moshi_lmgen_step` and `SAY` / `PAUSE` / `CLEAR` commands on stdin (`--inject-stdin`). |
| `moshi-patch/build.sh` | Builds only `libmoshi` and `personaplex` in a throwaway `ubuntu:24.04` container, against the release's own ggml libs. |
| `bridge/bridge.py`, `bridge/index.html` | A WebSocket bridge and browser page: talk to PersonaPlex from a phone or laptop. You can pick the model (q8 / q4 / bf16) and the voice, and switch the brain on or off. |
| `bridge/brain.py`, `bridge/numwords.py` | The **brain**: VAD → streaming speech-to-text → LLM (PASS / answer / web search / weather) → `SAY` into PersonaPlex. |
| `asr/Dockerfile` | Runs NVIDIA's [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) streaming ASR (Nemotron Speech Streaming 0.6B) on the XTX over Vulkan. |
| `brain/stt_server.py` | A small HTTP service: faster-whisper STT (CPU fallback), DuckDuckGo search (`ddgs`) and Open-Meteo weather. |
| `tools/bf16gguf.py`, `tools/verify_bf16.py` | A streaming bf16 GGUF writer. It turns a 77 s load into 12 s without the ~27 GB of RAM moshi.cpp's own `-g` writer needs. |
| `tools/gguftypes.py`, `tools/voice_bf16_to_f32.py` | A dependency-free GGUF tensor-type lister, and a converter for voice embeddings (RADV has no bf16→f32 copy). |
| `tools/builder` | "Builder mode" for the Proxmox host: frees RAM and VRAM for experiments, and restores everything afterwards. |
| `tests/brain_test.py`, `tests/stt_reply.py` | An end-to-end test. Kokoro TTS speaks questions into the bridge in real time, and the reply audio is transcribed. |
| `tests/brain_test_pace.py` | The same, for turn-taking: a sentence with a 1.1 s pause in the middle, "what can you do?", "that's not what I meant", and "wait a little longer". |
| `tools/tts_ttfa.py` | Time to first audio for any OpenAI-style `/v1/audio/speech` endpoint (streamed PCM). |

## Settings

Copy `bridge/brain.example.json` to `bridge/brain.json` (git-ignored) and set your city. It's the default place for weather questions and tells the LLM where you live. Or set the `BRAIN_HOME` env var. The STT/search service takes `WEATHER_HOME` for the same purpose. The router key goes in `bridge/.litellm-key` (git-ignored). The service URLs at the top of `brain.py` point at my LAN; change them for yours.

## Measured (RX 7900 XTX, Vulkan/RADV)

| model | fps (12.5 = real time) | VRAM | load |
|---|---|---|---|
| q4_k (Codes4Fun) | 33.8–34.2 | 5.4 GB | 2.4 s |
| q8_0 (local conversion) | 29.8–30.5 | 9.3 GB | 5–7 s |
| bf16 (full size) | 22.5–22.8 | 16.8 GB | 12 s with our GGUF (77 s from safetensors) |

With the brain on, and counting from the end-of-turn wait (see Turn-taking below):
- the answer is queued about **0.5–1 s** later for plain questions;
- about **2.3–3.5 s** later for web or weather lookups;
- PASS (small talk) is released about **0.4 s** later.

Speech-to-text is streamed while you talk, so it adds only 0–150 ms. The first version used faster-whisper after you stopped, which took about 1.3 s.

### Streaming ASR on AMD (NeMo-Speech.cpp v0.1.0, Vulkan)

| | whisper small.en (CPU, after you stop) | Nemotron streaming 0.6B (XTX, Vulkan) |
|---|---|---|
| words appear | only after you stop | 0.1–0.2 s after each word |
| final text after speech end | ~1.3 s (+0.7 s VAD) | ~0.6 s (server endpointing at 500 ms) |
| PersonaPlex q8 speed with it running | — | 27.8 → 27.6 fps (real time is 12.5) |
| VRAM | 0 | ~1.2 GB |

**RADV gotcha:** the prebuilt Linux Vulkan archive bundles an old `libstdc++.so.6`, which stops Mesa's RADV driver from loading, so ggml silently finds only the CPU ("no matching GPU device"). Rename `lib/libstdc++.so.6` and `lib/libgcc_s.so.1` in the install to `*.bundled`, and the XTX shows up.

### Text-to-speech on the XTX (for a cascade instead of PersonaPlex)

Kokoro-82M, [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) ROCm image, on the XTX (+1.4 GB VRAM), PersonaPlex not running:

| text | first audio | all audio |
|---|---|---|
| 5 words (1.7 s of speech) | 179 ms | 179 ms |
| 2 sentences (9.3 s of speech) | 308 ms | 309 ms |

The whole reply arrives at once, about 30× faster than real time, so sentence-by-sentence streaming wouldn't gain anything here.

## Turn-taking (brain rev 9)

A fixed 0.7 s silence cut people off mid-thought and made answers come faster than people talk. Now the wait depends on whether you *sound* finished, using the streaming ASR's punctuation and your last word:

- **~0.6 s** after a finished sentence ("…when do I get there?");
- **~1.6 s** after a trailing word ("and", "the", "I", "um") or no punctuation yet;
- **~1.0 s** when there's no transcript to judge by.

PersonaPlex is held quiet during those thinking pauses. A **pace** setting (0.5–2.5×) scales all three: there's a slider on the page, or you can say "wait a little longer" / "you can go faster".

In the test, a 1.1 s pause mid-sentence stayed one turn in one run out of two (Kokoro's own padding makes the real gap nearer 1.6 s). When it splits, the first half is dropped as stale and the answer still comes out right.

## How the injection works

Moshi-family models produce a text token every 80 ms frame (the inner monologue), and the audio codebooks are conditioned on it. The patch overrides that sampled text token from a queue:

- **Paced mode (default):** a queued word replaces a word the model was about to say anyway. While the model is silent, it gets a nudge after 6 frames. Forcing a token every frame (strict mode) garbles the speech.
- `PAUSE n` forces silence for n frames. The brain uses it to hold PersonaPlex quiet while it thinks, so PersonaPlex doesn't blurt out its own made-up answer first.
- `CLEAR` drops anything still queued. It's used when you start talking again (barge-in).

## Lessons learned

- PersonaPlex answers **instantly and confidently, and it's often wrong**. A stricter role prompt made this worse. Holding it silent until the LLM has decided (PASS lets it go) fixed it.
- Mark PersonaPlex's own lines as **unreliable** in the LLM's transcript. Otherwise the LLM builds on its made-up topics.
- Have the LLM **write numbers as words** ("ninety-eight degrees"). With digits, the audio sometimes said a different number from the forced text (2 % → "20 %", 98 → "78").
- Streaming ASR spells numbers out ("two fifteen"), and the LLM (reasoning off) then got time math wrong. `numwords.py` converts them to digits. The LLM also writes its working as `CALC: … SAY: …`, and only the SAY part is spoken; that took a 7-question arithmetic set from mostly wrong to 7/7 for +0.2–0.8 s.
- "When in doubt, PASS" made the LLM pass on real questions once PersonaPlex had started (wrongly) answering. "PASS only for pure small talk" fixed it.
- A filler ("Let me check.") without a forced silence afterwards teaches PersonaPlex to invent its own "lookup results".
- The LLM (reasoning off) kept treating "what can you do?" as small talk and passing it, even with a capabilities list in its prompt. PersonaPlex then invented its own limits ("I only answer time questions"). A regex and a fixed answer fixed it.
- "That's not what I meant" needs a question back ("What did you mean?"), not a guess. Otherwise the conversation loops on the wrong topic.

## Status

This is an experiment, not a product. Weights aren't included: PersonaPlex is gated on Hugging Face under NVIDIA's license. `moshi-patch/inject.patch` modifies [moshi.cpp](https://github.com/Codes4Fun/moshi.cpp) (MIT, © Codes4Fun). The Nemotron ASR model is under the NVIDIA Open Model License; NeMo-Speech.cpp is Apache-2.0.
