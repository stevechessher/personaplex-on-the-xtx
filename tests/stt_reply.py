import sys, wave, numpy as np
from faster_whisper import WhisperModel
m = WhisperModel("small.en", device="cpu", compute_type="int8")
w = wave.open(sys.argv[1]); a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
sr = w.getframerate()
a16 = np.interp(np.linspace(0, len(a) - 1, int(len(a) * 16000 / sr)), np.arange(len(a)), a).astype(np.float32)
segs, _ = m.transcribe(a16, language="en", beam_size=5, vad_filter=True)
for s in segs:
    print(f"  [{s.start:6.2f}-{s.end:6.2f}] {s.text.strip()}")
