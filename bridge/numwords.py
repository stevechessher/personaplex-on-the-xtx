"""Spoken numbers -> digits for the brain.

The streaming ASR spells numbers out ("two fifteen", "one hour fifty"). With words the LLM got the
train question wrong every time (3:05); with digits it's right (4:05). Measured 2026-09-19.
"""
import re
UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine".split())}
TEENS = {w: i + 10 for i, w in enumerate("ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: (i + 2) * 10 for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000}
NUMWORDS = set(UNITS) | set(TEENS) | set(TENS) | set(SCALES)


def _chunks(words):
    """Split a run of number words into separately spoken numbers: 'two fifteen' -> [2, 15]."""
    out, cur, last = [], None, None          # last: kind of the previous word
    for w in words:
        if w in SCALES:
            if cur is None:
                cur = 1
            if SCALES[w] == 100:
                cur = (cur // 1000) * 1000 + (cur % 1000) * 100 if cur % 1000 else cur * 100
            else:
                cur *= SCALES[w]
            last = "scale"
            continue
        v = UNITS.get(w, TEENS.get(w, TENS.get(w)))
        kind = "unit" if w in UNITS else "teen" if w in TEENS else "tens"
        joinable = cur is not None and (
            last == "scale" or (last == "tens" and kind == "unit" and cur % 10 == 0))
        if joinable:
            cur += v
        else:
            if cur is not None:
                out.append(cur)
            cur = v
        last = kind
    if cur is not None:
        out.append(cur)
    return out


def normalize_numbers(text):
    """Turn spelled-out numbers from speech-to-text into digits the LLM can do arithmetic on."""
    toks = re.findall(r"[A-Za-z']+|[^A-Za-z']+", text)
    out, i = [], 0
    while i < len(toks):
        w = toks[i].lower()
        if w in NUMWORDS and w != "hundred" and w != "thousand" and w != "million":
            # collect a run of number words (spaces/hyphens between, 'and' inside, 'oh' as zero)
            run, j = [], i
            while j < len(toks):
                t = toks[j].lower()
                if t in NUMWORDS or (t in ("oh", "and") and run):
                    run.append(t); j += 1
                elif toks[j] in (" ", "-") and j + 1 < len(toks):
                    j += 1
                else:
                    break
            while run and run[-1] in ("oh", "and"):
                run.pop()
            end = i
            # recompute where the run ends in toks (skip back over trailing oh/and + spaces)
            k, n = i, 0
            while n < len(run):
                if toks[k].lower() == run[n]:
                    n += 1
                k += 1
            end = k
            words = [r for r in run if r != "and"]
            if "oh" in words:                          # "four oh five" -> 4:05
                a, b = words[:words.index("oh")], words[words.index("oh") + 1:]
                ca, cb = _chunks(a), _chunks(b)
                if len(ca) == 1 and len(cb) == 1 and ca[0] <= 12 and cb[0] < 10:
                    s = f"{ca[0]}:0{cb[0]}"
                else:
                    s = " ".join(str(x) for x in ca + [0] + cb)
            else:
                c = _chunks(words)
                if len(c) == 2 and 1 <= c[0] <= 12 and 10 <= c[1] <= 59:
                    s = f"{c[0]}:{c[1]:02d}"                     # "two fifteen" -> 2:15
                elif len(c) == 2 and c[0] in (19, 20) and 0 <= c[1] <= 99:
                    s = f"{c[0]}{c[1]:02d}"                      # "twenty twenty six" -> 2026
                else:
                    s = " ".join(str(x) for x in c)
            out.append(s)
            i = end
        else:
            out.append(toks[i]); i += 1
    return "".join(out)


if __name__ == "__main__":
    for t in ["If a train leaves at two fifteen and the trip takes one hour fifty, what time does it arrive?",
              "I have three hundred twenty dollars and spend one hundred forty five. How much is left?",
              "A movie starts at seven forty and runs two hours thirty five minutes. When does it end?",
              "It arrives at four oh five.", "in twenty twenty six", "one of them", "I'm having a pretty relaxed evening",
              "roughly how many people live there", "the score was twenty one to seven", "two thousand five hundred people",
              "call me at nine", "twelve thirty", "Someone said it"]:
        print(f"{t!r}\n   -> {normalize_numbers(t)!r}")
