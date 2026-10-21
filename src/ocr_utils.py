import re, sys
from io import StringIO
from rapidfuzz.distance import Levenshtein

def clean_text(t):
    if t is None: return ""
    t = str(t)
    t = re.sub(r"[#*\-`_>]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.replace("ي", "ی").replace("ك", "ک").strip()

def edit_counts(ref, hyp):
    """(char_errors, ref_chars, word_errors, ref_words) for one pair."""
    return (Levenshtein.distance(ref, hyp), len(ref),
            Levenshtein.distance(ref.split(), hyp.split()), len(ref.split()))

def corpus_rates(refs, hyps):
    a = [edit_counts(r, h) for r, h in zip(refs, hyps)]
    ce, cn = sum(x[0] for x in a), sum(x[1] for x in a)
    we, wn = sum(x[2] for x in a), sum(x[3] for x in a)
    return ce / cn, we / wn

_NOISE = ["directly resize", "PATCHES", "====", "BASE:", "NO PATCHES", "torch.Size", "%|",
          "image size", "valid image", "output texts", "compression", "save results", "image:", "other:"]

def infer_text(model, tokenizer, img_path, prompt, **kw):
    """DeepSeek-OCR prints its answer instead of returning it, so capture stdout."""
    buf, old = StringIO(), sys.stdout
    sys.stdout = buf
    try:
        model.infer(tokenizer, prompt=prompt, image_file=str(img_path), output_path="tmp_out", **kw)
    finally:
        sys.stdout = old
    keep = []
    for line in buf.getvalue().split("\n"):
        s = line.strip()
        if not s or any(n in s for n in _NOISE): continue
        if s.endswith(("it/s]", "it]", "s/it]")): continue
        keep.append(line)
    return "\n".join(keep).strip()
