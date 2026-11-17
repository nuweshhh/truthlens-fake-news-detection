"""Text cleaning and tokenization used in the TruthLens Week 2 pipeline."""
import html
import re

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
DATELINE = re.compile(r"^[A-Z][A-Za-z .,'-]{1,40}\(Reuters\)\s*-\s*")


def clean(text):
    """Return cleaned text, or None when the text is missing or empty."""
    if text is None or not text.strip():
        return None
    t = html.unescape(text)
    t = re.sub(r"<[^>]+>", " ", t)                       # HTML tags
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)         # URLs
    t = DATELINE.sub("", t.strip())                      # agency dateline (leakage)
    t = re.sub(r"21st Century Wire says", " ", t, flags=re.I)
    t = EMOJI.sub(" ", t)
    t = re.sub(r"([!?])\1+", r"\1", t)                   # repeated ! and ?
    t = re.sub(r"\.{2,}", "...", t)                      # normalise dot runs
    return re.sub(r"\s+", " ", t).strip()


def tokens(text):
    """Simple lowercase word tokenizer for the TF-IDF baselines."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())
