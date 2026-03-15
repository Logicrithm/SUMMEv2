"""
summarizer.py  ─  100% Free Summarizer (No Paid API Keys)
==========================================================
PRIORITY ORDER (tries each in order, falls back automatically):

  1. GROQ        — Free account at console.groq.com, no credit card.
                   Uses Llama 3.1 70B. Speed: ~1-2s. Best quality.

  2. GEMINI      — Free Google account. 1500 requests/day free.
                   Uses gemini-1.5-flash. Speed: ~2-3s. Great quality.

  3. HUGGINGFACE — Free HF account at huggingface.co. Uses BART.
                   Speed: ~3-8s (cold start). Good English quality.

  4. EXTRACTIVE  — Zero signup, zero install. Pure Python stdlib.
                   Works offline. Speed: <0.5s. Decent quality.

Set in your .env file (you only need ONE):
  GROQ_API_KEY=gsk_...
  GEMINI_API_KEY=AIza...
  HF_API_KEY=hf_...
"""

import os
import re
import time
import math
import json
import logging
import hashlib
import urllib.request
import urllib.error
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CHUNK_SIZE   = 4000
OVERLAP      = 200
MAX_WORKERS  = 4
CACHE_DIR    = os.path.join("outputs", "summary_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

LANGUAGE_NAMES = {
    "hi": "Hindi",    "ta": "Tamil",     "te": "Telugu",   "bn": "Bengali",
    "ml": "Malayalam","kn": "Kannada",   "mr": "Marathi",  "gu": "Gujarati",
    "pa": "Punjabi",  "ur": "Urdu",      "en": "English",  "fr": "French",
    "es": "Spanish",  "de": "German",    "ja": "Japanese", "zh": "Chinese",
    "ar": "Arabic",   "ru": "Russian",   "pt": "Portuguese","ko": "Korean",
}

# Load API keys
GROQ_KEY   = os.environ.get("GROQ_API_KEY", "").strip()
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
HF_KEY     = os.environ.get("HF_API_KEY", "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# 1. GROQ  (Free — console.groq.com)
# ─────────────────────────────────────────────────────────────────────────────
def _call_groq(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model":       "llama-3.1-70b-versatile",
        "messages":    messages,
        "max_tokens":  1200,
        "temperature": 0.3,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {GROQ_KEY}",
        },
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2. GEMINI  (Free — aistudio.google.com — 1500 req/day)
# ─────────────────────────────────────────────────────────────────────────────
def _call_gemini(prompt: str, system: str = "") -> str:
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    payload = json.dumps({
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"maxOutputTokens": 1200, "temperature": 0.3}
    }).encode("utf-8")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={GEMINI_KEY}"
    )
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# 3. HUGGING FACE INFERENCE API  (Free — huggingface.co)
# ─────────────────────────────────────────────────────────────────────────────
HF_MODEL_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

def _call_huggingface(text: str) -> str:
    words = text.split()
    if len(words) > 750:
        text = " ".join(words[:750])

    payload = json.dumps({
        "inputs":     text,
        "parameters": {
            "max_length":    300,
            "min_length":    80,
            "do_sample":     False,
            "early_stopping": True,
        },
        "options": {"wait_for_model": True}
    }).encode("utf-8")

    req = urllib.request.Request(
        HF_MODEL_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {HF_KEY}",
            "Content-Type":  "application/json",
        },
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        if isinstance(data, list):
            return data[0].get("summary_text", "").strip()
        return str(data).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 4. EXTRACTIVE FALLBACK  (Zero dependencies — always works)
# ─────────────────────────────────────────────────────────────────────────────
def _extractive_summary(text: str, num_sentences: int = 12) -> str:
    sentences = re.split(r'(?<=[.!?|।॥])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.split()) >= 5]

    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    tokenized = [re.findall(r'\b\w+\b', s.lower()) for s in sentences]
    N = len(sentences)

    def score(i: int) -> float:
        words = tokenized[i]
        if not words:
            return 0.0
        tf = Counter(words)
        max_tf = max(tf.values())
        idf = {
            w: math.log((N + 1) / (1 + sum(1 for t in tokenized if w in t)))
            for w in set(words)
        }
        tfidf = sum((c / max_tf) * idf[w] for w, c in tf.items())
        pos = i / N
        pos_boost = 1.4 if pos < 0.1 else (1.2 if pos > 0.9 else 1.0)
        len_boost  = min(1.0, len(words) / 25)
        return tfidf * pos_boost * (0.75 + 0.25 * len_boost)

    top = sorted(sorted(range(N), key=score, reverse=True)[:num_sentences])
    return " ".join(sentences[i] for i in top)


def _format_extractive(text: str, title: str, language: str) -> str:
    lang = LANGUAGE_NAMES.get(language, "English")
    return (
        f"**{title}**\n\n"
        f"📝 Summary ({lang}):\n\n{text}\n\n"
        f"---\n"
        f"*Generated using offline extractive analysis. "
        f"For AI summaries, add a free API key (GROQ/GEMINI/HF) to your .env file.*"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────
def _cache_key(text: str, language: str, style: str) -> str:
    return hashlib.md5(f"{text[:500]}{language}{style}{len(text)}".encode()).hexdigest()

def _load_cache(key: str) -> Optional[str]:
    try:
        with open(os.path.join(CACHE_DIR, f"{key}.txt"), "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None

def _save_cache(key: str, text: str):
    with open(os.path.join(CACHE_DIR, f"{key}.txt"), "w", encoding="utf-8") as f:
        f.write(text)


# ─────────────────────────────────────────────────────────────────────────────
# Provider router
# ─────────────────────────────────────────────────────────────────────────────
def _which_provider() -> str:
    if GROQ_KEY:   return "groq"
    if GEMINI_KEY: return "gemini"
    if HF_KEY:     return "huggingface"
    return "extractive"

FALLBACK_CHAINS = {
    "groq":        ["groq",   "gemini", "huggingface", "extractive"],
    "gemini":      ["gemini", "huggingface", "extractive"],
    "huggingface": ["huggingface", "extractive"],
    "extractive":  ["extractive"],
}

def _call_provider(prompt: str, system: str, language: str, provider: str) -> str:
    if provider == "groq":
        return _call_groq(prompt, system)
    elif provider == "gemini":
        return _call_gemini(prompt, system)
    elif provider == "huggingface":
        return _call_huggingface(prompt)
    else:
        return _extractive_summary(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_prompt(text: str, language: str, style: str, is_final: bool) -> tuple[str, str]:
    lang = LANGUAGE_NAMES.get(language, "English")
    if is_final:
        system = (
            f"You are an expert video summarizer. Always respond in {lang}. "
            "Produce a clear, well-structured, professional summary."
        )
        user = (
            f"Combine these partial summaries into one unified video summary in {lang}.\n"
            f"Style: {style}\n\n"
            "Format:\n**[Video Topic]**\n\n"
            "📌 Key Points:\n• ...\n• ...\n• ...\n\n"
            "📝 Summary:\n[2-3 paragraphs]\n\n"
            "🎯 Main Takeaway:\n[One sentence]\n\n"
            f"PARTIAL SUMMARIES:\n{text}"
        )
    else:
        system = (
            f"Summarize video transcript segments. Respond in {lang}. "
            "Extract key facts, arguments, and conclusions. Be concise."
        )
        user = (
            f"Summarize this transcript in {lang}. Style: {style}.\n"
            f"Focus on key facts, main arguments, important examples.\n\n"
            f"TRANSCRIPT:\n{text}"
        )
    return system, user


# ─────────────────────────────────────────────────────────────────────────────
# Text chunker
# ─────────────────────────────────────────────────────────────────────────────
def _chunk_text(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            snap = text.rfind(". ", start + CHUNK_SIZE // 2, end)
            if snap != -1:
                end = snap + 2
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - OVERLAP
    return [c for c in chunks if c]


# ─────────────────────────────────────────────────────────────────────────────
# Parallel chunk summarizer
# ─────────────────────────────────────────────────────────────────────────────
def _summarize_chunks(chunks: list[str], language: str, style: str, provider: str) -> list[str]:
    chain   = FALLBACK_CHAINS.get(provider, ["extractive"])
    results = [None] * len(chunks)

    def worker(args):
        idx, chunk = args
        system, prompt = _build_prompt(chunk, language, style, is_final=False)
        for p in chain:
            try:
                out = _call_provider(prompt, system, language, p)
                if out.strip():
                    return idx, out
            except Exception as e:
                logger.debug(f"Chunk {idx} [{p}] failed: {e}")
        return idx, _extractive_summary(chunk)

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(chunks))) as pool:
        futures = {pool.submit(worker, (i, c)): i for i, c in enumerate(chunks)}
        for future in as_completed(futures):
            idx, text = future.result()
            results[idx] = text

    return [r for r in results if r]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def summarize(
    transcript: str,
    language:   str = "en",
    style:      str = "professional",
    use_cache:  bool = True,
    title:      str = "Video Summary",
) -> dict:
    t0 = time.time()
    transcript = (transcript or "").strip()

    if not transcript:
        return {"summary": "No transcript found.", "word_count": 0,
                "processing_time": 0, "method": "none",
                "chunks_processed": 0, "provider": "none"}

    ckey = _cache_key(transcript, language, style)
    if use_cache:
        cached = _load_cache(ckey)
        if cached:
            return {"summary": cached, "word_count": len(cached.split()),
                    "processing_time": 0.0, "method": "cache",
                    "chunks_processed": 0, "provider": "cache"}

    provider = _which_provider()
    chain    = FALLBACK_CHAINS.get(provider, ["extractive"])
    logger.info(f"Summarizing {len(transcript.split())} words | lang={language} | provider={provider}")

    # Short transcript
    if len(transcript) <= CHUNK_SIZE:
        system, prompt = _build_prompt(transcript, language, style, is_final=True)
        summary, used = None, "extractive"
        for p in chain:
            try:
                out = _call_provider(prompt, system, language, p)
                if out.strip():
                    summary, used = out, p
                    break
            except Exception as e:
                logger.warning(f"[{p}] failed: {e}")
        if not summary:
            summary = _format_extractive(_extractive_summary(transcript), title, language)
        chunks_used = 1

    # Long transcript
    else:
        chunks  = _chunk_text(transcript)
        partial = _summarize_chunks(chunks, language, style, provider)
        combined = "\n\n---\n\n".join(
            f"[Part {i+1}/{len(partial)}]:\n{s}" for i, s in enumerate(partial)
        )
        system, prompt = _build_prompt(combined, language, style, is_final=True)
        summary, used = None, "extractive"
        for p in chain:
            try:
                out = _call_provider(prompt, system, language, p)
                if out.strip():
                    summary = out
                    used    = f"{p}_hierarchical"
                    break
            except Exception as e:
                logger.warning(f"Final merge [{p}] failed: {e}")
        if not summary:
            raw     = _extractive_summary(combined, num_sentences=18)
            summary = _format_extractive(raw, title, language)
            used    = "extractive_hierarchical"
        chunks_used = len(chunks)

    if use_cache:
        _save_cache(ckey, summary)

    elapsed = round(time.time() - t0, 2)
    logger.info(f"Done in {elapsed}s via [{used}]")

    return {
        "summary":          summary,
        "word_count":       len(summary.split()),
        "processing_time":  elapsed,
        "method":           used,
        "chunks_processed": chunks_used,
        "provider":         provider,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Class Wrapper for existing App Backend
# ─────────────────────────────────────────────────────────────────────────────
class Summarizer:
    """Wrapper to make the multi-provider logic compatible with existing VidSummarize app.py"""
    def __init__(self, model_name=None):
        print(f"[Summarizer] Initialized Multi-Provider Summarizer. Active Provider: {_which_provider()}")

    def summarize_text(self, text: str, max_length: int = 300, language: str = "en") -> Dict:
        try:
            res = summarize(
                transcript=text, 
                language=language, 
                style="professional",
                use_cache=True, 
                title="VidSummarize Generation"
            )
            return {
                'success': True, 
                'summary': res['summary'], 
                'word_count': res['word_count'],
                'provider': res['provider'], 
                'method': res['method'],
                'processing_time': res['processing_time']
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def summarize(self, text: str, max_length: int = 300) -> str:
        res = summarize(transcript=text, use_cache=True)
        return res['summary']

if __name__ == "__main__":
    s = Summarizer()
    print("Multi-Provider Summarizer Ready.")
