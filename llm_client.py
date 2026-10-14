"""Tiny LLM client using only the standard library (no pip install needed).

Pick a provider with environment variables:
  Google Gemini (free tier):  set GEMINI_API_KEY=...
  Anthropic Claude:           set ANTHROPIC_API_KEY=...
Optional: set LLM_MODEL=... to override the default model name.
"""
import json
import os
import urllib.request


def _post(url, headers, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def ask_llm(prompt):
    """Send one prompt, return the model's text reply."""
    if os.environ.get("GEMINI_API_KEY"):
        model = os.environ.get("LLM_MODEL", "gemini-2.0-flash")
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent?key={os.environ['GEMINI_API_KEY']}")
        out = _post(url, {"Content-Type": "application/json"},
                    {"contents": [{"parts": [{"text": prompt}]}],
                     "generationConfig": {"temperature": 0}})
        return out["candidates"][0]["content"]["parts"][0]["text"]
    if os.environ.get("ANTHROPIC_API_KEY"):
        model = os.environ.get("LLM_MODEL", "claude-sonnet-5-5")
        out = _post("https://api.anthropic.com/v1/messages",
                    {"content-type": "application/json",
                     "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                     "anthropic-version": "2023-06-01"},
                    {"model": model, "max_tokens": 800, "temperature": 0,
                     "messages": [{"role": "user", "content": prompt}]})
        return out["content"][0]["text"]
    raise SystemExit("Set GEMINI_API_KEY or ANTHROPIC_API_KEY first.")
