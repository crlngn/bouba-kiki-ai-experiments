"""Shared interface over the model providers used in phase 1.

Every adapter exposes generate(prompt, images, temperature, max_tokens, options) -> str, returning
one sampled completion. `options` is only used by the mock model, which answers at random and
exists to test the pipeline without API keys or local weights.

Model specs on the command line look like provider:model_id, for example
anthropic:claude-sonnet-5-5, gemini:<model id>, ollama:gemma3:27b or mock:random.
API keys come from the environment or a .env file in the repo root.
"""

import base64
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SYSTEM_PROMPT = "Answer the question. Give only the answer, with no explanation."


def _b64(path):
    """Base64-encode an image file."""
    return base64.b64encode(Path(path).read_bytes()).decode()


def _retry(fn, attempts=5):
    """Call fn, retrying with exponential backoff on any exception."""
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(2 ** i)


class AnthropicModel:
    """Claude via the Anthropic Messages API."""

    def __init__(self, model_id):
        import anthropic
        self.model_id = model_id
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"),
                                          base_url="https://api.anthropic.com")

    def generate(self, prompt, images=(), temperature=1.0, max_tokens=32, options=None):
        """Sample one completion, with images placed before the text."""
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(p)}}
                   for p in images]
        content.append({"type": "text", "text": prompt})
        resp = _retry(lambda: self.client.messages.create(
            model=self.model_id, max_tokens=max_tokens, temperature=temperature, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}]))
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class GeminiModel:
    """Gemini via the google-genai SDK. Thinking models get a larger token budget so the answer isn't cut off."""

    def __init__(self, model_id):
        from google import genai
        self.model_id = model_id
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    def generate(self, prompt, images=(), temperature=1.0, max_tokens=32, options=None):
        """Sample one completion, with images placed before the text."""
        from google.genai import types
        parts = [types.Part.from_bytes(data=Path(p).read_bytes(), mime_type="image/png") for p in images]
        parts.append(prompt)
        config = types.GenerateContentConfig(temperature=temperature, system_instruction=SYSTEM_PROMPT,
                                             max_output_tokens=max(max_tokens, 2048))
        resp = _retry(lambda: self.client.models.generate_content(model=self.model_id, contents=parts, config=config))
        return resp.text or ""


class OllamaModel:
    """A local open-weights model served by Ollama."""

    def __init__(self, model_id):
        import requests
        self.model_id = model_id
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.session = requests.Session()

    def generate(self, prompt, images=(), temperature=1.0, max_tokens=32, options=None):
        """Sample one completion through /api/chat."""
        msg = {"role": "user", "content": prompt}
        if images:
            msg["images"] = [_b64(p) for p in images]
        body = {"model": self.model_id, "stream": False,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, msg],
                "options": {"temperature": temperature, "num_predict": max_tokens}}
        resp = _retry(lambda: self.session.post(f"{self.host}/api/chat", json=body, timeout=300))
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class MockModel:
    """Answers at random from the trial's options. For pipeline tests only."""

    def __init__(self, model_id):
        self.model_id = model_id
        self.rng = random.Random(0)

    def generate(self, prompt, images=(), temperature=1.0, max_tokens=32, options=None):
        """Return a random option."""
        return str(self.rng.choice(list(options or ["A", "B"])))


PROVIDERS = {"anthropic": AnthropicModel, "gemini": GeminiModel, "ollama": OllamaModel, "mock": MockModel}
VISION = {"anthropic", "gemini", "ollama", "mock"}


def load_model(spec):
    """Build a model from a provider:model_id spec."""
    provider, _, model_id = spec.partition(":")
    if provider not in PROVIDERS or not model_id:
        raise SystemExit(f"Unknown model spec {spec!r}; use one of {sorted(PROVIDERS)} as provider:model_id")
    model = PROVIDERS[provider](model_id)
    model.provider = provider
    model.spec = spec
    return model
