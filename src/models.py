"""Shared interface over the model providers used in phase 1.

Every adapter exposes generate(prompt, images, temperature, max_tokens, options, audio) -> str, returning
one sampled completion. `options` is only used by the mock model, which answers at random and
exists to test the pipeline without API keys or local weights. `audio` is a list of recordings
played before the text; only providers in AUDIO accept it, the others raise rather than drop it.

Model specs on the command line look like provider:model_id, for example
anthropic:claude-sonnet-5-5, gemini:<model id>, ollama:gemma3:27b, hf:google/gemma-4-E4B-it or mock:random.
The hf provider runs a Hugging Face model locally through the transformers "any-to-any" pipeline; it is
the route for Gemma 4 audio, since Ollama's audio support for Gemma 4 isn't documented. Install it with
`uv sync --extra hf`.
API keys come from the environment or a .env file in the repo root.
"""

import base64
import os
import random
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SYSTEM_PROMPT = "Answer the question. Give only the answer, with no explanation."


def _b64(path):
    """Base64-encode an image file."""
    return base64.b64encode(Path(path).read_bytes()).decode()


def _no_audio(provider, audio):
    """Refuse audio trials on providers that can't take audio, so a run never silently drops the sound."""
    if audio:
        raise ValueError(f"{provider} models don't accept audio input")


def _audio_mime(path):
    """MIME type for a recording."""
    return {".wav": "audio/wav", ".mp3": "audio/mp3", ".flac": "audio/flac"}[Path(path).suffix.lower()]


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

    def generate(self, prompt, images=(), temperature=1.0, max_tokens=32, options=None, audio=()):
        """Sample one completion, with images placed before the text."""
        _no_audio("anthropic", audio)
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

    def generate(self, prompt, images=(), temperature=1.0, max_tokens=32, options=None, audio=()):
        """Sample one completion, with recordings and images placed before the text."""
        from google.genai import types
        parts = [types.Part.from_bytes(data=Path(p).read_bytes(), mime_type=_audio_mime(p)) for p in audio]
        parts += [types.Part.from_bytes(data=Path(p).read_bytes(), mime_type="image/png") for p in images]
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

    def generate(self, prompt, images=(), temperature=1.0, max_tokens=32, options=None, audio=()):
        """Sample one completion through /api/chat."""
        _no_audio("ollama", audio)
        msg = {"role": "user", "content": prompt}
        if images:
            msg["images"] = [_b64(p) for p in images]
        body = {"model": self.model_id, "stream": False,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, msg],
                "options": {"temperature": temperature, "num_predict": max_tokens}}
        resp = _retry(lambda: self.session.post(f"{self.host}/api/chat", json=body, timeout=300))
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class HFModel:
    """A local Hugging Face model (e.g. google/gemma-4-E4B-it) through the transformers any-to-any pipeline.

    One pipeline is shared by all worker threads, so calls are serialised with a lock. Recordings and
    images go before the text, as in the other adapters; Gemma 4 takes up to 30 s of 16 kHz mono audio.
    """

    def __init__(self, model_id):
        from transformers import pipeline
        self.model_id = model_id
        self.pipe = pipeline(task="any-to-any", model=model_id, device_map="auto", dtype="auto")
        self.lock = threading.Lock()

    def generate(self, prompt, images=(), temperature=1.0, max_tokens=32, options=None, audio=()):
        """Sample one completion."""
        content = [{"type": "audio", "audio": str(ROOT / p)} for p in audio]
        content += [{"type": "image", "image": str(ROOT / p)} for p in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": content}]
        gen = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            gen["temperature"] = temperature
        with self.lock:
            out = self.pipe(messages, return_full_text=False, generate_kwargs=gen)
        return out[0]["generated_text"].replace("<turn|>", "").strip()


class MockModel:
    """Answers at random from the trial's options. For pipeline tests only."""

    def __init__(self, model_id):
        self.model_id = model_id
        self.rng = random.Random(0)

    def generate(self, prompt, images=(), temperature=1.0, max_tokens=32, options=None, audio=()):
        """Return a random option, or a made-up spelling when the trial has no options."""
        if not options:
            return self.rng.choice(["bouba", "kiki", "maluma", "takete"])
        return str(self.rng.choice(list(options)))


PROVIDERS = {"anthropic": AnthropicModel, "gemini": GeminiModel, "ollama": OllamaModel, "hf": HFModel,
             "mock": MockModel}
VISION = {"anthropic", "gemini", "ollama", "hf", "mock"}
AUDIO = {"gemini", "hf", "mock"}


def load_model(spec):
    """Build a model from a provider:model_id spec."""
    provider, _, model_id = spec.partition(":")
    if provider not in PROVIDERS or not model_id:
        raise SystemExit(f"Unknown model spec {spec!r}; use one of {sorted(PROVIDERS)} as provider:model_id")
    model = PROVIDERS[provider](model_id)
    model.provider = provider
    model.spec = spec
    return model
