"""
src/stage3_brain/brain_engine.py
────────────────────────────────
BrainEngine: Ollama native client for Gemma 4 E2B (gemma4:e2b).
Executes streaming/direct generation with think=False and num_ctx=8192.
"""

import os
import requests
import logging
from typing import List, Dict, Tuple, Any, Optional

log = logging.getLogger(__name__)


class BrainEngine:
    """Direct HTTP client for Ollama Gemma 4 E2B."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 250,
        temperature: float = 0.15,
        timeout: int = 60,
    ):
        self.base_url = (base_url or os.getenv("LLM_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "llama3.2")
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", str(max_tokens)))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", str(temperature)))
        self.timeout = int(os.getenv("LLM_TIMEOUT", str(timeout)))

    def is_available(self) -> Tuple[bool, str]:
        """Checks if Ollama server is up and model is present, with automatic fallback."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m.get("name") for m in r.json().get("models", [])]
                has_model = any(self.model in m or m.startswith(self.model.split(":")[0]) for m in models)
                if has_model:
                    return True, f"Connected to Ollama (Model: {self.model})"
                # Fallback check
                for fallback in ["llama3.2", "gemma4:e2b", "gemma4:e2b-it-qat"]:
                    if any(fallback in m for m in models):
                        self.model = fallback
                        return True, f"Connected to Ollama (Fell back to: {self.model})"
                return False, f"Model {self.model} not found in Ollama: {models}"
            return False, f"Ollama returned HTTP {r.status_code}"
        except Exception as e:
            return False, f"Could not connect to Ollama at {self.base_url}: {e}"

    def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        """
        Sends chat completion request to Ollama native /api/chat with think=False.
        Returns (reply_text, metadata).
        """
        import time
        t_start = time.perf_counter()
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "num_ctx": 4096,
                "num_thread": 8,
                "temperature": self.temperature,
                "num_predict": min(self.max_tokens, 120),
                "top_p": 0.9,
            },
        }

        meta: Dict[str, Any] = {
            "inference_ms": 0.0,
            "model": self.model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "tokens_per_sec": 0.0,
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            meta["inference_ms"] = round((time.perf_counter() - t_start) * 1000, 1)

            if resp.status_code != 200:
                log.error("Ollama API Error %s: %s", resp.status_code, resp.text)
                return "I don't have enough verified information about that in the knowledge base.", meta

            data = resp.json()
            meta["prompt_tokens"] = data.get("prompt_eval_count", 0)
            meta["completion_tokens"] = data.get("eval_count", 0)
            meta["total_tokens"] = meta["prompt_tokens"] + meta["completion_tokens"]

            if meta["inference_ms"] > 0 and meta["completion_tokens"] > 0:
                meta["tokens_per_sec"] = round(meta["completion_tokens"] / (meta["inference_ms"] / 1000.0), 2)

            msg_data = data.get("message", {})
            content = (msg_data.get("content") or "").strip()

            import re
            cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            text = cleaned if cleaned else content

            # Handle truncated answers (Item 3):
            done_reason = data.get("done_reason")
            if done_reason == "length" or (meta["completion_tokens"] >= min(self.max_tokens, 120) - 2 and not text.endswith((".", "!", "?"))):
                if "." in text:
                    text = text.rsplit(".", 1)[0] + "."

            return text, meta

        except requests.exceptions.Timeout:
            meta["inference_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
            log.error("Ollama inference timed out after %ds", self.timeout)
            return "Inference request timed out. Please try again.", meta
        except Exception as e:
            meta["inference_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
            log.error("Ollama client error: %s", e)
            return "I don't have enough verified information about that in the knowledge base.", meta


brain_engine = BrainEngine()

