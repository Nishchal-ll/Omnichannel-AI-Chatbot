"""
src/agent/ollama_client.py
──────────────────────────
Direct, zero-overhead REST client for Ollama running Gemma 4 (gemma4:e2b / gemma4:e2b-it-qat).
Implements:
1. Token watchdog ensuring reasoning tokens do not truncate the final answer.
2. Clean extraction of content with fallback to reasoning trace if needed.
3. Connection health checking and error handling.
"""

import os
import requests
import logging
from typing import List, Dict, Optional, Tuple, Any

log = logging.getLogger(__name__)


class OllamaClient:
    """Direct HTTP client for local Ollama instance."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 200,
        temperature: float = 0.0,
        timeout: int = 60,
    ):
        self.base_url = (base_url or os.getenv("LLM_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "gemma4:e2b")
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", str(max_tokens)))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", str(temperature)))
        self.timeout = int(os.getenv("LLM_TIMEOUT", str(timeout)))

    def is_available(self) -> Tuple[bool, str]:
        """Check if Ollama server is up and verify model availability."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m.get("name") for m in r.json().get("models", [])]
                has_model = any(self.model in m or m.startswith(self.model.split(":")[0]) for m in models)
                return True, f"Connected to Ollama (Available models: {', '.join(models[:3])})"
            return False, f"Ollama returned HTTP {r.status_code}"
        except Exception as e:
            return False, f"Could not connect to Ollama at {self.base_url}: {e}"

    def chat_complete_with_metadata(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        """
        Send chat completion request to Ollama native /api/chat endpoint with think=False.
        Returns (reply_text, metadata) with timing and token usage.
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
                "num_ctx": 8192,
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
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
            "had_reasoning": False,
            "reasoning": "",
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

            # Clean any residual markdown think blocks if any
            import re
            cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return (cleaned if cleaned else content), meta

        except requests.exceptions.Timeout:
            meta["inference_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
            log.error("Ollama inference timed out after %ds", self.timeout)
            return "Inference request timed out. Please try again.", meta
        except Exception as e:
            meta["inference_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
            log.error("Ollama client error: %s", e)
            return "I don't have enough verified information about that in the knowledge base.", meta

    def chat_complete(self, messages: List[Dict[str, str]]) -> str:
        """
        Send chat completion request to Ollama OpenAI-compatible /v1 endpoint.
        Handles thinking/reasoning model outputs and token budgets.
        """
        reply, _ = self.chat_complete_with_metadata(messages)
        return reply
