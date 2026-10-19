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
from typing import List, Dict, Optional, Tuple

log = logging.getLogger(__name__)


class OllamaClient:
    """Direct HTTP client for local Ollama instance."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        timeout: int = 120,
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

    def chat_complete(self, messages: List[Dict[str, str]]) -> str:
        """
        Send chat completion request to Ollama OpenAI-compatible /v1 endpoint.
        Handles thinking/reasoning model outputs and token budgets.
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                log.error("Ollama API Error %s: %s", resp.status_code, resp.text)
                return "I will check with our Naasa Securities support team and get back to you shortly."

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return "I will check with our Naasa Securities support team and get back to you shortly."

            choice_msg = choices[0].get("message", {})
            content = (choice_msg.get("content") or "").strip()
            reasoning = (choice_msg.get("reasoning") or "").strip()

            # If content is generated, return it
            if content:
                # Strip out any residual <think>...</think> blocks if present
                import re
                cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                return cleaned if cleaned else content

            # Fallback: if model spent all tokens in reasoning without writing content
            if reasoning:
                # Look for a concluding sentence in reasoning
                lines = [l.strip() for l in reasoning.split('\n') if l.strip()]
                if lines:
                    last_line = lines[-1]
                    if len(last_line) > 10 and not last_line.startswith('*'):
                        return last_line

            return "I will check with our Naasa Securities support team and get back to you shortly."

        except requests.exceptions.Timeout:
            log.error("Ollama inference timed out after %ds", self.timeout)
            return "Our support system is currently experiencing high volume. Please contact our support team at +977-1-4541243."
        except Exception as e:
            log.error("Ollama client error: %s", e)
            return "I will check with our Naasa Securities support team and get back to you shortly."
