"""
CrewAI native LLM wrapper for local Ollama.
CrewAI uses LiteLLM under the hood, which expects LLM provider prefixes.
Ollama exposes an OpenAI-compatible /v1/chat/completions endpoint on port 11434.
"""
import os
from crewai import LLM


def get_llm() -> LLM:
    """
    Return a CrewAI LLM instance pointed at local Ollama.
    Ollama accepts OpenAI-compatible API format under /v1.
    """
    llm_url = os.getenv("LLM_URL", "http://localhost:11434")
    model_name = os.getenv("LLM_MODEL", "gemma4:e2b")
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    timeout = int(os.getenv("LLM_TIMEOUT", "120"))

    # Prefix with 'openai/' provider so LiteLLM targets the OpenAI-compatible /v1 endpoint
    if not model_name.startswith("openai/"):
        model_name = f"openai/{model_name}"

    return LLM(
        model=model_name,
        base_url=f"{llm_url}/v1",
        api_key="ollama",               # Ollama accepts any dummy key
        temperature=0.1,
        max_tokens=max_tokens,
        timeout=timeout,
    )
