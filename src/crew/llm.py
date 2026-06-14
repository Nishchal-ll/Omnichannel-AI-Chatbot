"""
CrewAI native LLM wrapper for local llama.cpp.
CrewAI uses LiteLLM under the hood, which expects LLM provider prefixes.
llama.cpp server exposes an OpenAI-compatible /v1/chat/completions endpoint.
"""
import os
from crewai import LLM


def get_llm() -> LLM:
    """
    Return a CrewAI LLM instance pointed at local llama.cpp.
    llama.cpp's OpenAI-compatible server accepts the same API format.
    """
    llm_url = os.getenv("LLM_URL", "http://llama:8080")
    model_name = os.getenv("LLM_MODEL", "gemma-4-E2B_q4_0-it.gguf")

    # Prefix with 'openai/' provider so LiteLLM doesn't complain
    if not model_name.startswith("openai/"):
        model_name = f"openai/{model_name}"

    return LLM(
        model=model_name,
        base_url=f"{llm_url}/v1",
        api_key="not-needed",           # llama.cpp doesn't need a real key
        temperature=0.1,
        max_tokens=512,
        timeout=120,
    )
