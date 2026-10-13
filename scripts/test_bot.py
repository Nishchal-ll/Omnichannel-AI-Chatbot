#!/usr/bin/env python3
"""
Test script to verify end-to-end question answering with Ollama and Gemma 4 model.
Usage:
    python scripts/test_bot.py "Do you have coconut oil?"
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from channels.base import Message
from crew.pipeline import Pipeline

def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "Hello! Can you tell me what coconut oil is good for?"
    print(f"\n[Test] Sending question to Pipeline: \"{question}\"")
    print(f"[Test] Configured LLM_URL: {os.getenv('LLM_URL', 'http://localhost:11434')}")
    print(f"[Test] Configured LLM_MODEL: {os.getenv('LLM_MODEL', 'gemma4:e2b')}\n")

    pipeline = Pipeline()
    msg = Message(
        channel="whatsapp",
        sender="test_user_123",
        body=question,
    )

    reply = pipeline.process(msg)
    print("\n" + "=" * 50)
    print("BOT ANSWER:")
    print(reply)
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
