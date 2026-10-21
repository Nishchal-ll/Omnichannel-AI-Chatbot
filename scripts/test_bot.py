#!/usr/bin/env python3
"""
Test script to verify Naasa Securities (Broker 58) question answering with Ollama Gemma 4.
Usage:
    python scripts/test_bot.py "Where is your head office located?"
    python scripts/test_bot.py --interactive
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from channels.base import Message
from agent.pipeline import Pipeline


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
        return

    question = sys.argv[1] if len(sys.argv) > 1 else "Where is the Naasa Securities head office located and what is your broker number?"
    print(f"\n[Naasa Broker 58 Test] Query: \"{question}\"")
    print(f"[Naasa Broker 58 Test] LLM_URL: {os.getenv('LLM_URL', 'http://localhost:11434')}")
    print(f"[Naasa Broker 58 Test] LLM_MODEL: {os.getenv('LLM_MODEL', 'gemma4:e2b')}\n")

    pipeline = Pipeline()
    msg = Message(
        channel="local_test",
        sender="cli_test_user",
        body=question,
    )

    reply = pipeline.process(msg)
    print("\n" + "=" * 60)
    print("NAASA BROKER 58 AGENT RESPONSE:")
    print(reply)
    print("=" * 60 + "\n")


def interactive_mode():
    print("\n=== Naasa Securities (Broker 58) Interactive Agent CLI ===")
    print("Type your message below (or 'exit' / 'quit' to end):\n")
    pipeline = Pipeline()
    sender = "cli_interactive_user"

    while True:
        try:
            user_input = input("Customer: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting interactive test. Namaste!")
                break

            msg = Message(channel="local_test", sender=sender, body=user_input)
            reply = pipeline.process(msg)
            print(f"\nNaasa Assistant: {reply}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Namaste!")
            break


if __name__ == "__main__":
    main()
