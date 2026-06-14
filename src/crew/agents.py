"""
CrewAI agents.

Two agents work together on every incoming message:
  1. ClassifierAgent  — reads the message, identifies intent and tone
  2. ResponderAgent   — writes the actual reply based on classifier output

Both use the local llama.cpp LLM via LangChain-compatible wrapper.
"""
import os
from crewai import Agent
from .llm import get_llm


def make_classifier_agent() -> Agent:
    """
    Reads the incoming message and produces a structured brief:
    - Intent (question / complaint / order / greeting / other)
    - Tone   (friendly / frustrated / urgent / neutral)
    - Key topics mentioned
    - Channel context passed through
    """
    return Agent(
        role="Message Classifier",
        goal=(
            "Analyse the incoming customer message and produce a short classification brief "
            "with: intent, tone, key topics. Be concise — 3-5 lines max."
        ),
        backstory=(
            "You are an expert at understanding customer messages across different channels. "
            "You identify what the customer needs and the emotional tone so the responder "
            "can craft the perfect reply."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=2,
    )


def make_responder_agent() -> Agent:
    """
    Takes the classifier brief and the original message,
    writes a concise, helpful reply appropriate for the channel.
    """
    system_prompt = os.getenv(
        "BOT_SYSTEM_PROMPT",
        "You are a helpful customer support assistant. Keep replies concise and friendly.",
    )
    return Agent(
        role="Customer Support Responder",
        goal=(
            "Write a helpful, concise reply to the customer message. "
            "Reply in the same language the customer used. "
            "Output ONLY the reply text — no preamble, no labels, no quotes."
        ),
        backstory=(
            system_prompt
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=2,
    )
