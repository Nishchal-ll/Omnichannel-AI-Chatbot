"""
CrewAI tasks.
Each task is bound to an agent and describes exactly what to produce.
"""
import sys
import os
from crewai import Task
from crewai import Agent
from channels.base import Message

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledge import get_relevant_knowledge


def make_classify_task(agent: Agent, message: Message) -> Task:
    channel_context = f"Channel: {message.channel}"
    if message.subject:
        channel_context += f" | Subject: {message.subject}"

    return Task(
        description=(
            f"Classify this incoming customer message.\n\n"
            f"{channel_context}\n"
            f"From: {message.sender}\n"
            f"Message: {message.body}\n\n"
            f"Produce a brief (3-5 lines) covering:\n"
            f"- Intent: what does the customer want?\n"
            f"- Tone: friendly / frustrated / urgent / neutral\n"
            f"- Key topics mentioned"
        ),
        expected_output="A 3-5 line classification brief.",
        agent=agent,
    )


def make_respond_task(agent: Agent, message: Message, classify_task: Task) -> Task:
    # Trim message body to 80 words to save tokens
    body_words = message.body.split()
    body_short = " ".join(body_words[:80]) + ("…" if len(body_words) > 80 else "")

    # Retrieve only the relevant KB chunks — NOT the whole KB
    kb_context = get_relevant_knowledge(message.body, top_k=3)
    kb_block   = kb_context if kb_context.strip() else "No specific product info found."

    channel_context = f"Channel: {message.channel.upper()}"
    if message.subject:
        channel_context += f" | Email subject: {message.subject}"

    return Task(
        description=(
            "You are a coconut oil product support agent.\n\n"
            f"{channel_context}\n\n"
            "PRODUCT KNOWLEDGE:\n"
            f"{kb_block}\n\n"
            "CUSTOMER MESSAGE:\n"
            f"{body_short}\n\n"
            "Use the classification brief from the previous task as context.\n\n"
            "RULES:\n"
            "- Answer using ONLY the product knowledge above\n"
            "- If the answer is not in the knowledge base, say:\n"
            "  'I will check and get back to you shortly.'\n"
            "- Keep reply to 2–4 sentences\n"
            "- Be polite and professional\n"
            "- Reply in the same language the customer used\n"
            "- Output ONLY the reply text, nothing else"
        ),
        expected_output="A 2–4 sentence reply using only the provided knowledge base. The reply text only. No labels, no quotes, no preamble.",
        agent=agent,
        context=[classify_task],
    )
