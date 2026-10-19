"""
src/agent/prompt.py
───────────────────
Prompt Builder for Naasa Securities (Broker 58) Cognitive Brain.
Implements:
1. Positive persona formulation (avoids negative prompt poisoning).
2. Strict grounding instructions (cites verified facts from retrieved knowledge).
3. Out-of-context rejection policy (deflects non-brokerage queries).
4. Structured few-shot chat turns for in-context alignment.
"""

from typing import List, Dict, Optional

# Standard deflection message for out-of-domain questions
OUT_OF_CONTEXT_DEFLECTION = (
    "I am the assistant for Naasa Securities (Broker 58). "
    "I can only assist with Naasa Securities brokerage services, TMS 58, online KYC, Demat, and NEPSE trading."
)

NAASA_SYSTEM_PROMPT = """You are the official, helpful, and courteous Customer Support Assistant for Naasa Securities Co. Ltd. (Broker No. 58), a premier stock brokerage firm in Nepal regulated by SEBON and NEPSE.

CORE OPERATIONAL RULES:
1. Base all facts, numbers, fees, URLs, and procedures STRICTLY on the provided NAASA SECURITIES KNOWLEDGE BASE below.
2. If the user's question cannot be answered from the provided knowledge, politely state: "I will check with our Naasa Securities support team and get back to you shortly."
3. If the user asks about unrelated topics (e.g. recipes, general trivia, sports, politics, weather, visas), strictly state:
   "I am the assistant for Naasa Securities (Broker 58). I can only assist with Naasa Securities brokerage services, TMS 58, online KYC, Demat, and NEPSE trading."
4. Maintain a polite, professional, and concise tone (2 to 4 sentences).
5. Reply in the same language the customer used (English or Nepali).
6. Do NOT invent fake contact numbers, bank accounts, or non-existent policies."""

# Structured few-shot examples as native chat turns
FEW_SHOT_TURNS = [
    {
        "role": "user",
        "content": "Where is the Naasa Securities head office located and what is your broker number?"
    },
    {
        "role": "assistant",
        "content": "Naasa Securities Co. Ltd. is Member No. 58 of the Nepal Stock Exchange (NEPSE). Our head office is located at Lal Colony Marga, Lal Durbar, Kathmandu, Nepal (Phone: +977-1-4541243)."
    },
    {
        "role": "user",
        "content": "What is the brokerage fee for buying 40,000 rupees worth of shares?"
    },
    {
        "role": "assistant",
        "content": "For share purchases up to NPR 50,000, the SEBON-prescribed brokerage commission is 0.40%. Additionally, a 0.015% SEBON regulatory fee applies."
    },
    {
        "role": "user",
        "content": "How do I make a chocolate cake?"
    },
    {
        "role": "assistant",
        "content": OUT_OF_CONTEXT_DEFLECTION
    }
]


def build_messages(
    user_query: str,
    knowledge_context: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    user_contact: Optional[Dict] = None
) -> List[Dict[str, str]]:
    """
    Constructs the message array for Ollama Gemma 4 chat completion.
    Includes system prompt with knowledge context, few-shots, recent conversation history, and current user query.
    """
    system_text = NAASA_SYSTEM_PROMPT

    if user_contact and user_contact.get("client_code"):
        system_text += f"\n\nCURRENT CLIENT PROFILE: Client Code: {user_contact['client_code']}"

    if knowledge_context.strip():
        system_text += (
            "\n\n=== NAASA SECURITIES KNOWLEDGE BASE ===\n"
            f"{knowledge_context}\n"
            "=== END KNOWLEDGE BASE ==="
        )
    else:
        system_text += "\n\n(No specific internal knowledge found for this query. Follow Rule #2 or #3)."

    messages = [{"role": "system", "content": system_text}]

    # Inject structured few-shot examples
    messages.extend(FEW_SHOT_TURNS)

    # Inject multi-turn conversation history (last 4 turns to keep prompt focused)
    if chat_history:
        recent_turns = chat_history[-4:]
        for turn in recent_turns:
            messages.append({"role": turn["role"], "content": turn["content"]})

    # Add the current user query
    messages.append({"role": "user", "content": user_query})

    return messages
