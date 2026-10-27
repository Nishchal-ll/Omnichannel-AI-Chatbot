"""
src/agent/prompt.py
───────────────────
Prompt Builder for Lionel Messi AI Knowledge Assistant.
Implements:
1. Strict grounding instructions (cites verified facts from retrieved knowledge).
2. Direct-response enforcement (suppresses Gemma 4 internal thought generation).
3. Out-of-context rejection policy (refuses non-Messi queries).
4. Structured few-shot chat turns for in-context alignment.
"""

from typing import List, Dict, Optional

# Standard deflection message for out-of-domain questions
OUT_OF_CONTEXT_DEFLECTION = (
    "I am the AI assistant specializing in Lionel Messi's career, records, and biography. "
    "I can only assist with questions regarding Lionel Messi's football career, clubs, trophies, and statistics."
)

MESSI_SYSTEM_PROMPT = """You are an expert, precise, and courteous AI Assistant specializing in Lionel Messi's football career, achievements, and biography.

CRITICAL INSTRUCTIONS:
- Answer the user's question ONLY using the verified knowledge context provided below.
- If the question cannot be answered from the provided knowledge, strictly state: "I don't have enough verified information about that in the knowledge base."
- Be concise, direct, and factually accurate. Mention the relevant source section in brackets (e.g., [FC Barcelona Career] or [Individual Awards and Honors]).
- Output ONLY the final response directly. Do NOT output internal reasoning, analysis steps, thinking traces, or markdown tags like <thought>.
- Keep replies within 2 to 4 clear sentences."""

# Structured few-shot examples as native chat turns
FEW_SHOT_TURNS = [
    {
        "role": "user",
        "content": "How many Ballon d'Or awards has Lionel Messi won?"
    },
    {
        "role": "assistant",
        "content": "Lionel Messi has won a record 8 Ballon d'Or awards, receiving them in 2009, 2010, 2011, 2012, 2015, 2019, 2021, and 2023. [Individual Awards and Honors]"
    },
    {
        "role": "user",
        "content": "Which club did Messi join in 2023?"
    },
    {
        "role": "assistant",
        "content": "On July 15, 2023, Lionel Messi officially joined Major League Soccer (MLS) club Inter Miami CF in the United States, wearing jersey number 10. [Inter Miami CF and Major League Soccer]"
    },
    {
        "role": "user",
        "content": "How do I bake a chocolate cake?"
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
    system_text = MESSI_SYSTEM_PROMPT

    if knowledge_context.strip():
        system_text += (
            "\n\n=== VERIFIED KNOWLEDGE CONTEXT ===\n"
            f"{knowledge_context}\n"
            "=== END KNOWLEDGE CONTEXT ==="
        )
    else:
        system_text += "\n\n(No specific knowledge found for this query. If not answering a greeting, state you don't know)."

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
