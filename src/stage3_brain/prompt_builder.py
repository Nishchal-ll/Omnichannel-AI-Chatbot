"""
src/stage3_brain/prompt_builder.py
──────────────────────────────────
PromptBuilder: Assembles persona, customer profile memory, retrieved vehicle
inventory knowledge with citations, and dialogue history into a grounded
messages array for Llama 3.2.
"""

from typing import List, Dict, Any, Optional
from stage2_memory.models import ChatMessage, UserMemoryItem

OUT_OF_CONTEXT_DEFLECTION = (
    "I am the Customer Support Assistant for Apex Auto Recondition House Nepal (Tinkune, Kathmandu). "
    "I can only assist with inquiries regarding our certified car inventory (Santro, i10, Magnite, Swift, Alto, Brezza, Creta), "
    "prices in Lakhs, 20-min spot exchange (साटासाट), Bluebook ownership transfer (नामसारी), and bank EMI financing."
)

CAR_DEALERSHIP_PERSONA_PROMPT = """You are the AI Assistant for Apex Auto Recondition House Nepal (Tinkune, Kathmandu).
Answer strictly from the CONTEXT below. Quote vehicle prices in Nepali Rupees / Lakhs (e.g. NPR 15.5 Lakhs).
Mention Model, Year, Engine CC, Lot Number, Mileage, and Price in Lakhs. Cite [Source 1], [Source 2] where relevant.
Keep responses concise, polite, and within 2 to 3 sentences."""

FEW_SHOT_TURNS = [
    {
        "role": "user",
        "content": "Do you have Hyundai Santro?"
    },
    {
        "role": "assistant",
        "content": "Namaste! Yes [Source 1], we have a 2020 Hyundai Santro Magna (1100cc, Lot 28, 38,000 km, 2081/82 tax paid) for NPR 15.5 Lakhs. It includes our 6-month engine warranty and free test drives at Tinkune!"
    }
]


class PromptBuilder:
    """Constructs Llama 3.2 chat message arrays with grounded context, citations, and user memory."""

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate (1 word ~ 1.35 tokens)."""
        return int(len(text.split()) * 1.35)

    @staticmethod
    def build_messages(
        user_query: str,
        knowledge_context: str,
        chat_history: List[ChatMessage],
        user_memories: List[UserMemoryItem],
        max_prompt_tokens: int = 2400,
    ) -> List[Dict[str, str]]:
        """
        Builds the messages array for Ollama /api/chat.
        Includes System Prompt + Persistent Customer Memory + Few-Shots + Dialogue History + Grounded Context.
        """
        system_content_parts = [CAR_DEALERSHIP_PERSONA_PROMPT]

        # Inject Persistent Customer Memory MESH
        if user_memories:
            mem_lines = ["\n[CUSTOMER INTELLIGENCE & SAVED PREFERENCES (SQLite)]"]
            for m in user_memories:
                mem_lines.append(f"- ({m.category.upper()}): {m.memory_text}")
            mem_lines.append("Use these customer preferences naturally (e.g. remember their budget, preferred model, or exchange vehicle).")
            system_content_parts.append("\n".join(mem_lines))

        # Inject Grounded Showroom Knowledge & Citations
        if knowledge_context.strip():
            system_content_parts.append(
                f"\n[VERIFIED SHOWROOM INVENTORY & POLICY CONTEXT]\n{knowledge_context.strip()}\n"
                "INSTRUCTION: Base your factual answers strictly on this context. Cite [Source 1], [Source 2] where relevant. If the context does not contain the answer, say so politely."
            )

        system_msg = {"role": "system", "content": "\n\n".join(system_content_parts)}

        messages: List[Dict[str, str]] = [system_msg]

        # Append few-shot demonstrations
        messages.extend(FEW_SHOT_TURNS)

        # Budget calculation for dialogue history
        budget = max_prompt_tokens - PromptBuilder._estimate_tokens(system_msg["content"])
        recent_history: List[Dict[str, str]] = []

        for turn in reversed(chat_history):
            turn_tokens = PromptBuilder._estimate_tokens(turn.content)
            if budget - turn_tokens < 300:
                break
            recent_history.insert(0, {"role": turn.role, "content": turn.content})
            budget -= turn_tokens

        messages.extend(recent_history)
        messages.append({"role": "user", "content": user_query})

        return messages


prompt_builder = PromptBuilder()
