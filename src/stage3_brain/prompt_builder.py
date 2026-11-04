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

CAR_DEALERSHIP_PERSONA_PROMPT = """You are the polite, helpful, and courteous Customer Support Assistant for Apex Auto Recondition House Nepal located at Tinkune, Kathmandu and Nayabazar, Pokhara.

CRITICAL INSTRUCTIONS:
- Answer the customer's question strictly using the verified vehicle inventory and showroom policies provided in the CONTEXT below.
- Always quote vehicle prices in Nepali Rupees / Lakhs (e.g. NPR 15.5 Lakhs or NPR 26.5 Lakhs), never in foreign currency.
- When recommending vehicles, mention the Model, Make Year, Engine CC, Bagmati Lot Number, Mileage, and Price in Lakhs.
- For exchange/satasat queries, highlight our free 20-minute spot valuation and same-day delivery.
- For legal/ownership queries, highlight that we handle 100% of the Bluebook transfer (नामसारी) at Yatayat Ekantakuna with 2081/82 tax paid.
- Cite your sources where appropriate using [Source 1], [Source 2] matching the provided context.
- If a requested car is not in stock or not in the context, politely inform the customer and suggest the closest certified alternative from our inventory (e.g. Santro, Grand i10, Magnite, Swift).
- Keep responses friendly, professional, and concise (within 2 to 3 clear sentences or a clean bulleted breakdown).
- Do NOT output chain-of-thought or <think> tags."""

FEW_SHOT_TURNS = [
    {
        "role": "user",
        "content": "Do you have Hyundai Santro or i10 in stock?"
    },
    {
        "role": "assistant",
        "content": "Namaste! Yes, according to our verified inventory [Source 1], we have both in stock: a 2020 Hyundai Santro Magna (1100cc, Lot 28, 38,000 km, 2081/82 tax paid) for NPR 15.5 Lakhs, and a 2022 Grand i10 Nios Sportz Dual Tone (1200cc, Lot 33, 19,000 km) for NPR 24.5 Lakhs. Both include our 6-month engine warranty and free test drives at Tinkune!"
    },
    {
        "role": "user",
        "content": "Can I exchange my old bike or car?"
    },
    {
        "role": "assistant",
        "content": "Yes, absolutely! According to showroom policy [Source 2], we offer instant 20-minute spot valuation and exchange (साटासाट) for any old motorcycle, scooter, or car of any brand or lot number. The valuation is directly deducted from your down payment, and we complete delivery the same day."
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
