"""
src/stage3_brain/prompt_builder.py
──────────────────────────────────
PromptBuilder: Assembles persona, customer profile memory, retrieved vehicle
inventory knowledge with citations, and dialogue history into a grounded
messages array for Llama 3.2.
"""

from typing import List, Dict, Any, Optional
from stage2_memory.models import ChatMessage, UserMemoryItem
from stage2_memory.catalog_service import catalog_service

OUT_OF_CONTEXT_DEFLECTION = (
    "I am the Customer Support Assistant for Apex Auto Recondition House Nepal (Tinkune, Kathmandu). "
    "I can only assist with inquiries regarding our certified car inventory (Santro, i10, Magnite, Swift, Alto, Brezza, Creta), "
    "prices in Lakhs, 20-min spot exchange (साटासाट), Bluebook ownership transfer (नामसारी), and bank EMI financing."
)

CAR_DEALERSHIP_PERSONA_PROMPT = """You are the AI Assistant for Apex Auto Recondition House Nepal (Tinkune, Kathmandu).
Answer strictly using the verified vehicle inventory catalog and showroom policies provided below.
Always quote vehicle prices in Nepali Rupees / Lakhs (e.g. NPR 15.5 Lakhs or NPR 26.5 Lakhs).
Keep responses friendly, professional, concise, and within 2 to 3 sentences."""

FEW_SHOT_TURNS = [
    {
        "role": "user",
        "content": "Do you have Hyundai Santro?"
    },
    {
        "role": "assistant",
        "content": "Namaste! Yes, we have a 2020 Hyundai Santro Magna (1100cc, Lot Ba 02-028 Cha, 38,000 km, 2081/82 tax paid) for NPR 15.5 Lakhs. It includes our 6-month engine warranty and free test drives at Tinkune!"
    },
    {
        "role": "user",
        "content": "Which cars fall under my budget of 16 Lakhs?"
    },
    {
        "role": "assistant",
        "content": "For a budget of NPR 16 Lakhs, our certified qualifying cars are the 2020 Hyundai Santro Magna (NPR 15.5 Lakhs), 2022 Maruti Suzuki Alto K10 VXi Plus (NPR 14.2 Lakhs), and 2019 Maruti Suzuki Alto 800 LXi (NPR 10.5 Lakhs)."
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

        # 1. Always Inject Complete Certified Inventory Catalog (~480 tokens)
        catalog_table = catalog_service.get_catalog_prompt_text()
        system_content_parts.append(
            f"\n[APEX AUTO RECONDITION NEPAL — CERTIFIED VEHICLE STOCK CATALOG (25 Vehicles)]\n"
            f"{catalog_table}\n"
            "ANALYTICAL REASONING RULES:\n"
            "- Mathematical Budget Rule: When evaluating customer budget, strictly compare car prices against it. ONLY recommend cars where Price <= Budget. Never claim a car from above budget is within budget.\n"
            "- Tricky/Out-of-Budget Questions: If asked about a car that exceeds budget, state its exact price and how much it exceeds budget, then suggest the best qualifying option.\n"
            "- Multi-Criteria Searches: You can answer any query (cheapest automatic, high ground clearance, family SUV) by consulting the catalog table."
        )

        # 2. Inject Retrieved Policy / Warranty Knowledge Chunks (if applicable)
        if knowledge_context.strip():
            system_content_parts.append(
                f"\n[DETAILED SHOWROOM POLICIES & GUIDELINES]\n{knowledge_context.strip()}\n"
            )

        # 3. Inject Persistent Customer Memory Profile
        customer_budget = None
        if user_memories:
            mem_lines = ["\n[CUSTOMER INTELLIGENCE & SAVED PROFILE (SQLite)]"]
            for m in user_memories:
                mem_lines.append(f"- ({m.category.upper()}): {m.memory_text}")
                if m.category.lower() == "budget" or "budget" in m.memory_text.lower():
                    customer_budget = m.memory_text
            system_content_parts.append("\n".join(mem_lines))

        # Check if current query mentions or overrides budget
        import re
        q_budget = re.search(r"(?:budget|under|below|within|upto|around|have)\s*(?:is|of|npr|rs\.?)?\s*(\d+(?:\.\d+)?)\s*(?:lakh|lakhs)?", user_query, re.IGNORECASE)
        if q_budget:
            customer_budget = q_budget.group(1)

        # 4. Inject Mathematical Budget Audit at the VERY END for maximum attention/recency
        if customer_budget:
            budget_block = catalog_service.get_budget_context_block(customer_budget)
            if budget_block:
                system_content_parts.append(
                    f"{budget_block}\n"
                    f"CRITICAL INSTRUCTION: When answering inquiries about cars within the customer's budget, ONLY recommend cars from the 'STRICTLY WITHIN BUDGET' list. Never recommend cars from the 'ABOVE BUDGET' list as being within budget."
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
