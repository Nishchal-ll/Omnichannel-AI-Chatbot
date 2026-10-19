"""
src/agent/router.py
───────────────────
Intent routing, fast-path anchors, query contextualization, and out-of-context guard.
Directly implements recommendations from pipeline_analysis_and_architecture_documentation.md:
- 0ms canonical anchors for greetings/signoffs to eliminate LLM latency.
- Multi-turn pronoun resolution for follow-up questions.
- Strict detection of out-of-domain queries to prevent ungrounded hallucinations.
"""

import re
from typing import List, Dict, Optional, Tuple

# Fast Greeting Patterns
GREETING_REGEX = re.compile(
    r"^(hi|hello|hey|namaste|namaskar|good\s+morning|good\s+afternoon|good\s+evening|नमस्ते|नमस्कार)\b[!\.\s]*$",
    re.IGNORECASE
)

# Fast Gratitude / Signoff Patterns
SIGNOFF_REGEX = re.compile(
    r"^(thank\s*you|thanks|thank\s*u|dhanyabad|dhanybhad|bye|goodbye|dhanyabaad|धन्यवाद)\b[!\.\s]*$",
    re.IGNORECASE
)

# Common pronouns and reference words needing contextual resolution
PRONOUN_TRIGGERS = [
    "it", "that", "this", "them", "those", "for that", "for this", "there",
    "त्यो", "त्यहाँ", "यसको", "त्यसको", "यसमा", "त्यसमा", "what about", "how much"
]

# Obvious Out-Of-Domain Triggers (General Trivia, Cooking, Politics, Code, Entertainment)
OUT_OF_DOMAIN_PATTERNS = [
    r"\b(bake|cake|recipe|cook|cooking|pizza|pasta|bread)\b",
    r"\b(capital\s+of|weather\s+in|who\s+won|world\s+cup|football|cricket\s+score)\b",
    r"\b(write\s+a\s+poem|tell\s+me\s+a\s+joke|sing\s+a\s+song|alien|dinosaur)\b",
    r"\b(visa\s+for\s+japan|ielts|toefl|study\s+in\s+australia)\b",
]
OUT_OF_DOMAIN_REGEX = re.compile("|".join(OUT_OF_DOMAIN_PATTERNS), re.IGNORECASE)

# Stock / Brokerage Keywords that confirm domain relevance
DOMAIN_KEYWORDS = [
    "naasa", "nasa", "broker", "58", "nepse", "tms", "kyc", "demat", "meroshare",
    "boid", "share", "stock", "trade", "trading", "order", "buy", "sell", "collateral",
    "settlement", "edis", "wacc", "commission", "charge", "fee", "tax", "cgt", "close-out",
    "sebon", "cdsc", "payout", "refund", "connectips", "kathmandu", "durbar", "branch",
    "शेयर", "ब्रोकर", "नेप्से", "खाता", "धितो", "कमिसन", "लाइसेन्स"
]


class IntentRouter:
    """Routes customer queries, executes deterministic fast-paths, and contextualizes follow-ups."""

    @staticmethod
    def check_fast_path(query: str) -> Optional[str]:
        """Check for 0ms deterministic responses (greetings, signoffs)."""
        clean = query.strip()
        
        if GREETING_REGEX.match(clean):
            return (
                "Namaste! Welcome to Naasa Securities (Broker No. 58). "
                "How can I assist you with your TMS account, KYC, Demat, trading, or settlement today?"
            )
            
        if SIGNOFF_REGEX.match(clean):
            return (
                "You are most welcome! If you need further assistance with your Naasa Securities "
                "Broker 58 account or trading, please feel free to ask. Happy investing!"
            )
            
        return None

    @staticmethod
    def is_out_of_domain(query: str) -> bool:
        """
        Detects if query is completely unrelated to stock brokerage / Naasa Securities.
        Returns True if query matches obvious non-domain topics and contains no domain keywords.
        """
        clean_lower = query.lower()
        
        # If it explicitly mentions broker/share/Naasa, it's NOT out of domain
        if any(kw in clean_lower for kw in DOMAIN_KEYWORDS):
            return False

        # If it matches obvious out-of-domain patterns, flag it
        if OUT_OF_DOMAIN_REGEX.search(clean_lower):
            return True

        return False

    @staticmethod
    def rewrite_query(current_query: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Resolves pronouns ('it', 'that', 'for that') against prior conversation turns.
        Produces a standalone query suitable for ChromaDB vector retrieval.
        """
        if not chat_history:
            return current_query

        clean_lower = current_query.lower()
        needs_context = any(p in clean_lower for p in PRONOUN_TRIGGERS) or len(current_query.split()) <= 4

        if not needs_context:
            return current_query

        # Retrieve the most recent user and assistant turn
        last_user = ""
        last_assistant = ""
        for turn in reversed(chat_history):
            if turn["role"] == "user" and not last_user:
                last_user = turn["content"]
            elif turn["role"] == "assistant" and not last_assistant:
                last_assistant = turn["content"]
            if last_user and last_assistant:
                break

        if not last_user:
            return current_query

        # Form a combined contextualized query string
        contextualized = f"{last_user} -> {current_query}"
        return contextualized

    @staticmethod
    def extract_contact_info(text: str) -> Dict[str, str]:
        """Extract phone number, email, or client code from user message if present."""
        extracted = {}
        
        # Email extraction
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if email_match:
            extracted["email"] = email_match.group(0)

        # Phone extraction (Nepali mobile numbers usually 98XXXXXXXX or 97XXXXXXXX)
        phone_match = re.search(r'\b(98\d{8}|97\d{8}|\+977-?\d{10})\b', text)
        if phone_match:
            extracted["phone"] = phone_match.group(0)

        # Client code extraction (e.g. "client code is 581234" or "code: ABC123")
        code_match = re.search(r'\b(?:client\s*code|code)\s*(?:is|:)?\s*([A-Za-z0-9]{4,10})\b', text, re.IGNORECASE)
        if code_match:
            extracted["client_code"] = code_match.group(1).upper()

        return extracted
