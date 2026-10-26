"""
src/stage2_memory/router.py
───────────────────────────
Intent routing, fast-path anchors, memory command dispatch, and query rewriting.
Directly implements Stage 2 routing contracts.
"""

import re
from typing import List, Dict, Optional, Tuple
from .models import ChatMessage


def normalize_query(query: str) -> str:
    """Collapses elongated/repeated characters (e.g. 'hiii' -> 'hi', 'heyyy' -> 'hey')."""
    clean = query.strip()
    return re.sub(r"(.)\1{2,}", r"\1", clean)


# Fast Greeting Patterns
GREETING_REGEX = re.compile(
    r"^(hi|hii|hey|heya|hello|howdy|namaste|namaskar|good\s+morning|good\s+afternoon|good\s+evening|gm)\b([!\.\s]*(messi|leo|bot|ai|there|everyone))?[!\.\s]*$",
    re.IGNORECASE
)

# Fast Gratitude / Signoff Patterns
SIGNOFF_REGEX = re.compile(
    r"^(thank\s*you(\s+so\s+much|\s+a\s+lot)?|thanks(\s+a\s+lot|\s+so\s+much)?|thank\s*u|thx|ty|dhanyabad|bye|goodbye|see\s*you|take\s*care)\b[!\.\s]*$",
    re.IGNORECASE
)

# Standalone pronoun pattern with strict word boundaries
PRONOUN_PATTERN = re.compile(
    r"\b(he|him|his|it|that|them|those|there|which\s+years|when\s+did\s+he|how\s+many\s+did\s+he)\b",
    re.IGNORECASE
)

# Obvious Out-Of-Domain Triggers (Cooking, Politics, Visa, Weather, General Coding)
OUT_OF_DOMAIN_PATTERNS = [
    r"\b(bake|cake|recipe|cook|cooking|pizza|pasta|bread)\b",
    r"\b(capital\s+of|weather\s+in|cricket\s+score|nba|basketball)\b",
    r"\b(write\s+a\s+poem|tell\s+me\s+a\s+joke|sing\s+a\s+song|alien|dinosaur)\b",
    r"\b(visa\s+for\s+japan|ielts|toefl|study\s+in\s+australia)\b",
]
OUT_OF_DOMAIN_REGEX = re.compile("|".join(OUT_OF_DOMAIN_PATTERNS), re.IGNORECASE)

DOMAIN_KEYWORDS = [
    "messi", "lionel", "leo", "barca", "barcelona", "psg", "paris", "miami", "inter miami",
    "argentina", "world cup", "ballon", "dor", "copa", "america", "goals", "assists",
    "trophy", "trophies", "stats", "statistics", "record", "records", "golden shoe",
    "newell", "rosario", "la masia", "antonela", "thiago", "mateo", "ciro", "mls",
    "ligue", "la liga", "champions league", "ucl", "finalissima", "leagues cup"
]

# 0ms Instant FAQ Canonical Anchors
FAQ_ANCHORS = [
    (
        re.compile(r"\b(how\s+many\s+clubs|which\s+clubs|what\s+clubs|all\s+clubs|clubs\s+(has\s+)?(messi|he|lionel)\s+played)\b", re.IGNORECASE),
        "Lionel Messi has played for 3 senior professional clubs throughout his career: FC Barcelona (2004–2021), Paris Saint-Germain (PSG, 2021–2023), and Inter Miami CF (2023–present). In his youth in Argentina, he also played for Grandoli and Newell's Old Boys. [Senior Professional Clubs Summary (Clubs Played)]"
    ),
    (
        re.compile(r"\b(current\s+club|which\s+club\s+does\s+messi\s+play\s+for|where\s+does\s+messi\s+play\s+now)\b", re.IGNORECASE),
        "Lionel Messi currently plays for Major League Soccer (MLS) club Inter Miami CF in the United States, wearing jersey number 10. [Inter Miami CF and Major League Soccer]"
    ),
    (
        re.compile(r"\b(how\s+many\s+ballon\s*d'?or|total\s+ballon\s*d'?or)\b", re.IGNORECASE),
        "Lionel Messi has won a world-record 8 Ballon d'Or awards (2009, 2010, 2011, 2012, 2015, 2019, 2021, and 2023). [Individual Awards and Honors]"
    ),
    (
        re.compile(r"\b(goals\s+in\s+2012|record\s+in\s+2012|91\s+goals)\b", re.IGNORECASE),
        "Lionel Messi holds the Guinness World Record for scoring 91 official goals in the calendar year 2012 (79 for Barcelona, 12 for Argentina). [World Records and Career Statistics]"
    ),
    (
        re.compile(r"\b(did\s+messi\s+win\s+(the\s+)?world\s+cup|world\s+cup\s+2022)\b", re.IGNORECASE),
        "Yes, Lionel Messi captained Argentina to victory in the 2022 FIFA World Cup in Qatar, scoring 7 goals and winning the tournament's Golden Ball. [Argentina National Team Career]"
    ),
    (
        re.compile(r"\b(when\s+was\s+messi\s+born|messi\s+birthday|date\s+of\s+birth)\b", re.IGNORECASE),
        "Lionel Messi was born on June 24, 1987, in Rosario, Santa Fe, Argentina. [Overview and Early Life]"
    ),
]


class IntentRouter:
    """Stage 2 Intent Router & Context contextualizer."""

    @staticmethod
    def check_fast_path(query: str) -> Optional[str]:
        """Checks for 0ms deterministic responses (greetings, signoffs, FAQs)."""
        clean = normalize_query(query)

        if GREETING_REGEX.match(clean):
            return (
                "Hello! Welcome to the Lionel Messi AI Knowledge Assistant. "
                "How can I assist you with facts, statistics, trophies, or records from Lionel Messi's career?"
            )

        if SIGNOFF_REGEX.match(clean):
            return (
                "You are most welcome! Feel free to ask anytime about Lionel Messi's career and achievements. "
                "Enjoy following the GOAT!"
            )

        for pattern, response in FAQ_ANCHORS:
            if pattern.search(clean):
                return response

        return None

    @staticmethod
    def check_memory_recall(query: str) -> bool:
        """Detects if user is asking the bot what it knows/remembers about them."""
        clean = query.lower().strip()
        patterns = [
            r"\bwhat('?s|\s+is)\s+my\s+name\b",
            r"\bwho\s+am\s+i\b",
            r"\bwhere\s+do\s+i\s+live\b",
            r"\bwhere\s+am\s+i\s+from\b",
            r"\bwhat\s+club\s+do\s+i\s+support\b",
            r"\bwhich\s+club\s+do\s+i\s+support\b",
            r"\bwhat\s+is\s+my\s+favorite\b",
            r"\bwhat\s+do\s+you\s+(know|remember)\s+about\s+me\b",
            r"\bdo\s+you\s+know\s+me\b",
            r"\bdo\s+you\s+remember\s+me\b",
            r"\bmy\s+details\b",
            r"\bmy\s+profile\b",
            r"\bshow\s+my\s+memories\b",
            r"\btell\s+me\s+about\s+myself\b",
            r"\bwhat\s+do\s+you\s+know\s+about\s+my\b",
        ]
        return any(re.search(p, clean) for p in patterns)

    @staticmethod
    def check_memory_clear(query: str) -> bool:
        """Detects if user is asking to forget/clear their memories."""
        clean = query.lower().strip()
        patterns = [
            r"\b(forget|clear|delete|erase)\s+(my\s+)?(memory|memories|information|profile|details)\b",
            r"\bforget\s+me\b",
        ]
        return any(re.search(p, clean) for p in patterns)

    @staticmethod
    def is_out_of_domain(query: str) -> bool:
        """Detects if query is completely unrelated to Lionel Messi / football."""
        clean_lower = query.lower()
        if any(kw in clean_lower for kw in DOMAIN_KEYWORDS):
            return False
        if OUT_OF_DOMAIN_REGEX.search(clean_lower):
            return True
        return False

    @staticmethod
    def rewrite_query(current_query: str, chat_history: List[ChatMessage]) -> str:
        """Resolves pronouns ('he', 'his', 'it') and context against prior turns."""
        if not chat_history:
            return current_query

        clean_lower = current_query.lower()

        # If query already explicitly names Messi, no rewrite is needed
        if "messi" in clean_lower or "lionel" in clean_lower:
            return current_query

        # Contextualize follow-ups:
        clean_user = current_query.strip("?. ")
        # Replace 'his' with 'Lionel Messi's'
        contextualized = re.sub(r"\bhis\b", "Lionel Messi's", clean_user, flags=re.IGNORECASE)
        # Replace 'he/him/it' with 'Lionel Messi'
        contextualized = re.sub(r"\b(he|him|it)\b", "Lionel Messi", contextualized, flags=re.IGNORECASE)

        if "messi" not in contextualized.lower():
            return f"{contextualized} (context: Lionel Messi)"
        return contextualized

