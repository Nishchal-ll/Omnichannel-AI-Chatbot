"""
src/agent/router.py
───────────────────
Intent routing, fast-path anchors, query contextualization, and out-of-context guard
for Lionel Messi AI Knowledge Assistant.
"""

import re
from typing import List, Dict, Optional, Tuple


def normalize_query(query: str) -> str:
    """Collapses elongated/repeated characters (e.g. 'hiii' -> 'hi', 'heyyy' -> 'hey')."""
    clean = query.strip()
    clean = re.sub(r"(.)\1{2,}", r"\1", clean)
    return clean


# Fast Greeting Patterns
GREETING_REGEX = re.compile(
    r"^(hi|hii|hey|heya|hello|howdy|namaste|namaskar|good\s+morning|good\s+afternoon|good\s+evening|gm|k\s+cha|sanchai\s+cha|नमस्ते|नमस्कार)\b([!\.\s]*(messi|leo|bot|ai|there|everyone))?[!\.\s]*$",
    re.IGNORECASE
)

# Fast Gratitude / Signoff Patterns
SIGNOFF_REGEX = re.compile(
    r"^(thank\s*you(\s+so\s+much|\s+a\s+lot)?|thanks(\s+a\s+lot|\s+so\s+much)?|thank\s*u|thx|ty|dhanyabad|dhanybhad|bye|goodbye|dhanyabaad|see\s*you|take\s*care|धन्यवाद)\b[!\.\s]*$",
    re.IGNORECASE
)

# Strict standalone pronoun regex with word boundaries
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
    r"\b(stock\s*broker|sebon|nepse|tms\s*58|naasa)\b",
]
OUT_OF_DOMAIN_REGEX = re.compile("|".join(OUT_OF_DOMAIN_PATTERNS), re.IGNORECASE)

# Keywords that confirm domain relevance to Lionel Messi & Football
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
    """Routes customer queries, executes deterministic fast-paths, and contextualizes follow-ups."""

    @staticmethod
    def check_fast_path(query: str) -> Optional[str]:
        """Check for 0ms deterministic responses (greetings, signoffs, FAQs)."""
        clean = normalize_query(query)

        # 1. Greetings (including elongated forms: hiii, heyyy, etc.)
        if GREETING_REGEX.match(clean):
            return (
                "Hello! Welcome to the Lionel Messi AI Knowledge Assistant. "
                "How can I assist you with facts, statistics, trophies, or records from Lionel Messi's career?"
            )

        # 2. Signoffs / Gratitude
        if SIGNOFF_REGEX.match(clean):
            return (
                "You are most welcome! Feel free to ask anytime about Lionel Messi's career and achievements. "
                "Enjoy following the GOAT!"
            )

        # 3. 0ms Canonical FAQ Anchors
        for pattern, response in FAQ_ANCHORS:
            if pattern.search(clean):
                return response

        return None

    @staticmethod
    def is_out_of_domain(query: str) -> bool:
        """
        Detects if query is completely unrelated to Lionel Messi / Football.
        Returns True if query matches obvious non-domain topics and contains no domain keywords.
        """
        clean_lower = query.lower()

        # If it explicitly mentions Messi/football keywords, it's NOT out of domain
        if any(kw in clean_lower for kw in DOMAIN_KEYWORDS):
            return False

        # If it matches obvious out-of-domain patterns, flag it
        if OUT_OF_DOMAIN_REGEX.search(clean_lower):
            return True

        return False

    @staticmethod
    def rewrite_query(current_query: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Resolves pronouns ('he', 'his', 'him', 'it') against prior conversation turns.
        Only rewrites if the query actually contains unresolved pronouns and omits explicit entity names.
        """
        if not chat_history:
            return current_query

        clean_lower = current_query.lower()

        # If query already explicitly names Messi, no rewrite is needed
        if "messi" in clean_lower or "lionel" in clean_lower:
            return current_query

        # Check if query contains standalone pronoun references
        has_pronoun = bool(PRONOUN_PATTERN.search(clean_lower))
        if not has_pronoun and len(current_query.split()) > 3:
            return current_query

        # Retrieve the most recent user turn
        last_user = ""
        for turn in reversed(chat_history):
            if turn["role"] == "user":
                last_user = turn["content"]
                break

        if not last_user:
            return current_query

        # Form a clean contextualized query
        clean_user = current_query.strip("?. ")
        contextualized = PRONOUN_PATTERN.sub("Lionel Messi", clean_user)
        return f"{contextualized} (context: Lionel Messi)"

    @staticmethod
    def extract_contact_info(text: str) -> Dict[str, str]:
        """Extracts sender phone or email if provided."""
        info = {}
        phone_match = re.search(r"\b(9[78]\d{8})\b", text)
        if phone_match:
            info["phone"] = phone_match.group(1)
        email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
        if email_match:
            info["email"] = email_match.group(0)
        return info
