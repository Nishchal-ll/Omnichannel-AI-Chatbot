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
    r"^(hi|hii|hey|heya|hello|howdy|namaste|namaskar|good\s+morning|good\s+afternoon|good\s+evening|gm)\b([!\.\s]*(car|dealer|showroom|apex|bot|ai|there|everyone))?[!\.\s]*$",
    re.IGNORECASE
)

# Fast Gratitude / Signoff Patterns
SIGNOFF_REGEX = re.compile(
    r"^(thank\s*you(\s+so\s+much|\s+a\s+lot)?|thanks(\s+a\s+lot|\s+so\s+much)?|thank\s*u|thx|ty|dhanyabad|bye|goodbye|see\s*you|take\s*care)\b[!\.\s]*$",
    re.IGNORECASE
)

# Standalone pronoun pattern for cars
PRONOUN_PATTERN = re.compile(
    r"\b(it|this|that|its|the\s+car|this\s+car|this\s+vehicle|the\s+suv|how\s+much\s+is\s+it|what\s+is\s+its\s+price|is\s+it\s+available)\b",
    re.IGNORECASE
)

# Obvious Out-Of-Domain Triggers (Cooking, Football, Politics, Visa, Weather, Coding)
OUT_OF_DOMAIN_PATTERNS = [
    r"\b(bake|cake|recipe|cook|cooking|pizza|pasta|bread)\b",
    r"\b(messi|ronaldo|barcelona|real\s+madrid|ballon\s*d'?or|world\s+cup|football|cricket|nba)\b",
    r"\b(capital\s+of|weather\s+in|earthquake|volcano|alien|dinosaur)\b",
    r"\b(write\s+a\s+poem|tell\s+me\s+a\s+joke|sing\s+a\s+song|python\s+script|binary\s+tree)\b",
    r"\b(visa\s+for\s+japan|ielts|toefl|study\s+in\s+australia)\b",
]
OUT_OF_DOMAIN_REGEX = re.compile("|".join(OUT_OF_DOMAIN_PATTERNS), re.IGNORECASE)

DOMAIN_KEYWORDS = [
    "car", "cars", "auto", "vehicle", "vehicles", "recondition", "showroom", "inventory",
    "hyundai", "suzuki", "maruti", "nissan", "tata", "kia", "toyota", "byd", "mahindra",
    "santro", "i10", "nios", "magnite", "alto", "wagonr", "swift", "baleno", "brezza",
    "creta", "venue", "tiago", "nexon", "sonet", "seltos", "kicks", "raize", "rush", "hilux", "atto",
    "engine", "cc", "motor", "mileage", "odometer", "km", "price", "cost", "lakh", "lakhs", "rupees", "rs",
    "transmission", "automatic", "manual", "amt", "cvt", "hybrid", "electric", "ev", "petrol", "diesel",
    "warranty", "exchange", "satasat", "trade-in", "trade", "swap", "finance", "financing", "emi", "loan",
    "down payment", "inspection", "certified", "test drive", "stock", "bluebook", "namsari", "yatayat",
    "bagmati", "pradesh", "lot", "ekantakuna", "gurjudhara", "tinkune", "kathmandu", "pokhara"
]

# 0ms Instant FAQ Canonical Anchors
FAQ_ANCHORS = [
    (
        re.compile(r"\b(where\s+is\s+the\s+showroom|showroom\s+location|where\s+are\s+you\s+located|address|contact\s+number|phone\s+number)\b", re.IGNORECASE),
        "Apex Auto Recondition House Nepal is located at Tinkune Subidhanagar (near Ring Road Bridge), Kathmandu. Phone: 01-4491234 / +977-9801234567. We also have a Pokhara Branch at Nayabazar (Ph: 061-534567)."
    ),
    (
        re.compile(r"\b(opening\s+hours|business\s+hours|working\s+hours|what\s+time\s+do\s+you\s+open)\b", re.IGNORECASE),
        "Our showroom is open Sunday to Friday from 9:30 AM to 6:30 PM, and Saturday from 10:00 AM to 5:00 PM for test drives and vehicle inspections."
    ),
    (
        re.compile(r"\b(warranty|guarantee|warranty\s+policy|after\s+sales)\b", re.IGNORECASE),
        "Every certified vehicle comes with a 6-Month / 8,000 km warranty covering the engine block and gearbox, plus 1 complimentary full servicing (oil change & tune-up)."
    ),
    (
        re.compile(r"\b(inspection|inspection\s+checklist|quality\s+inspection|how\s+do\s+you\s+inspect)\b", re.IGNORECASE),
        "All cars undergo a strict 120-Point Inspection covering chassis frame integrity (accident-free guarantee), engine compression, suspension, electricals, and guaranteed authentic odometer reading."
    ),
    (
        re.compile(r"\b(exchange|satasat|trade\s*in|can\s+i\s+exchange|car\s+exchange|swap|bike\s+exchange)\b", re.IGNORECASE),
        "Yes! We offer instant 20-minute spot valuation and exchange (साटासाट) for any old motorcycle, scooter, or car (any brand/lot). The valuation is directly adjusted into your down payment, and we provide same-day vehicle delivery."
    ),
    (
        re.compile(r"\b(bluebook|namsari|name\s+transfer|ownership\s+transfer|yatayat|tax\s+paid)\b", re.IGNORECASE),
        "We handle 100% of the Bluebook ownership transfer (नामसारी) at Yatayat Karyalaya (Ekantakuna, Lalitpur & Gurjudhara) within 24 to 48 hours. All cars have 2081/82 road tax paid and clear legal records."
    ),
    (
        re.compile(r"\b(financing|bank\s+loan|emi|down\s+payment|installment)\b", re.IGNORECASE),
        "We provide up to 70% bank financing via partner commercial banks (NIC Asia Bank, Nabil Bank, Global IME, Kumari Bank) with quick 3-day approval and tenures from 3 to 7 years. Minimum down payment starts at 30%."
    ),
    (
        re.compile(r"\b(test\s+drive|can\s+i\s+test\s+drive|booking\s+a\s+test\s+drive)\b", re.IGNORECASE),
        "Walk-in test drives are free and available anytime during showroom hours! Just bring your valid driving license to our Tinkune, Kathmandu or Nayabazar, Pokhara showroom."
    ),
]


class IntentRouter:
    """Stage 2 Intent Router & Context contextualizer for Recondition Car House."""

    @staticmethod
    def check_fast_path(query: str) -> Optional[str]:
        """Checks for 0ms deterministic responses (greetings, signoffs, FAQs)."""
        clean = normalize_query(query)

        if GREETING_REGEX.match(clean):
            return (
                "Namaste! Welcome to Apex Auto Recondition House Nepal (Tinkune, Kathmandu). "
                "How can I assist you today with our certified cars (Santro, i10, Magnite, Swift, Alto, Brezza, Creta), "
                "instant exchange (साटासाट), Bluebook transfer, or bank EMI?"
            )

        if SIGNOFF_REGEX.match(clean):
            return (
                "Dhanyabad! You are always welcome at Apex Auto Recondition House Nepal. "
                "Visit our Tinkune, Kathmandu showroom or call us on 01-4491234 / 9801234567 anytime. Have a great day!"
            )

        for pattern, response in FAQ_ANCHORS:
            if pattern.search(clean):
                return response

        return None

    @staticmethod
    def check_memory_recall(query: str) -> bool:
        """Detects if user is asking about their saved budget, preferred cars, or profile."""
        clean = query.lower().strip()
        patterns = [
            r"\bwhat('?s|\s+is)\s+my\s+(budget|name|preference|choice|selection)\b",
            r"\bwho\s+am\s+i\b",
            r"\bwhere\s+do\s+i\s+live\b",
            r"\bwhat\s+(car|cars|brand)\s+did\s+i\s+(say|choose|prefer|want)\b",
            r"\bwhat\s+(car|cars)\s+match\s+my\s+budget\b",
            r"\bwhat\s+do\s+you\s+(know|remember)\s+about\s+me\b",
            r"\bdo\s+you\s+remember\s+me\b",
            r"\bmy\s+details\b",
            r"\bmy\s+profile\b",
            r"\bshow\s+my\s+preferences\b",
            r"\bshow\s+my\s+memories\b",
        ]
        return any(re.search(p, clean) for p in patterns)

    @staticmethod
    def check_memory_clear(query: str) -> bool:
        """Detects if user is asking to forget/clear their memories."""
        clean = query.lower().strip()
        patterns = [
            r"\b(forget|clear|delete|erase)\s+(my\s+)?(memory|memories|information|profile|details|preferences)\b",
            r"\bforget\s+me\b",
        ]
        return any(re.search(p, clean) for p in patterns)

    @staticmethod
    def is_out_of_domain(query: str) -> bool:
        """Detects if query is completely unrelated to cars, automotive dealership, or inventory."""
        clean_lower = query.lower()
        if any(kw in clean_lower for kw in DOMAIN_KEYWORDS):
            return False
        if OUT_OF_DOMAIN_REGEX.search(clean_lower):
            return True
        return False

    @staticmethod
    def rewrite_query(current_query: str, chat_history: List[ChatMessage]) -> str:
        """Resolves pronouns ('it', 'this car', 'its') and context against prior turns."""
        if not chat_history:
            return current_query

        clean_lower = current_query.lower()

        # Check if query already mentions a specific car brand or model
        KNOWN_MODELS = [
            "santro", "i10", "nios", "magnite", "alto", "wagonr", "swift", "baleno", "brezza",
            "creta", "venue", "tiago", "nexon", "sonet", "seltos", "kicks", "raize", "rush", "hilux", "atto",
            "rav4", "corolla"
        ]
        if any(m in clean_lower for m in KNOWN_MODELS):
            return current_query

        # Look back in history to find the most recently mentioned vehicle
        referenced_car = ""
        for turn in reversed(chat_history[-3:]):
            turn_lower = turn.content.lower()
            for m in KNOWN_MODELS:
                if m in turn_lower:
                    referenced_car = m.title()
                    break
            if referenced_car:
                break

        if not referenced_car:
            return current_query

        # Contextualize follow-ups:
        contextualized = current_query.strip("?. ")
        contextualized = re.sub(r"\b(its|it|the\s+car|this\s+car|the\s+suv)\b", referenced_car, contextualized, flags=re.IGNORECASE)
        if referenced_car.lower() not in contextualized.lower():
            contextualized = f"{contextualized} (regarding: {referenced_car})"

        return contextualized

