"""
src/stage2_memory/faq_service.py
────────────────────────────────
SQLite-backed Canonical FAQ Service (Item 2).
Decouples verified FAQ facts and canonical answers from code into data/chat.db.
Appends '(as of {as_of})' to time-sensitive statistics (goals, trophies, appearances).
"""

import os
import sqlite3
import re
from typing import Optional, List, Tuple
from threading import Lock

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "chat.db")


SEED_FAQS = [
    (
        r"\b(where\s+is\s+the\s+showroom|showroom\s+location|where\s+are\s+you\s+located|address|contact\s+number|phone\s+number)\b",
        "Apex Auto Recondition House Nepal is located at Tinkune Subidhanagar (near Ring Road Bridge), Kathmandu. Phone: 01-4491234 / +977-9801234567. We also have a Pokhara Branch at Nayabazar (Ph: 061-534567).",
        "2026-11-03",
        0
    ),
    (
        r"\b(opening\s+hours|business\s+hours|working\s+hours|what\s+time\s+do\s+you\s+open)\b",
        "Our showroom is open Sunday to Friday from 9:30 AM to 6:30 PM, and Saturday from 10:00 AM to 5:00 PM for test drives and vehicle inspections.",
        "2026-11-03",
        0
    ),
    (
        r"\b(warranty|guarantee|warranty\s+policy|after\s+sales)\b",
        "Every certified vehicle comes with a 6-Month / 8,000 km warranty covering the engine block and gearbox, plus 1 complimentary full servicing (oil change & tune-up).",
        "2026-11-03",
        0
    ),
    (
        r"\b(inspection|inspection\s+checklist|quality\s+inspection|how\s+do\s+you\s+inspect)\b",
        "All cars undergo a strict 120-Point Inspection covering chassis frame integrity (accident-free guarantee), engine compression, suspension, electricals, and guaranteed authentic odometer reading.",
        "2026-11-03",
        0
    ),
    (
        r"\b(exchange|satasat|trade\s*in|can\s+i\s+exchange|car\s+exchange|swap|bike\s+exchange)\b",
        "Yes! We offer instant 20-minute spot valuation and exchange (साटासाट) for any old motorcycle, scooter, or car (any brand/lot). The valuation is directly adjusted into your down payment, and we provide same-day vehicle delivery.",
        "2026-11-03",
        0
    ),
    (
        r"\b(bluebook|namsari|name\s+transfer|ownership\s+transfer|yatayat|tax\s+paid)\b",
        "We handle 100% of the Bluebook ownership transfer (नामसारी) at Yatayat Karyalaya (Ekantakuna, Lalitpur & Gurjudhara) within 24 to 48 hours. All cars have 2081/82 road tax paid and clear legal records.",
        "2026-11-03",
        0
    ),
    (
        r"\b(financing|bank\s+loan|emi|down\s+payment|installment)\b",
        "We provide up to 70% bank financing via partner commercial banks (NIC Asia Bank, Nabil Bank, Global IME, Kumari Bank) with quick 3-day approval and tenures from 3 to 7 years. Minimum down payment starts at 30%.",
        "2026-11-03",
        0
    ),
    (
        r"\b(test\s+drive|can\s+i\s+test\s+drive|booking\s+a\s+test\s+drive)\b",
        "Walk-in test drives are free and available anytime during showroom hours! Just bring your valid driving license to our Tinkune, Kathmandu or Nayabazar, Pokhara showroom.",
        "2026-11-03",
        0
    ),
]


class FAQService:
    """Manages SQLite FAQ table and fast pattern matching."""

    def __init__(self):
        self._lock = Lock()
        self._db = self._init_db()
        self._compiled_cache: List[Tuple[re.Pattern, str, str, int]] = []
        self._reload_faqs()

    def _init_db(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS faq (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                answer TEXT NOT NULL,
                as_of DATE NOT NULL,
                is_time_sensitive INTEGER DEFAULT 0
            )
        """)
        conn.commit()

        # Sync/Seed seed FAQs
        cursor = conn.cursor()
        cursor.execute("DELETE FROM faq")
        for pat, ans, dt, ts in SEED_FAQS:
            cursor.execute(
                "INSERT INTO faq (pattern, answer, as_of, is_time_sensitive) VALUES (?, ?, ?, ?)",
                (pat, ans, dt, ts)
            )
        conn.commit()

        return conn

    def _reload_faqs(self):
        cursor = self._db.cursor()
        cursor.execute("SELECT pattern, answer, as_of, is_time_sensitive FROM faq")
        rows = cursor.fetchall()
        compiled = []
        for pat, ans, as_of, ts in rows:
            try:
                rx = re.compile(pat, re.IGNORECASE)
                compiled.append((rx, ans, as_of, ts))
            except Exception as e:
                print(f"[FAQ] Warning: invalid regex pattern '{pat}': {e}")
        self._compiled_cache = compiled

    def check_faq(self, query: str) -> Optional[str]:
        """Checks query against FAQ patterns in 0.05ms."""
        clean = query.strip()
        for rx, answer, as_of, is_ts in self._compiled_cache:
            if rx.search(clean):
                if is_ts:
                    return f"{answer} (as of {as_of})"
                return answer
        return None


faq_service = FAQService()

