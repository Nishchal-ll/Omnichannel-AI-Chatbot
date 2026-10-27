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
        r"\b(how\s+many\s+clubs|which\s+clubs|what\s+clubs|all\s+clubs|clubs\s+(has\s+)?(messi|he|lionel)\s+played)\b",
        "Lionel Messi has played for 3 senior professional clubs throughout his career: FC Barcelona (2004–2021), Paris Saint-Germain (PSG, 2021–2023), and Inter Miami CF (2023–present). In his youth in Argentina, he also played for Grandoli and Newell's Old Boys. [Senior Professional Clubs Summary (Clubs Played)]",
        "2026-10-01",
        0
    ),
    (
        r"\b(current\s+club|which\s+club\s+does\s+messi\s+play\s+for|where\s+does\s+messi\s+play\s+now)\b",
        "Lionel Messi currently plays for Major League Soccer (MLS) club Inter Miami CF in the United States, wearing jersey number 10. [Inter Miami CF and Major League Soccer]",
        "2026-10-01",
        0
    ),
    (
        r"\b(how\s+many\s+ballon\s*d'?or|total\s+ballon\s*d'?or)\b",
        "Lionel Messi has won a world-record 8 Ballon d'Or awards (2009, 2010, 2011, 2012, 2015, 2019, 2021, and 2023). [Individual Awards and Honors]",
        "2026-10-01",
        0
    ),
    (
        r"\b(goals\s+in\s+2012|record\s+in\s+2012|world\s+record.*2012|91\s+goals)\b",
        "Lionel Messi holds the Guinness World Record for scoring 91 official goals in the calendar year 2012 (79 for Barcelona, 12 for Argentina). [World Records and Career Statistics]",
        "2026-10-01",
        0
    ),
    (
        r"\b(did\s+messi\s+win\s+(the\s+)?world\s+cup|world\s+cup\s+2022)\b",
        "Yes, Lionel Messi captained Argentina to victory in the 2022 FIFA World Cup in Qatar, scoring 7 goals and winning the tournament's Golden Ball. [Argentina National Team Career]",
        "2026-10-01",
        0
    ),
    (
        r"\b(when\s+was\s+messi\s+born|messi\s+birthday|date\s+of\s+birth)\b",
        "Lionel Messi was born on June 24, 1987, in Rosario, Santa Fe, Argentina. [Overview and Early Life]",
        "2026-10-01",
        0
    ),
    (
        r"\b(how\s+many\s+trophies|total\s+trophies\s+won)\b",
        "Lionel Messi has won over 45 official team trophies across his senior career with FC Barcelona, PSG, Inter Miami, and Argentina. [World Records and Career Statistics]",
        "2026-10-01",
        1
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

        # Seed if empty
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM faq")
        if cursor.fetchone()[0] == 0:
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

