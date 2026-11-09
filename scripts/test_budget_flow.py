"""
scripts/test_budget_flow.py
───────────────────────────
Tests the exact 4-turn multi-turn conversation reported by the user:
Turn 1: "Customer budget is 16."
Turn 2: "what is my budge"
Turn 3: "so which car falls under my budget"
Turn 4: "what is its price?"
Turn 5 (tricky): "is Hyundai Grand i10 in my budget?"
Turn 6 (tricky): "can I buy a Creta with 16 lakhs?"
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stage3_brain.orchestrator import orchestrator
from stage2_memory.session import session_buffer

def main():
    test_user = "test_budget_user_001"
    # Clean previous state
    session_buffer.delete_all_user_data(test_user)

    turns = [
        "Customer budget is 16.",
        "what is my budge",
        "so which car falls under my budget",
        "what is its price?",
        "i increased my budget its 20 lakhs",
        "which cars fall under my budget now?",
        "is the Grand i10 in my budget?"
    ]

    print("==================================================")
    print("RUNNING MULTI-TURN BUDGET CONVERSATION TEST")
    print("==================================================")

    for i, query in enumerate(turns, 1):
        print(f"\n--- TURN {i} ---")
        print(f"USER: {query}")
        result = orchestrator.process(test_user, query)
        reply = result.get("reply", "")
        metrics = result.get("metrics", {})
        path_type = metrics.get("path_type", "unknown")
        total_ms = metrics.get("total_ms", 0.0)

        print(f"BOT ({path_type}, {total_ms}ms):\n{reply}")

    print("\n==================================================")
    print("TEST FINISHED")
    print("==================================================")

if __name__ == "__main__":
    main()
