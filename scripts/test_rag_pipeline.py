#!/usr/bin/env python3
"""
scripts/test_rag_pipeline.py
────────────────────────────
Automated Evaluation Suite for Naasa Securities (Broker 58) AI Agent.
Evaluates the RAG + Agent pipeline directly WITHOUT requiring WhatsApp or Email network connectivity.

Test Categories:
1. In-Domain Fact Grounding (Broker 58, Head Office location)
2. Brokerage Commission Precision (0.40% under Rs. 50k slab)
3. Multi-Turn Context & Pronoun Resolution (resolving 'for that' across turns)
4. User Contact & Profile Persistence (tracking client code from conversation)
5. Out-of-Context Rejection Guard (strictly deflecting non-brokerage queries like cake recipes or world trivia)

Outputs a comprehensive [PASSED] / [FAILED] verdict for each test.
"""

import sys
import os
import re

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from channels.base import Message
from agent.pipeline import Pipeline
from agent.session import session_buffer
from agent.prompt import OUT_OF_CONTEXT_DEFLECTION
from knowledge import ingest_kb, kb_stats


class TestReport:
    def __init__(self):
        self.results = []

    def record(self, test_name: str, passed: bool, details: str, output: str = ""):
        self.results.append({
            "name": test_name,
            "passed": passed,
            "details": details,
            "output": output
        })

    def print_summary(self):
        print("\n" + "=" * 80)
        print("          NAASA SECURITIES (BROKER 58) RAG PIPELINE EVALUATION REPORT")
        print("=" * 80)
        
        all_passed = True
        for i, r in enumerate(self.results, 1):
            tag = "\033[92m[PASSED]\033[0m" if r["passed"] else "\033[91m[FAILED]\033[0m"
            if not r["passed"]:
                all_passed = False
            print(f"{i}. {r['name']:<50} {tag}")
            print(f"   Criteria: {r['details']}")
            if r["output"]:
                sample = r["output"].replace('\n', ' ')[:100]
                print(f"   Response: \"{sample}...\"")
            print("-" * 80)

        print("\nSUMMARY:")
        passed_count = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        pct = (passed_count / total) * 100 if total else 0
        print(f"Total Tests: {total} | Passed: {passed_count} | Failed: {total - passed_count} | Success Rate: {pct:.1f}%")

        if all_passed:
            print("\033[92m>>> OVERALL STATUS: ALL TESTS PASSED SUCCESSFULLY! <<<\033[0m\n")
            return 0
        else:
            print("\033[91m>>> OVERALL STATUS: ONE OR MORE TESTS FAILED! <<<\033[0m\n")
            return 1


def run_evaluation():
    report = TestReport()
    
    print("\n[Setup] Step 1: Ingesting Naasa Securities Knowledge Base into ChromaDB...")
    num_chunks = ingest_kb(force=False)
    stats = kb_stats()
    print(f"[Setup] Loaded {stats['total_chunks']} chunks in {stats['collection']}.")

    pipeline = Pipeline()

    # ── Test 1: In-Domain Fact Grounding ──────────────────────────────
    print("\n[Running Test 1] In-Domain Fact Grounding...")
    t1_sender = "eval_user_fact_1"
    t1_msg = Message(
        channel="eval_test",
        sender=t1_sender,
        body="What is your broker number and where is your head office located in Kathmandu?"
    )
    t1_reply = pipeline.process(t1_msg)
    
    t1_has_broker = "58" in t1_reply
    t1_has_location = any(loc in t1_reply.lower() for loc in ["lal durbar", "lal colony", "kathmandu"])
    t1_pass = t1_has_broker and t1_has_location
    report.record(
        test_name="In-Domain Fact Grounding (Broker No. & Head Office)",
        passed=t1_pass,
        details="Must identify Broker 58 and Lal Durbar / Kathmandu head office",
        output=t1_reply
    )

    # ── Test 2: Fee Calculation Precision ─────────────────────────────
    print("\n[Running Test 2] Fee Calculation Precision...")
    t2_sender = "eval_user_fee_2"
    t2_msg = Message(
        channel="eval_test",
        sender=t2_sender,
        body="What is the brokerage commission rate for buying Rs. 40,000 worth of shares?"
    )
    t2_reply = pipeline.process(t2_msg)
    t2_pass = ("0.40%" in t2_reply) or ("0.4%" in t2_reply)
    report.record(
        test_name="Brokerage Commission Precision (Slab <= Rs 50k)",
        passed=t2_pass,
        details="Must state 0.40% commission for transaction up to Rs. 50,000",
        output=t2_reply
    )

    # ── Test 3: Multi-Turn Context & Pronoun Resolution ───────────────
    print("\n[Running Test 3] Multi-Turn Context & Pronoun Resolution...")
    t3_sender = "eval_user_multiturn_3"
    # Turn 1
    t3_m1 = Message(
        channel="eval_test",
        sender=t3_sender,
        body="How can I open an online TMS account with Naasa Securities?"
    )
    t3_r1 = pipeline.process(t3_m1)
    
    # Turn 2: uses 'for that' (pronoun resolution needed)
    t3_m2 = Message(
        channel="eval_test",
        sender=t3_sender,
        body="What documents are required to upload for that?"
    )
    t3_r2 = pipeline.process(t3_m2)
    t3_pass = any(doc in t3_r2.lower() for doc in ["citizenship", "photo", "bank", "cheque", "signature", "pan"])
    report.record(
        test_name="Multi-Turn Context Resolution ('for that' -> KYC docs)",
        passed=t3_pass,
        details="Turn 2 must resolve 'for that' and state KYC documents (citizenship, photo, bank cheque)",
        output=t3_r2
    )

    # ── Test 4: User Contact & Profile Persistence ────────────────────
    print("\n[Running Test 4] User Contact & Profile Persistence...")
    t4_sender = "eval_user_contact_4"
    t4_msg = Message(
        channel="eval_test",
        sender=t4_sender,
        body="Hello, my name is Sita Shrestha, client code is 589922 and phone is 9841234567. Please check my account."
    )
    t4_reply = pipeline.process(t4_msg)
    contact = session_buffer.get_user_contact(t4_sender)
    t4_pass = (contact.get("client_code") == "589922") and (contact.get("phone") == "9841234567")
    report.record(
        test_name="User Profile & Contact Memory Tracking",
        passed=t4_pass,
        details="Must extract and persist client_code=589922 and phone=9841234567 in session memory",
        output=f"Stored Contact State: {contact}"
    )

    # ── Test 5: Out-of-Context Rejection Guard (Query 1: Baking) ─────
    print("\n[Running Test 5] Out-of-Context Rejection Guard (Recipe)...")
    t5_sender = "eval_user_ooc_5"
    t5_msg = Message(
        channel="eval_test",
        sender=t5_sender,
        body="Can you give me a step-by-step recipe to bake a chocolate cake at home?"
    )
    t5_reply = pipeline.process(t5_msg)
    
    # Must NOT contain actual recipe words (flour, cocoa, bake, oven, eggs)
    has_recipe_leak = any(w in t5_reply.lower() for w in ["flour", "cocoa", "preheat", "baking powder", "eggs"])
    has_deflection = ("naasa securities" in t5_reply.lower()) or ("broker 58" in t5_reply.lower()) or (t5_reply == OUT_OF_CONTEXT_DEFLECTION)
    t5_pass = has_deflection and not has_recipe_leak
    report.record(
        test_name="Out-of-Context Guard #1 (Chocolate Cake Recipe)",
        passed=t5_pass,
        details="Must decline recipe request and deflect to Naasa Securities Broker 58 services",
        output=t5_reply
    )

    # ── Test 6: Out-of-Context Rejection Guard (Query 2: Trivia) ─────
    print("\n[Running Test 6] Out-of-Context Rejection Guard (World Trivia)...")
    t6_sender = "eval_user_ooc_6"
    t6_msg = Message(
        channel="eval_test",
        sender=t6_sender,
        body="What is the capital city of France?"
    )
    t6_reply = pipeline.process(t6_msg)
    has_trivia_leak = "paris" in t6_reply.lower()
    has_deflection_2 = ("naasa securities" in t6_reply.lower()) or ("broker 58" in t6_reply.lower()) or (t6_reply == OUT_OF_CONTEXT_DEFLECTION)
    t6_pass = has_deflection_2 and not has_trivia_leak
    report.record(
        test_name="Out-of-Context Guard #2 (World Trivia - Paris)",
        passed=t6_pass,
        details="Must decline general trivia and deflect to Naasa Broker 58 services (must not say Paris)",
        output=t6_reply
    )

    # Print overall report
    exit_code = report.print_summary()
    return exit_code


if __name__ == "__main__":
    code = run_evaluation()
    sys.exit(code)
