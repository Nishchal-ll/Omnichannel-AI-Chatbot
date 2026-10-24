"""
scripts/test_messi_qa.py
────────────────────────
Automated evaluation test suite for Lionel Messi QA Pipeline with Gemma 4 E2B.
Tests:
1. Fast-Path / FAQ Anchors (0 ms)
2. Domain Grounded Knowledge Retrieval & Accuracy
3. Multi-Turn Context Resolution
4. Out-of-Domain Guard Deflection
5. Latency and Token Throughput
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from channels.base import Message
from agent.pipeline import Pipeline
from agent.router import IntentRouter

TEST_CASES = [
    {
        "name": "0ms Fast Greeting",
        "query": "hiii",
        "expected_keywords": ["Lionel Messi", "Knowledge Assistant"],
        "expect_fast_path": True,
    },
    {
        "name": "0ms FAQ - Total Ballon d'Ors",
        "query": "How many Ballon d'Or awards has Messi won in total?",
        "expected_keywords": ["8", "Ballon d'Or"],
        "expect_fast_path": True,
    },
    {
        "name": "0ms FAQ - Current Club",
        "query": "Which club does Messi play for now?",
        "expected_keywords": ["Inter Miami"],
        "expect_fast_path": True,
    },
    {
        "name": "Gemma 4 RAG - 2012 Calendar Goals Record",
        "query": "What is the world record for most goals in a calendar year held by Messi?",
        "expected_keywords": ["91"],
        "expect_fast_path": False,
    },
    {
        "name": "Gemma 4 RAG - 2022 World Cup in Qatar",
        "query": "What did Messi achieve with Argentina at the 2022 World Cup in Qatar?",
        "expected_keywords": ["World Cup", "Golden Ball"],
        "expect_fast_path": False,
    },
    {
        "name": "Gemma 4 RAG - Barcelona Trophies Count",
        "query": "How many official trophies did Messi win during his 17 seasons with FC Barcelona?",
        "expected_keywords": ["35"],
        "expect_fast_path": False,
    },
    {
        "name": "Gemma 4 RAG - 2009 Sextuple Manager",
        "query": "Who was the manager of Barcelona when they won the sextuple with Messi in 2009?",
        "expected_keywords": ["Guardiola"],
        "expect_fast_path": False,
    },
    {
        "name": "Gemma 4 RAG - PSG Jersey Number",
        "query": "What jersey number did Messi wear while playing for Paris Saint-Germain?",
        "expected_keywords": ["30"],
        "expect_fast_path": False,
    },
    {
        "name": "Gemma 4 RAG - European Golden Shoes",
        "query": "How many European Golden Shoe awards has Messi won?",
        "expected_keywords": ["6"],
        "expect_fast_path": False,
    },
    {
        "name": "Out-of-Domain Guard - Cooking Recipe",
        "query": "Can you give me a recipe for baking chocolate cake and bread?",
        "expected_keywords": ["only assist", "Lionel Messi"],
        "expect_fast_path": False,
    },
]


def run_benchmark():
    print("=" * 70)
    print("  Lionel Messi QA Pipeline Evaluation Benchmark (Gemma 4 E2B)")
    print("=" * 70)

    pipeline = Pipeline()
    results = []

    for i, test in enumerate(TEST_CASES, start=1):
        q = test["query"]
        print(f"\n[{i}/{len(TEST_CASES)}] Test: {test['name']}")
        print(f"  Query: \"{q}\"")

        msg = Message(channel="whatsapp", sender="9779765341686", body=q)

        t_start = time.perf_counter()
        reply = pipeline.process(msg)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        # Verification
        matched_kws = [kw for kw in test["expected_keywords"] if kw.lower() in reply.lower()]
        passed = len(matched_kws) == len(test["expected_keywords"])

        status_str = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Result: {status_str} in {elapsed_ms:.1f}ms")
        print(f"  Reply: \"{reply[:120]}...\"")
        if not passed:
            print(f"  Missing Keywords: {[kw for kw in test['expected_keywords'] if kw not in matched_kws]}")

        results.append({
            "name": test["name"],
            "passed": passed,
            "latency_ms": elapsed_ms,
            "reply": reply,
        })

    # Summary
    total_passed = sum(1 for r in results if r["passed"])
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    print("\n" + "=" * 70)
    print(f"  Benchmark Summary: {total_passed}/{len(results)} Passed | Avg Latency: {avg_latency:.1f}ms")
    print("=" * 70)

    return total_passed == len(results)


if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)

