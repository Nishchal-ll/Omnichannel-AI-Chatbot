"""
scripts/run_eval.py
───────────────────
Automated Benchmark Evaluation Suite for Lionel Messi AI Assistant.
Reads eval.jsonl, executes test queries through PipelineOrchestrator,
and calculates accuracy, latency breakdown, and OOD false-positive rates.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stage3_brain.orchestrator import orchestrator
from stage2_memory.session import session_buffer
from stage2_memory.memory_service import memory_service

EVAL_FILE = os.path.join(os.path.dirname(__file__), "..", "eval.jsonl")


def run_benchmark():
    if not os.path.exists(EVAL_FILE):
        print(f"Error: {EVAL_FILE} not found.")
        sys.exit(1)

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        test_cases = [json.loads(line) for line in f if line.strip()]

    print(f"\n=======================================================")
    print(f"📊 RUNNING BENCHMARK EVALUATION ({len(test_cases)} Test Cases)")
    print(f"=======================================================\n")

    passed = 0
    failed = 0
    latencies = []
    category_stats = {}
    ood_false_positives = 0
    ood_false_negatives = 0

    eval_user = "benchmark_eval_user"

    for idx, case in enumerate(test_cases, 1):
        case_id = case.get("id", idx)
        category = case.get("category", "general")
        q = case["q"]
        history = case.get("history", [])
        must_include = case.get("must_include", [])
        expect = case.get("expect")

        if category not in category_stats:
            category_stats[category] = {"total": 0, "passed": 0, "latencies": []}
        category_stats[category]["total"] += 1

        # Reset session buffer for test user
        session_buffer.clear_history(eval_user)
        # If the case does not explicitly continue memory, clear memory too unless it's a memory chain
        if category != "memory":
            memory_service.clear_memories(eval_user)

        # Pre-populate history if specified
        for hist_turn in history:
            orchestrator.process(eval_user, hist_turn, channel="chat")

        t0 = time.perf_counter()
        res = orchestrator.process(eval_user, q, channel="chat")
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        latencies.append(duration_ms)
        category_stats[category]["latencies"].append(duration_ms)

        reply = res.get("reply", "")
        path_type = res.get("metrics", {}).get("path_type", "")
        is_passed = True
        fail_reasons = []

        # Check expectations
        if expect == "deflect":
            is_deflected = ("I can only" in reply or "deflection" in reply or path_type == "out_of_domain")
            if not is_deflected:
                is_passed = False
                ood_false_negatives += 1
                fail_reasons.append("Expected OOD deflection but got answer.")
        else:
            # Check for False Positive OOD (football query deflected)
            if "I can only assist with" in reply or path_type == "out_of_domain":
                is_passed = False
                ood_false_positives += 1
                fail_reasons.append("False Positive OOD: Real query was wrongly deflected.")

            # Check must_include keywords (case-insensitive and comma-flexible)
            reply_lower = reply.lower().replace(",", "")
            for kw in must_include:
                kw_clean = kw.lower().replace(",", "")
                if kw_clean not in reply_lower:
                    is_passed = False
                    fail_reasons.append(f"Missing required keyword: '{kw}'")

        if is_passed:
            passed += 1
            category_stats[category]["passed"] += 1
            status_icon = "✓"
        else:
            failed += 1
            status_icon = "✗"

        print(f"[{status_icon}] #{case_id:02d} [{category.upper():<8}] {q[:45]:<47} | {duration_ms:>7.1f}ms | {path_type}")
        if not is_passed:
            print(f"    ↳ Failed: {', '.join(fail_reasons)}")
            print(f"    ↳ Reply: {reply[:120]}...")

    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0
    pass_rate = round((passed / len(test_cases)) * 100, 1)

    print("\n" + "=" * 55)
    print("📈 EVALUATION RESULTS SUMMARY")
    print("=" * 55)
    print(f"Total Cases:            {len(test_cases)}")
    print(f"Passed:                 {passed}")
    print(f"Failed:                 {failed}")
    print(f"Overall Pass Rate:      {pass_rate}%")
    print(f"Average Latency:        {avg_latency} ms")
    print(f"OOD False Positives:    {ood_false_positives} (Valid questions wrongly deflected)")
    print(f"OOD False Negatives:    {ood_false_negatives} (OOD questions that leaked through)")
    print("-" * 55)
    print("Category Breakdown:")
    for cat, stats in category_stats.items():
        c_rate = round((stats["passed"] / stats["total"]) * 100, 1)
        c_avg = round(sum(stats["latencies"]) / len(stats["latencies"]), 1) if stats["latencies"] else 0
        print(f"  • {cat.capitalize():<10} {stats['passed']}/{stats['total']} ({c_rate:>5.1f}%) | Avg Latency: {c_avg:>6.1f} ms")
    print("=" * 55 + "\n")

    return 0 if pass_rate >= 90 else 1


if __name__ == "__main__":
    sys.exit(run_benchmark())

