"""
scripts/terminal_chat.py
────────────────────────
Interactive Terminal Benchmark & Chatbot for Apex Auto Recondition Nepal.
Enables instant conversational testing with real-time performance telemetry:
1. Turn execution latency and token generation speed (t/s).
2. Latency waterfall across Memory MESH, Router, FAISS/NumPy Vector DB, and Llama 3.2.
3. Grounded knowledge chunks with Cosine distances and confidence scores.
4. Currency and domain accuracy validation.
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, List
from pathlib import Path

# Add project root and src to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from stage3_brain.orchestrator import orchestrator
from stage2_memory.session import session_buffer
from stage2_memory.memory_service import memory_service
from stage2_memory.retriever import retriever

# ANSI Color codes for clean terminal display
BOLD = "\033[1m"
GREEN = "\033[32m"
BLUE = "\033[34m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
GRAY = "\033[90m"
RESET = "\033[0m"


def print_banner(user_id: str):
    print("\n" + "=" * 70)
    print(f"{BOLD}{CYAN}🚗 APEX AUTO RECONDITION NEPAL — INTERACTIVE TERMINAL BENCHMARK{RESET}")
    print(f"{GRAY}Grounded in Nepali Inventory (Santro, i10, Magnite, Swift, Alto, Creta){RESET}")
    print(f"{GRAY}Hardware Profile: RTX 3050 Laptop (4GB VRAM) · Llama 3.2 3B · nomic-embed-text{RESET}")
    print("=" * 70)
    print(f"Active Customer Session: {BOLD}{user_id}{RESET}")
    print(f"Commands: {YELLOW}/bench{RESET} (run 5-turn speed test) | {YELLOW}/memory{RESET} (view SQLite profile) | {YELLOW}/clear{RESET} | {YELLOW}/quit{RESET}")
    print("-" * 70 + "\n")


def print_telemetry_card(result: Dict[str, Any], query: str):
    metrics = result.get("metrics", {})
    total_ms = metrics.get("total_ms", 0.0)
    path_type = metrics.get("path_type", "unknown")
    tok_info = metrics.get("tokens", {})
    tps = tok_info.get("tokens_per_sec") or metrics.get("tokens_per_sec", 0.0)
    tokens = tok_info.get("completion_tokens") or metrics.get("completion_tokens", 0)
    stages = metrics.get("stages", {})
    chunks = metrics.get("rag", {}).get("chunks") or result.get("chunks", [])

    # Format latency display
    sec_display = f"({total_ms / 1000.0:.2f}s)" if total_ms >= 1000 else ""
    latency_color = GREEN if total_ms < 500 else (YELLOW if total_ms < 4000 else RED)

    print("\n" + "─" * 70)
    print(f"{BOLD}📊 REAL-TIME TURN PERFORMANCE & TELEMETRY{RESET}")
    print("─" * 70)
    print(f"• Total Latency       : {latency_color}{BOLD}{total_ms} ms {sec_display}{RESET}")
    print(f"• Execution Path      : {CYAN}{BOLD}{path_type}{RESET}")

    if path_type == "rag_llm":
        print(f"• Llama 3.2 Speed     : {GREEN}{BOLD}{tps} tokens/sec{RESET} ({tokens} completion tokens)")
    elif path_type == "fast_anchor":
        print(f"• Fast FAQ Overhead   : {GREEN}{BOLD}0 ms LLM time{RESET} (Instant Canonical Route)")
    elif path_type == "memory_store":
        print(f"• SQLite Write Speed  : {GREEN}{BOLD}Instant Directive Intercept{RESET}")

    # Latency Waterfall Breakdown
    print(f"\n{BOLD}⏱️ Stage Latency Breakdown:{RESET}")
    for stage_name, duration in stages.items():
        pct = (duration / total_ms * 100) if total_ms > 0 else 0
        bar = "█" * max(1, int(pct / 5))
        label_map = {
            "memory_ms": "Memory MESH Interceptor",
            "router_ms": "Intent Routing & Anchors",
            "rewrite_ms": "Dialogue Context Rewrite",
            "rag_retrieval_ms": "Vector DB Retrieval (FAISS/NumPy)",
            "prompt_build_ms": "Prompt Assembly & Citations",
            "llm_inference_ms": "Llama 3.2 3B Generation"
        }
        human_name = label_map.get(stage_name, stage_name)
        print(f"  {GRAY}├─{RESET} {human_name:<34} : {duration:>7.2f} ms ({pct:>4.1f}%) {GRAY}{bar}{RESET}")

    # Retrieved Grounded Chunks with Cosine Distance & Confidence
    if chunks:
        print(f"\n{BOLD}📚 Retrieved Knowledge Chunks ({len(chunks)} sources):{RESET}")
        for i, ch in enumerate(chunks, 1):
            src = ch.get("source", "Knowledge Base")
            title = ch.get("title", f"Chunk {i}")
            dist = ch.get("distance", 0.0)
            conf = ch.get("confidence", "high").upper()
            conf_color = GREEN if conf == "HIGH" else (YELLOW if conf == "MEDIUM" else RED)
            print(f"  {GRAY}[{i}]{RESET} {BOLD}{src}{RESET} — {title}")
            print(f"      ↳ Cosine Distance: {BOLD}{dist:.4f}{RESET} | Confidence: {conf_color}{conf}{RESET}")
    elif path_type == "out_of_domain":
        print(f"\n{YELLOW}🛡️ Out-of-Domain Guard Triggered: Non-dealership query safely deflected.{RESET}")

    # Accuracy / Rule Checks
    reply = result.get("reply", "")
    has_lakhs = "lakh" in reply.lower()
    has_dollars = "$" in reply or "usd" in reply.lower()
    currency_status = f"{GREEN}✓ Verified Nepali Lakhs (NPR){RESET}" if (has_lakhs and not has_dollars) else (f"{RED}⚠️ Foreign Currency Detected!{RESET}" if has_dollars else f"{GRAY}• N/A{RESET}")
    print(f"\n{BOLD}🔍 Accuracy Validation:{RESET} {currency_status}")
    print("─" * 70)


def run_benchmark_suite(user_id: str):
    """Executes a 5-query benchmark testing all architectural paths."""
    queries = [
        ("Fast FAQ (Warranty)", "What warranty and free servicing do you offer?"),
        ("Memory Store", "Remember that my name is Rohan and my budget is 25 Lakhs"),
        ("Vehicle RAG (Santro & i10)", "What Hyundai cars do you have in stock and what are their prices?"),
        ("Multi-turn Follow-up", "What is the mileage and lot number of the Santro?"),
        ("Out-of-Domain Deflection", "How do I make chocolate cake?")
    ]

    print("\n" + "=" * 70)
    print(f"{BOLD}{YELLOW}⚡ RUNNING 5-TURN SPEED & ACCURACY BENCHMARK SUITE{RESET}")
    print("=" * 70)

    total_time = 0.0
    results_summary = []

    for label, q in queries:
        print(f"\n{BOLD}Query [{label}]:{RESET} \"{q}\"")
        res = orchestrator.process(user_id=user_id, query=q, channel="chat")
        total_ms = res.get("metrics", {}).get("total_ms", 0.0)
        path = res.get("metrics", {}).get("path_type", "")
        tps = res.get("metrics", {}).get("tokens_per_sec", 0.0)
        total_time += total_ms

        print(f"{BOLD}Bot Reply:{RESET} {res['reply'][:140]}...")
        print(f"↳ Path: {CYAN}{path}{RESET} | Latency: {GREEN}{total_ms:.1f} ms{RESET} | Speed: {tps} t/s")
        results_summary.append((label, path, total_ms, tps))

    print("\n" + "=" * 70)
    print(f"{BOLD}🏆 BENCHMARK SUMMARY REPORT{RESET}")
    print("=" * 70)
    print(f"{'Test Category':<28} {'Path Type':<16} {'Latency':<12} {'Speed'}")
    print("-" * 70)
    for cat, path, lat, tps in results_summary:
        print(f"{cat:<28} {path:<16} {lat:>7.1f} ms     {tps:>5.1f} t/s")
    print("-" * 70)
    print(f"{BOLD}Total Benchmark Time: {total_time:.1f} ms ({total_time / 1000.0:.2f}s) across 5 turns{RESET}")
    print("=" * 70 + "\n")


def interactive_loop(user_id: str):
    print_banner(user_id)

    while True:
        try:
            query = input(f"{BOLD}{BLUE}Customer ({user_id}) > {RESET}").strip()
            if not query:
                continue

            if query.lower() in ("exit", "quit", "/quit", "/exit"):
                print(f"\n{GREEN}Exiting. Have a great day!{RESET}\n")
                break

            if query.lower() == "/clear":
                session_buffer.clear(user_id)
                print(f"\n{YELLOW}✓ Cleared conversation history for {user_id}.{RESET}\n")
                continue

            if query.lower() == "/bench":
                run_benchmark_suite(user_id)
                continue

            if query.lower() == "/memory":
                mems = memory_service.get_memories(user_id)
                print(f"\n{BOLD}🧠 Saved SQLite Memories for {user_id} ({len(mems)} items):{RESET}")
                if not mems:
                    print(f"  {GRAY}(No memories saved yet. Try saying: 'Remember that my budget is 20 Lakhs'){RESET}")
                for m in mems:
                    print(f"  • [{m.category.upper()}] {m.memory_text}")
                print()
                continue

            # Process turn through Stage 2 & Stage 3 pipeline
            result = orchestrator.process(user_id=user_id, query=query, channel="chat")

            # Display response
            print(f"\n{BOLD}{GREEN}Apex AI Assistant > {RESET}{result.get('reply', '')}")

            # Display rich telemetry card
            print_telemetry_card(result, query)
            print()

        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{GREEN}Session ended.{RESET}\n")
            break


def main():
    parser = argparse.ArgumentParser(description="Apex Auto Nepal Terminal Chat & Benchmark Runner")
    parser.add_argument("--user", default="test_customer", help="Customer user ID / Phone number for session")
    parser.add_argument("--test-query", type=str, default=None, help="Run a single test query and exit")
    parser.add_argument("--bench", action="store_true", help="Run automated benchmark suite and exit")
    args = parser.parse_args()

    if args.bench:
        run_benchmark_suite(args.user)
    elif args.test_query:
        print_banner(args.user)
        print(f"{BOLD}{BLUE}Customer ({args.user}) > {RESET}{args.test_query}")
        result = orchestrator.process(user_id=args.user, query=args.test_query, channel="chat")
        print(f"\n{BOLD}{GREEN}Apex AI Assistant > {RESET}{result.get('reply', '')}")
        print_telemetry_card(result, args.test_query)
    else:
        interactive_loop(args.user)


if __name__ == "__main__":
    main()
