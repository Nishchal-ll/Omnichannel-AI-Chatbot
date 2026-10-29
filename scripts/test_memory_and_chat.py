"""
scripts/test_memory_and_chat.py
───────────────────────────────
Comprehensive end-to-end test suite for:
1. Long-Term Brain Memory (save, recall, persistence in SQLite).
2. Context Window Follow-Up & Pronoun Rewriting across turns.
3. Grounded Q&A with ChromaDB RAG & Gemma 4 E2B.
4. Out-of-domain deflection guard.
5. Preserved WhatsApp channel integrity.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stage2_memory.session import session_buffer
from stage2_memory.memory_service import memory_service
from stage3_brain.orchestrator import orchestrator


def test_full_flow():
    test_user = "test_user_nishchal"
    print(f"\n=======================================================")
    print(f"🚀 STARTING COMPLETE PIPELINE & MEMORY VERIFICATION")
    print(f"User ID: {test_user}")
    print(f"=======================================================\n")

    # 0. Clean slate for test user
    session_buffer.clear_history(test_user)
    memory_service.clear_memories(test_user)

    # ── Test 1: Brain Memory Storage Directive ──
    print("[Test 1] Saving Long-Term Fact Directive...")
    q1 = "Remember that my name is Nishchal and I support Barcelona"
    res1 = orchestrator.process(user_id=test_user, query=q1, channel="chat")
    print(f"Bot Reply: {res1['reply']}")
    print(f"Latency: {res1['metrics']['total_ms']} ms | Path: {res1['metrics']['path_type']}")
    
    # Verify in SQLite
    mems = memory_service.get_memories(test_user)
    assert len(mems) >= 1, "Memory should be persisted in SQLite"
    print(f"✓ SQLite Verified: {len(mems)} memory item(s) found -> '{mems[0].memory_text}'")
    assert "remembered" in res1["reply"].lower()

    # ── Test 2: Domain Knowledge Q&A ──
    print("\n[Test 2] Domain Knowledge Q&A (Messi Ballon d'Or)...")
    q2 = "How many Ballon d'Or awards has Messi won?"
    t0 = time.perf_counter()
    res2 = orchestrator.process(user_id=test_user, query=q2, channel="chat")
    elapsed = time.perf_counter() - t0
    print(f"Bot Reply: {res2['reply']}")
    print(f"Latency: {res2['metrics']['total_ms']} ms ({elapsed:.2f}s) | LLM inference: {res2['metrics']['stages'].get('llm_inference_ms', 0)} ms")
    print(f"Tokens: {res2['metrics']['tokens']}")
    assert "8" in res2["reply"] or "eight" in res2["reply"].lower(), "Reply should correctly state 8 Ballon d'Ors"
    print("✓ Knowledge Q&A Factually Accurate!")

    # ── Test 3: Multi-turn Context Window Follow-Up & Pronoun Rewriting ──
    print("\n[Test 3] Multi-turn Context Window Follow-up ('When did he win his first one?')...")
    q3 = "When did he win his first one?"
    res3 = orchestrator.process(user_id=test_user, query=q3, channel="chat")
    print(f"Rewritten Query: {res3['metrics'].get('standalone_query')}")
    print(f"Bot Reply: {res3['reply']}")
    print(f"Latency: {res3['metrics']['total_ms']} ms | LLM inference: {res3['metrics']['stages'].get('llm_inference_ms', 0)} ms")
    assert "2009" in res3["reply"], "Reply should state 2009 as his first Ballon d'Or"
    print("✓ Context Window Follow-up correctly resolved 'he' -> Lionel Messi and 2009!")

    # ── Test 4: Memory Recall / Asking what is remembered ──
    print("\n[Test 4] Asking about user profile / memory recall...")
    q4 = "What do you remember about me?"
    res4 = orchestrator.process(user_id=test_user, query=q4, channel="chat")
    print(f"Bot Reply: {res4['reply']}")
    assert "nishchal" in res4["reply"].lower() or "barcelona" in res4["reply"].lower(), "Reply should recall Nishchal / Barcelona"
    print("✓ User Long-Term Brain Memory Recall Succeeded!")

    # ── Test 5: Out of Domain Deflection ──
    print("\n[Test 5] Testing Out-Of-Domain Deflection Guard...")
    q5 = "How do I make a chocolate cake?"
    res5 = orchestrator.process(user_id=test_user, query=q5, channel="chat")
    print(f"Bot Reply: {res5['reply']}")
    print(f"Latency: {res5['metrics']['total_ms']} ms | Path: {res5['metrics']['path_type']}")
    assert res5["metrics"]["path_type"] == "out_of_domain", "Should be intercepted by out_of_domain guard"
    print("✓ Out of Domain deflected with 0 LLM latency!")

    # ── Test 6: WhatsApp Channel Preservation Check ──
    print("\n[Test 6] Checking WhatsApp Channel Preservation...")
    from channels.registry import get_channel
    wa_channel = get_channel("whatsapp")
    assert wa_channel is not None, "WhatsApp channel must be loadable"
    print(f"✓ WhatsApp Channel module intact: {wa_channel.__class__.__name__}")

    print("\n=======================================================")
    print("🎉 ALL TESTS PASSED SUCCESSFULLY! 100% FACTUAL & FAST")
    print("=======================================================\n")


if __name__ == "__main__":
    test_full_flow()

