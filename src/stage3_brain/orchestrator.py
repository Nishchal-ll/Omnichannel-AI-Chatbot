"""
src/stage3_brain/orchestrator.py
────────────────────────────────
PipelineOrchestrator: Connects Stage 2 (Memory MESH) and Stage 3 (Brain).
Supports direct Web Chat and preserved WhatsApp channel dispatch.
"""

import time
import os
import sys
from typing import Dict, Any, Optional

from stage2_memory.session import session_buffer
from stage2_memory.memory_service import memory_service
from stage2_memory.faq_service import faq_service
from stage2_memory.router import IntentRouter, normalize_query
from stage2_memory.retriever import retriever
from stage3_brain.prompt_builder import prompt_builder, OUT_OF_CONTEXT_DEFLECTION
from stage3_brain.brain_engine import brain_engine
import threading


class PipelineOrchestrator:
    """Unified Orchestrator across Stage 2 Memory and Stage 3 Brain."""

    def __init__(self):
        self.brain = brain_engine
        # Item 6: Concurrency serialization for CPU LLM execution
        self._llm_lock = threading.BoundedSemaphore(1)

    def process(
        self,
        user_id: str,
        query: str,
        channel: str = "chat",
        message_obj: Any = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end conversation turn:
        1. Memory directive interception (save / recall / forget)
        2. Fast-path anchor check (0 ms)
        3. Out-of-domain deflection guard
        4. Multi-turn dialogue history & pronoun query rewriting
        5. ChromaDB domain knowledge RAG retrieval
        6. Grounded Gemma 4 E2B generation with persistent user memories
        7. History & memory persistence in SQLite
        8. Optional channel dispatch (WhatsApp)
        """
        t_start = time.perf_counter()
        clean_query = normalize_query(query)
        stage_timings: Dict[str, float] = {}

        # ── 1. Check Memory Directives (Save / Recall / Forget) ──────────────
        t0 = time.perf_counter()

        # A. Explicit Save: 'Remember that...' / 'My name is...'
        mem_directive = memory_service.check_memory_directive(clean_query)
        if mem_directive:
            fact, category = mem_directive
            item = memory_service.add_memory(user_id, fact, category)
            reply = f"✓ I have remembered that: {fact}."
            session_buffer.add_turn(user_id, query, reply)
            stage_timings["memory_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            total_ms = round((time.perf_counter() - t_start) * 1000, 1)

            if channel == "whatsapp" and message_obj:
                self._dispatch_whatsapp(message_obj, reply)

            return {
                "reply": reply,
                "metrics": {
                    "total_ms": total_ms,
                    "path_type": "memory_store",
                    "stages": stage_timings,
                    "tokens": {"completion_tokens": len(reply.split()), "tokens_per_sec": 0.0},
                    "rag": {"chunks_found": 0, "chunks": []},
                    "model": "memory_service",
                },
                "user_memories": [m.to_dict() for m in memory_service.get_memories(user_id)],
            }

        # B. Recall: 'What do you remember about me?' / 'What is my name?' / 'Where do I live?'
        if IntentRouter.check_memory_recall(clean_query):
            import re
            memories = memory_service.get_memories(user_id)
            clean_lower = clean_query.lower()

            if not memories:
                reply = "I don't have any saved profile information for you yet. You can tell me facts like 'My name is...' or 'I live in...'!"
            elif "name" in clean_lower:
                name_mems = [m for m in memories if m.category == "identity" or "name is" in m.memory_text.lower()]
                if name_mems:
                    name_val = re.sub(r"User's name is\s*", "", name_mems[-1].memory_text, flags=re.IGNORECASE).strip()
                    reply = f"Your name is {name_val}."
                else:
                    items_str = "\n".join(f"• {m.memory_text}" for m in memories)
                    reply = f"I don't have your name recorded yet, but here is what I know about you:\n{items_str}"
            elif "live" in clean_lower or "where" in clean_lower or "from" in clean_lower:
                loc_mems = [m for m in memories if m.category == "location" or "lives in" in m.memory_text.lower() or "from" in m.memory_text.lower()]
                if loc_mems:
                    loc_val = re.sub(r"User lives in\s*", "", loc_mems[-1].memory_text, flags=re.IGNORECASE).strip()
                    reply = f"You live in {loc_val}."
                else:
                    items_str = "\n".join(f"• {m.memory_text}" for m in memories)
                    reply = f"I don't have your location recorded yet, but here is what I know about you:\n{items_str}"
            elif "support" in clean_lower or "favorite" in clean_lower:
                pref_mems = [m for m in memories if m.category == "preference" or "supports" in m.memory_text.lower()]
                if pref_mems:
                    reply = f"According to your saved profile: {pref_mems[-1].memory_text}."
                else:
                    items_str = "\n".join(f"• {m.memory_text}" for m in memories)
                    reply = f"Here is what I remember about you:\n{items_str}"
            else:
                items_str = "\n".join(f"• {m.memory_text}" for m in memories)
                reply = f"Here is what I remember about you:\n{items_str}"

            session_buffer.add_turn(user_id, query, reply)
            stage_timings["memory_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            total_ms = round((time.perf_counter() - t_start) * 1000, 1)

            if channel == "whatsapp" and message_obj:
                self._dispatch_whatsapp(message_obj, reply)

            return {
                "reply": reply,
                "metrics": {
                    "total_ms": total_ms,
                    "path_type": "memory_recall",
                    "stages": stage_timings,
                    "tokens": {"completion_tokens": len(reply.split()), "tokens_per_sec": 0.0},
                    "rag": {"chunks_found": 0, "chunks": []},
                    "model": "memory_service",
                },
                "user_memories": [m.to_dict() for m in memories],
            }

        # C. Forget: 'Forget me' / 'Clear my data' (Item 4)
        if IntentRouter.check_memory_clear(clean_query):
            session_buffer.delete_all_user_data(user_id)
            reply = "I have completely erased your profile, saved memories, and dialogue history."
            stage_timings["memory_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            total_ms = round((time.perf_counter() - t_start) * 1000, 1)

            if channel == "whatsapp" and message_obj:
                self._dispatch_whatsapp(message_obj, reply)

            return {
                "reply": reply,
                "metrics": {
                    "total_ms": total_ms,
                    "path_type": "memory_clear",
                    "stages": stage_timings,
                    "tokens": {"completion_tokens": len(reply.split()), "tokens_per_sec": 0.0},
                    "rag": {"chunks_found": 0, "chunks": []},
                    "model": "memory_service",
                },
                "user_memories": [],
            }

        stage_timings["memory_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── 2. Fast-Path Anchors & SQLite FAQ (Item 2: 0 ms) ──────────────────
        t0 = time.perf_counter()
        faq_reply = faq_service.check_faq(clean_query)
        fast_reply = faq_reply if faq_reply else IntentRouter.check_fast_path(clean_query)
        stage_timings["routing_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        if fast_reply:
            session_buffer.add_turn(user_id, query, fast_reply)
            total_ms = round((time.perf_counter() - t_start) * 1000, 1)

            if channel == "whatsapp" and message_obj:
                self._dispatch_whatsapp(message_obj, fast_reply)

            return {
                "reply": fast_reply,
                "metrics": {
                    "total_ms": total_ms,
                    "path_type": "fast_anchor",
                    "stages": stage_timings,
                    "tokens": {"completion_tokens": len(fast_reply.split()), "tokens_per_sec": 0.0},
                    "rag": {"chunks_found": 0, "chunks": []},
                    "model": "fast_anchor",
                },
                "user_memories": [m.to_dict() for m in memory_service.get_memories(user_id)],
            }

        # ── 3. Out-Of-Domain Deflection Guard ────────────────────────────────
        if IntentRouter.is_out_of_domain(clean_query):
            reply = OUT_OF_CONTEXT_DEFLECTION
            session_buffer.add_turn(user_id, query, reply)
            total_ms = round((time.perf_counter() - t_start) * 1000, 1)

            if channel == "whatsapp" and message_obj:
                self._dispatch_whatsapp(message_obj, reply)

            return {
                "reply": reply,
                "metrics": {
                    "total_ms": total_ms,
                    "path_type": "out_of_domain",
                    "stages": stage_timings,
                    "tokens": {"completion_tokens": len(reply.split()), "tokens_per_sec": 0.0},
                    "rag": {"chunks_found": 0, "chunks": []},
                    "model": "out_of_domain_guard",
                },
                "user_memories": [m.to_dict() for m in memory_service.get_memories(user_id)],
            }

        # ── 4. Short-Term History & Query Rewriter ───────────────────────────
        t0 = time.perf_counter()
        history = session_buffer.get_history(user_id)
        standalone_query = IntentRouter.rewrite_query(clean_query, history)
        stage_timings["rewrite_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── 5. ChromaDB Domain Knowledge RAG ─────────────────────────────────
        t0 = time.perf_counter()
        kb_context, chunks = retriever.retrieve_with_metadata(standalone_query, top_k=4)
        stage_timings["rag_retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── Semantic Out-of-Domain Retrieval Gate (Item 1) ───────────────────
        from stage2_memory.retriever import OOD_DISTANCE
        best_distance = chunks[0]["distance"] if chunks else 1.0
        has_domain_kw = any(kw in standalone_query.lower() for kw in ["messi", "lionel", "leo", "barca", "barcelona", "psg", "inter miami", "argentina"])

        if not chunks or (best_distance > OOD_DISTANCE and not has_domain_kw) or best_distance > 0.78:
            reply = OUT_OF_CONTEXT_DEFLECTION
            session_buffer.add_turn(user_id, query, reply)
            total_ms = round((time.perf_counter() - t_start) * 1000, 1)

            if channel == "whatsapp" and message_obj:
                self._dispatch_whatsapp(message_obj, reply)

            return {
                "reply": reply,
                "metrics": {
                    "total_ms": total_ms,
                    "path_type": "out_of_domain",
                    "stages": stage_timings,
                    "tokens": {"completion_tokens": len(reply.split()), "tokens_per_sec": 0.0},
                    "rag": {"chunks_found": len(chunks), "best_distance": best_distance, "chunks": chunks},
                    "model": "semantic_ood_gate",
                    "standalone_query": standalone_query if standalone_query != query else None,
                },
                "user_memories": [m.to_dict() for m in memory_service.get_memories(user_id)],
            }

        # ── 6. Fetch User Long-Term Memories ─────────────────────────────────
        user_memories = memory_service.get_memories(user_id)

        # ── 7. Build Grounded Messages Array ─────────────────────────────────
        t0 = time.perf_counter()
        messages = prompt_builder.build_messages(
            user_query=query,
            knowledge_context=kb_context,
            chat_history=history,
            user_memories=user_memories,
        )
        stage_timings["prompt_build_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── 8. Gemma 4 E2B Inference (Item 6: Serialized with 25s timeout) ──
        acquired = self._llm_lock.acquire(timeout=25.0)
        if not acquired:
            reply_text = "I am currently handling another complex query. Please try your question again in a moment."
            llm_meta = {"inference_ms": 25000.0, "completion_tokens": len(reply_text.split()), "tokens_per_sec": 0.0}
        else:
            try:
                reply_text, llm_meta = self.brain.generate(messages)
            finally:
                self._llm_lock.release()
        stage_timings["llm_inference_ms"] = llm_meta.get("inference_ms", 0.0)

        # ── 9. Commit Turn to Session Buffer (SQLite) ────────────────────────
        session_buffer.add_turn(user_id, query, reply_text)

        # ── 10. Channel Dispatch (if WhatsApp) ───────────────────────────────
        if channel == "whatsapp" and message_obj:
            t0 = time.perf_counter()
            self._dispatch_whatsapp(message_obj, reply_text)
            stage_timings["dispatch_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        total_ms = round((time.perf_counter() - t_start) * 1000, 1)

        return {
            "reply": reply_text,
            "metrics": {
                "total_ms": total_ms,
                "path_type": "rag_llm",
                "stages": stage_timings,
                "tokens": {
                    "prompt_tokens": llm_meta.get("prompt_tokens", 0),
                    "completion_tokens": llm_meta.get("completion_tokens", 0),
                    "total_tokens": llm_meta.get("total_tokens", 0),
                    "tokens_per_sec": llm_meta.get("tokens_per_sec", 0.0),
                },
                "rag": {
                    "chunks_found": len(chunks),
                    "chunks": chunks,
                },
                "model": llm_meta.get("model", self.brain.model),
                "standalone_query": standalone_query if standalone_query != query else None,
            },
            "user_memories": [m.to_dict() for m in user_memories],
        }

    def _dispatch_whatsapp(self, message_obj: Any, reply_text: str):
        """Sends reply back via WhatsApp channel if configured."""
        try:
            from channels.registry import get_channel
            ch = get_channel("whatsapp")
            ch.send(message_obj, reply_text)
        except Exception as e:
            print(f"[Orchestrator] WhatsApp dispatch warning: {e}")


orchestrator = PipelineOrchestrator()

