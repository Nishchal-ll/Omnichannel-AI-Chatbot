"""
src/agent/pipeline.py
─────────────────────
Main Pipeline Orchestrator for Naasa Securities (Broker 58) Customer Support Agent.
Replaces the legacy CrewAI pipeline with a fast, zero-overhead architecture:
1. Message Persistence (data/messages.json)
2. User Contact & Profile Extraction
3. Intent Routing & 0ms Fast Anchors
4. Out-of-Domain Guard Rejection
5. Multi-Turn Session Context & Query Rewriting
6. Section-Aware ChromaDB RAG Retrieval
7. Grounded Ollama Gemma 4 Inference
8. Channel Dispatch & Status Updating
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from channels.base import Message
from channels.registry import get_channel
from knowledge import get_relevant_knowledge_with_metadata
import store

from .session import session_buffer
from .router import IntentRouter
from .prompt import build_messages, OUT_OF_CONTEXT_DEFLECTION
from .ollama_client import OllamaClient

log = logging.getLogger(__name__)


class Pipeline:
    """
    Stateless, multi-thread safe pipeline.
    Invoked by Webhook server, Email poller, Streamlit dashboard, or CLI tests.
    Equipped with granular execution telemetry and stage latency profiling.
    """

    def __init__(self):
        self.client = OllamaClient()

    def process(self, message: Message, record_id: str = None) -> str:
        """Process an incoming message and return the generated answer with execution telemetry."""
        t_start = time.perf_counter()
        sender = message.sender
        query = message.body.strip()

        # 1. Audit / Store Message
        if record_id is None:
            record = store.add_message(
                channel=message.channel,
                sender=sender,
                body=query,
                subject=message.subject,
            )
            record_id = record["id"]

        print(f"[Pipeline] Processing [{message.channel}] from {sender}: {query[:60]}...")

        # 2. Extract and update contact / profile info (client code, phone, email)
        t_route_start = time.perf_counter()
        contact_info = IntentRouter.extract_contact_info(query)
        if contact_info:
            session_buffer.update_user_contact(sender, **contact_info)

        user_contact = session_buffer.get_user_contact(sender)

        # 3. Check 0ms Fast Paths (Greetings, Signoffs)
        fast_reply = IntentRouter.check_fast_path(query)
        if fast_reply:
            t_routing_ms = round((time.perf_counter() - t_route_start) * 1000, 1)
            print("[Pipeline] ✓ Fast-path anchor triggered")
            session_buffer.add_turn(sender, query, fast_reply)

            t_disp_start = time.perf_counter()
            status = self._dispatch_reply(message, fast_reply)
            t_disp_ms = round((time.perf_counter() - t_disp_start) * 1000, 1)
            t_total_ms = round((time.perf_counter() - t_start) * 1000, 1)

            metrics = {
                "total_ms": t_total_ms,
                "path_type": "fast_path",
                "stages": {
                    "routing_ms": t_routing_ms,
                    "channel_dispatch_ms": t_disp_ms,
                },
                "tokens": {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "tokens_per_sec": 0.0},
                "rag": {"chunks_found": 0, "sources": []},
                "model": "fast_anchor",
            }
            store.set_reply(record_id, fast_reply, status, metrics=metrics)
            print(f"[Pipeline] ⏱️ Fast-path done in {t_total_ms}ms")
            return fast_reply

        # 4. Out-of-Domain Rejection Guard
        if IntentRouter.is_out_of_domain(query):
            t_routing_ms = round((time.perf_counter() - t_route_start) * 1000, 1)
            print("[Pipeline] ✗ Out-of-domain query detected — deflecting")
            deflection = OUT_OF_CONTEXT_DEFLECTION
            session_buffer.add_turn(sender, query, deflection)

            t_disp_start = time.perf_counter()
            status = self._dispatch_reply(message, deflection)
            t_disp_ms = round((time.perf_counter() - t_disp_start) * 1000, 1)
            t_total_ms = round((time.perf_counter() - t_start) * 1000, 1)

            metrics = {
                "total_ms": t_total_ms,
                "path_type": "out_of_domain",
                "stages": {
                    "routing_ms": t_routing_ms,
                    "channel_dispatch_ms": t_disp_ms,
                },
                "tokens": {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "tokens_per_sec": 0.0},
                "rag": {"chunks_found": 0, "sources": []},
                "model": "out_of_domain_guard",
            }
            store.set_reply(record_id, deflection, status, metrics=metrics)
            print(f"[Pipeline] ⏱️ Out-of-domain deflection done in {t_total_ms}ms")
            return deflection

        t_routing_ms = round((time.perf_counter() - t_route_start) * 1000, 1)

        # 5. Multi-Turn Session History & Query Rewriting
        t_rewrite_start = time.perf_counter()
        chat_history = session_buffer.get_history(sender)
        standalone_query = IntentRouter.rewrite_query(query, chat_history)
        if standalone_query != query:
            print(f"[Pipeline] Query rewritten: \"{query}\" → \"{standalone_query}\"")
        t_rewrite_ms = round((time.perf_counter() - t_rewrite_start) * 1000, 1)

        # 6. Retrieve Domain Knowledge from ChromaDB
        t_rag_start = time.perf_counter()
        kb_context, chunks = get_relevant_knowledge_with_metadata(standalone_query, top_k=3, max_distance=0.85)
        t_rag_ms = round((time.perf_counter() - t_rag_start) * 1000, 1)

        # 7. Build Grounded Prompt
        t_prompt_start = time.perf_counter()
        messages = build_messages(
            user_query=query,
            knowledge_context=kb_context,
            chat_history=chat_history,
            user_contact=user_contact
        )
        t_prompt_ms = round((time.perf_counter() - t_prompt_start) * 1000, 1)

        # 8. Ollama Gemma 4 Inference with Telemetry
        reply_text, llm_meta = self.client.chat_complete_with_metadata(messages)
        t_llm_ms = llm_meta.get("inference_ms", 0.0)
        print(f"[Pipeline] Ollama Reply: {reply_text[:80]}...")

        # 9. Update Session Memory
        session_buffer.add_turn(sender, query, reply_text)

        # 10. Dispatch to Channel & Persist
        t_disp_start = time.perf_counter()
        status = self._dispatch_reply(message, reply_text)
        t_disp_ms = round((time.perf_counter() - t_disp_start) * 1000, 1)
        t_total_ms = round((time.perf_counter() - t_start) * 1000, 1)

        metrics = {
            "total_ms": t_total_ms,
            "path_type": "rag_llm",
            "stages": {
                "routing_ms": t_routing_ms,
                "rewrite_ms": t_rewrite_ms,
                "rag_retrieval_ms": t_rag_ms,
                "prompt_build_ms": t_prompt_ms,
                "llm_inference_ms": t_llm_ms,
                "channel_dispatch_ms": t_disp_ms,
            },
            "tokens": {
                "prompt_tokens": llm_meta.get("prompt_tokens", 0),
                "completion_tokens": llm_meta.get("completion_tokens", 0),
                "total_tokens": llm_meta.get("total_tokens", 0),
                "tokens_per_sec": llm_meta.get("tokens_per_sec", 0.0),
                "had_reasoning": llm_meta.get("had_reasoning", False),
                "reasoning": llm_meta.get("reasoning", ""),
            },
            "rag": {
                "chunks_found": len(chunks),
                "sources": [c.get("title", "") for c in chunks],
                "top_distance": round(chunks[0]["distance"], 4) if chunks else None,
                "chunks": [
                    {
                        "title": c.get("title", ""),
                        "source": c.get("source", ""),
                        "distance": round(c.get("distance", 0.0), 4),
                        "snippet": c.get("text", "")[:250],
                    }
                    for c in chunks
                ],
            },
            "model": llm_meta.get("model", self.client.model),
            "rewritten_query": standalone_query if standalone_query != query else None,
        }

        store.set_reply(record_id, reply_text, status, metrics=metrics)
        print(f"[Pipeline] ⏱️ Done in {t_total_ms}ms (LLM: {t_llm_ms}ms, RAG: {t_rag_ms}ms, Send: {t_disp_ms}ms) | {llm_meta.get('tokens_per_sec', 0)} tok/s")
        return reply_text

    def _dispatch_reply(self, message: Message, reply_text: str) -> str:
        """Sends reply back via channel (if configured) and returns status."""
        try:
            channel = get_channel(message.channel)
            ok = channel.send(message, reply_text)
            return "replied" if ok else "saved_local"
        except KeyError:
            # Local test or unregistered channel
            return "saved_local"
        except Exception as e:
            print(f"[Pipeline] Channel send warning: {e}")
            return "saved_local"
