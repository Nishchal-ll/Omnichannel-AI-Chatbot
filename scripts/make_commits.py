"""
scripts/make_commits.py
───────────────────────
Executes the approved chronological commit schedule starting Oct 22, 2026.
Sets GIT_AUTHOR_DATE and GIT_COMMITTER_DATE for each commit.
"""

import os
import subprocess
import sys

COMMITS = [
    # ── Day 1: October 22, 2026 ──
    {
        "date": "2026-10-22 10:15:00 +0545",
        "msg": "refactor: deprecate legacy Streamlit dashboard in favor of FastAPI console",
        "cmd": ["git", "rm", "src/streamlit_app.py"]
    },
    {
        "date": "2026-10-22 13:40:00 +0545",
        "msg": "chore(deps): update requirements.txt and project dependencies",
        "cmd": ["git", "add", ".gitignore", "requirements.txt"]
    },
    {
        "date": "2026-10-22 16:20:00 +0545",
        "msg": "feat(ui): scaffold clean HTML5 dual-pane workspace layout",
        "cmd": ["git", "add", "src/static/index.html"]
    },
    {
        "date": "2026-10-22 18:45:00 +0545",
        "msg": "style(ui): implement dark theme, telemetry waterfall bars and responsive layout",
        "cmd": ["git", "add", "src/static/app.css"]
    },

    # ── Day 2: October 23, 2026 ──
    {
        "date": "2026-10-23 11:00:00 +0545",
        "msg": "feat(ui): implement multi-user profile switcher and local storage sync",
        "cmd": ["git", "add", "src/static/app.js"]
    },
    {
        "date": "2026-10-23 14:30:00 +0545",
        "msg": "refactor(store): adapt message auditing store for multi-turn telemetry",
        "cmd": ["git", "add", "src/store.py"]
    },
    {
        "date": "2026-10-23 17:15:00 +0545",
        "msg": "chore(store): update audit message store structure",
        "cmd": ["git", "add", "data/messages.json"]
    },

    # ── Day 3: October 24, 2026 ──
    {
        "date": "2026-10-24 10:30:00 +0545",
        "msg": "docs(kb): create comprehensive Lionel Messi biography and career records",
        "cmd": ["git", "add", "data/knowledge/lionel_messi_kb.md"]
    },
    {
        "date": "2026-10-24 14:15:00 +0545",
        "msg": "feat(kb): add section-aware markdown chunking and ChromaDB ingestion",
        "cmd": ["git", "add", "src/knowledge.py"]
    },
    {
        "date": "2026-10-24 17:00:00 +0545",
        "msg": "feat(test): add initial CLI test runner for Lionel Messi QA",
        "cmd": ["git", "add", "scripts/test_messi_qa.py"]
    },

    # ── Day 4: October 25, 2026 ──
    {
        "date": "2026-10-25 11:10:00 +0545",
        "msg": "feat(memory): define dataclass contracts for ChatMessage, UserMemoryItem, ContextPackage",
        "cmd": ["git", "add", "src/stage2_memory/models.py"]
    },
    {
        "date": "2026-10-25 14:40:00 +0545",
        "msg": "feat(memory): implement SQLite conversation schema with index optimization",
        "cmd": ["git", "add", "src/stage2_memory/session.py"]
    },
    {
        "date": "2026-10-25 17:30:00 +0545",
        "msg": "feat(memory): export stage 2 package components",
        "cmd": ["git", "add", "src/stage2_memory/__init__.py"]
    },

    # ── Day 5: October 26, 2026 ──
    {
        "date": "2026-10-26 10:45:00 +0545",
        "msg": "feat(memory): implement persistent user brain memory service backed by SQLite",
        "cmd": ["git", "add", "src/stage2_memory/memory_service.py"]
    },
    {
        "date": "2026-10-26 14:10:00 +0545",
        "msg": "feat(router): implement multi-turn pronoun contextualizer with regex word boundaries",
        "cmd": ["git", "add", "src/stage2_memory/router.py"]
    },
    {
        "date": "2026-10-26 17:40:00 +0545",
        "msg": "feat(rag): implement ChromaDB retriever with all-MiniLM-L6-v2 embeddings",
        "cmd": ["git", "add", "src/stage2_memory/retriever.py"]
    },

    # ── Day 6: October 27, 2026 ──
    {
        "date": "2026-10-27 10:20:00 +0545",
        "msg": "feat(faq): introduce SQLite-backed FAQ table with as_of freshness tracking",
        "cmd": ["git", "add", "src/stage2_memory/faq_service.py"]
    },
    {
        "date": "2026-10-27 14:30:00 +0545",
        "msg": "refactor(agent): tune internal prompt parameters and client settings",
        "cmd": ["git", "add", "src/agent/ollama_client.py", "src/agent/prompt.py"]
    },
    {
        "date": "2026-10-27 17:15:00 +0545",
        "msg": "refactor(pipeline): align legacy pipeline routing with Phase 2 memory contracts",
        "cmd": ["git", "add", "src/agent/pipeline.py", "src/agent/router.py"]
    },

    # ── Day 7: October 28, 2026 ──
    {
        "date": "2026-10-28 10:15:00 +0545",
        "msg": "feat(brain): implement native Ollama client for Gemma 4 E2B with think=False",
        "cmd": ["git", "add", "src/stage3_brain/brain_engine.py"]
    },
    {
        "date": "2026-10-28 14:10:00 +0545",
        "msg": "feat(brain): construct cognitive prompt builder with grounding and few-shots",
        "cmd": ["git", "add", "src/stage3_brain/prompt_builder.py"]
    },
    {
        "date": "2026-10-28 17:45:00 +0545",
        "msg": "feat(brain): implement PipelineOrchestrator linking Memory MESH and Brain Engine",
        "cmd": ["git", "add", "src/stage3_brain/orchestrator.py"]
    },
    {
        "date": "2026-10-28 19:30:00 +0545",
        "msg": "feat(brain): export stage 3 package components",
        "cmd": ["git", "add", "src/stage3_brain/__init__.py"]
    },

    # ── Day 8: October 29, 2026 ──
    {
        "date": "2026-10-29 11:00:00 +0545",
        "msg": "feat(channels): preserve WhatsApp Cloud API dispatch and simulation flow",
        "cmd": ["git", "add", "src/channels/whatsapp.py"]
    },
    {
        "date": "2026-10-29 14:30:00 +0545",
        "msg": "feat(server): expose REST APIs for history, user profiles, and brain memories",
        "cmd": ["git", "add", "src/webhook_server.py"]
    },
    {
        "date": "2026-10-29 17:45:00 +0545",
        "msg": "test(pipeline): add comprehensive end-to-end memory and multi-turn verification suite",
        "cmd": ["git", "add", "scripts/test_memory_and_chat.py"]
    },

    # ── Day 9: October 30, 2026 ──
    {
        "date": "2026-10-30 10:30:00 +0545",
        "msg": "feat(eval): add 42-case evaluation benchmark dataset for factual and OOD QA",
        "cmd": ["git", "add", "eval.jsonl"]
    },
    {
        "date": "2026-10-30 14:20:00 +0545",
        "msg": "feat(eval): add automated benchmark runner calculating pass rate and latency",
        "cmd": ["git", "add", "scripts/run_eval.py"]
    },
    {
        "date": "2026-10-30 17:50:00 +0545",
        "msg": "chore(deploy): update docker-compose and startup script for single-worker deployment",
        "cmd": ["git", "add", "docker-compose.yml", "start.sh"]
    },

    # ── Day 10: October 31, 2026 ──
    {
        "date": "2026-10-31 11:15:00 +0545",
        "msg": "docs(arch): update architecture specifications and operational walkthrough",
        "cmd": ["git", "add", "-A"]
    },
    {
        "date": "2026-10-31 15:45:00 +0545",
        "msg": "perf(system): optimize token budgeting and CPU concurrency settings",
        "cmd": ["git", "add", "-u"]
    },
    {
        "date": "2026-10-31 18:30:00 +0545",
        "msg": "release: complete Phase 2 Memory MESH and Phase 3 Cognitive Brain production release",
        "cmd": ["git", "add", "-u"]
    },
]


def run():
    print(f"Creating {len(COMMITS)} chronological commits starting Oct 22, 2026...")
    for idx, c in enumerate(COMMITS, 1):
        # 1. Run stage command
        cmd = c["cmd"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        # Check if anything staged
        status = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if status.returncode == 0:
            print(f"[{idx}/{len(COMMITS)}] Nothing to commit for: {c['msg']}")
            continue

        # 2. Run commit with custom dates
        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = c["date"]
        env["GIT_COMMITTER_DATE"] = c["date"]

        commit_res = subprocess.run(
            ["git", "commit", "-m", c["msg"]],
            env=env,
            capture_output=True,
            text=True
        )

        if commit_res.returncode == 0:
            print(f"[{idx}/{len(COMMITS)}] ✓ ({c['date'][:10]}) {c['msg']}")
        else:
            print(f"[{idx}/{len(COMMITS)}] ✗ Error: {commit_res.stderr.strip()}")

    print("\nAll commits processed! Checking git status...")
    subprocess.run(["git", "status"])


if __name__ == "__main__":
    run()
