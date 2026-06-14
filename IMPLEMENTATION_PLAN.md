# Implementation Plan — Coconut Oil Knowledge Base with ChromaDB RAG

## Goal

Make the CrewAI bot answer coconut oil customer questions accurately and fast
by retrieving only the relevant 2–3 paragraphs from ChromaDB and injecting
them into the agent prompt — keeping total context under 500 tokens so the
local 1.5B model can respond quickly without hallucinating.

---

## Why ChromaDB + Chunking

Customer emails and WhatsApp messages can be long. Passing the entire KB
into every prompt wastes context window and confuses small models. Instead:

1. KB files are split into 150-word chunks on first load
2. Each chunk is embedded into a vector (384 numbers) by `all-MiniLM-L6-v2`
3. ChromaDB stores these vectors locally in `./chroma_db/`
4. At runtime, the query is embedded and the top 3 closest chunks are returned
5. Only those ~300 words go into the agent prompt — nothing else

This means a 10,000-word KB costs the same prompt space as a 300-word one.

---

## Token Budget (Qwen 2.5 1.5B has 1024 token context window)

```
System instruction      ~50 tokens
Retrieved KB context   ~350 tokens   ← 3 chunks × 150 words
Customer message        ~80 tokens   ← trimmed to 80 words
Classification brief    ~30 tokens
Reply instructions      ~20 tokens
─────────────────────────────────────
Total prompt            ~530 tokens
LLM reply budget        ~250 tokens
─────────────────────────────────────
Grand total             ~780 tokens  ← safely under 1024
```

---

## File Structure After Implementation

```
email-ai-agent/
│
├── data/
│   ├── knowledge/
│   │   ├── coconut_oil_faq.md       ← Q&A pairs
│   │   ├── coconut_oil_uses.md      ← cooking/hair/skin uses
│   │   └── coconut_oil_products.md  ← your prices, delivery, contact
│   └── messages.json                ← existing message audit trail
│
├── chroma_db/                       ← auto-created by ChromaDB on first run
│
├── src/
│   ├── knowledge.py                 ← NEW: chunk + embed + retrieve
│   ├── crew/
│   │   ├── tasks.py                 ← EDIT: inject KB into responder task
│   │   ├── agents.py                ← unchanged
│   │   ├── pipeline.py              ← unchanged
│   │   └── llm.py                   ← unchanged
│   ├── channels/
│   │   ├── email.py                 ← unchanged
│   │   └── whatsapp.py              ← unchanged
│   ├── store.py                     ← unchanged
│   ├── streamlit_app.py             ← EDIT: add KB admin page
│   ├── webhook_server.py            ← unchanged
│   └── email_poller.py              ← unchanged
│
├── scripts/
│   └── ingest_kb.py                 ← NEW: one-time KB loader script
│
└── requirements.txt                 ← ADD: chromadb, sentence-transformers
```

---

## Phase 1 — Install Dependencies

Add to `requirements.txt`:

```
chromadb>=0.5.0
sentence-transformers>=2.7.0
```

Install:

```bash
pip install chromadb sentence-transformers
```

No GPU needed. `all-MiniLM-L6-v2` runs on CPU in under 1 second per query.

---

## Phase 2 — Write the Knowledge Files

### `data/knowledge/coconut_oil_faq.md`

Write one question-answer pair per paragraph. Keep each under 100 words.
Short paragraphs = better retrieval precision.

```markdown
## What is virgin coconut oil?
Virgin coconut oil is cold-pressed from fresh coconut meat without heat or chemicals.
It retains natural aroma, antioxidants, and lauric acid. Best for raw consumption,
skin, and hair care.

## What is refined coconut oil?
Refined coconut oil is made from dried copra and processed to remove odour and colour.
It has a higher smoke point (~230°C) making it better for frying and baking.
It contains less antioxidants than virgin oil.

## How should I store coconut oil?
Store in a cool, dark place away from sunlight. It solidifies below 24°C and
liquefies above — this is normal and does not affect quality. Shelf life is
18–24 months when stored correctly.

## Is coconut oil safe for all skin types?
Patch test before using on skin. Coconut oil is comedogenic (can clog pores)
and is not recommended for oily or acne-prone skin. It works well for dry skin,
eczema, and as a hair mask.

## Can I use coconut oil for cooking?
Yes. Virgin oil is best for low-to-medium heat cooking. Refined oil suits
deep frying. One tablespoon contains approximately 120 calories and 14g fat.
```

### `data/knowledge/coconut_oil_uses.md`

```markdown
## Hair Care Uses
Apply warm coconut oil to scalp and hair roots. Leave for 30–60 minutes then
wash out with shampoo. Reduces protein loss in chemically treated hair.
Use as a leave-in conditioner on dry ends. Repeat 2–3 times per week.

## Skin Care Uses
Use as a natural body moisturiser after shower. Apply thin layer to lips as
balm. Works as a gentle makeup remover. For dry hands, apply before bed and
wear cotton gloves overnight. Avoid applying directly on open wounds.

## Cooking Uses
Substitute 1:1 for butter in baking. Use for sautéing vegetables, making
popcorn, or greasing baking trays. Adds mild coconut flavour to curries and
smoothies. Not recommended for very high temperature deep frying with virgin oil.

## Oil Pulling
Swish one tablespoon of coconut oil in the mouth for 15–20 minutes on an
empty stomach. Spit out — do not swallow. Rinse with warm water. Reported
to reduce harmful mouth bacteria. Not a substitute for brushing and flossing.
```

### `data/knowledge/coconut_oil_products.md`

```markdown
## Virgin Coconut Oil 500ml
Price: NPR 850. Cold-pressed, unrefined, organic certified. Strong coconut
aroma. Best for cooking, hair masks, and skin care. In stock.

## Virgin Coconut Oil 250ml
Price: NPR 480. Same quality as 500ml. Ideal for first-time buyers or travel.

## Refined Coconut Oil 1 Litre
Price: NPR 1,100. Neutral smell, high smoke point. Best for daily cooking
and deep frying. In stock.

## Fractionated Coconut Oil 200ml
Price: NPR 650. Always liquid at room temperature. Odourless. Used for
massage, cosmetic blending, and carrier oil for essential oils.

## Delivery Policy
Kathmandu Valley: 1–2 business days. Outside valley: 3–5 business days.
Free delivery on orders above NPR 2,000. Cash on delivery available.

## Refund Policy
Unopened products can be returned within 7 days. Opened products cannot
be returned unless defective. Contact support within 48 hours of delivery.

## Contact
Phone: +977-XXXXXXXXXX (10am–6pm, Sunday–Friday)
Email: support@yourcoconutbrand.com
WhatsApp: same number as above
```

---

## Phase 3 — Create `src/knowledge.py`

This file handles all ChromaDB operations: ingestion and retrieval.

```python
"""
knowledge.py
────────────
Chunks KB files → embeds with all-MiniLM-L6-v2 → stores in ChromaDB.
At query time: embeds query → returns top-k chunks as one context string.

Design constraints:
- Each chunk is ≤150 words → always under all-MiniLM-L6-v2's 512-token limit
- top_k=3 → ~300-350 words of context → ~420 tokens → fits 1.5B context window
"""

import re
import logging
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

log = logging.getLogger(__name__)

KB_DIR   = Path(__file__).resolve().parent.parent / "data" / "knowledge"
DB_PATH  = str(Path(__file__).resolve().parent.parent / "chroma_db")

# Embedding model: 384-dim vectors, runs on CPU, ~50ms per query
_ef      = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu",
)
_client  = chromadb.PersistentClient(path=DB_PATH)
_col     = _client.get_or_create_collection(
    name="coconut_kb",
    embedding_function=_ef,
    metadata={"hnsw:space": "cosine"},
)


# ── Chunker ───────────────────────────────────────────────────────
def _chunk(text: str, size: int = 150, overlap: int = 20) -> list[str]:
    """
    Split text into overlapping word-level chunks.
    size=150 words stays safely under the 512-token embedding model limit.
    overlap=20 prevents losing context at chunk boundaries.
    """
    text  = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if not words:
        return []
    chunks, step = [], size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + size])
        if len(chunk.strip()) > 30:
            chunks.append(chunk)
        if i + size >= len(words):
            break
    return chunks


# ── Ingest ────────────────────────────────────────────────────────
def ingest_kb(force: bool = False) -> int:
    """
    Load all .md and .txt files from data/knowledge/ into ChromaDB.
    Skip if already loaded unless force=True.
    Returns number of chunks stored.
    """
    if _col.count() > 0 and not force:
        log.info("KB already loaded (%d chunks). Pass force=True to reload.", _col.count())
        return _col.count()

    if not KB_DIR.exists():
        log.warning("data/knowledge/ not found. Create it and add .md files.")
        return 0

    all_files = list(KB_DIR.glob("*.md")) + list(KB_DIR.glob("*.txt"))
    if not all_files:
        log.warning("No .md or .txt files found in data/knowledge/")
        return 0

    all_chunks, all_ids, all_metas = [], [], []

    for path in sorted(all_files):
        text   = path.read_text(encoding="utf-8")
        chunks = _chunk(text)
        for i, chunk in enumerate(chunks):
            cid = f"{path.stem}_c{i}"
            all_chunks.append(chunk)
            all_ids.append(cid)
            all_metas.append({"source": path.name, "chunk_index": i})

    if not all_chunks:
        return 0

    # Upsert in batches of 100 to avoid memory spikes
    batch = 100
    for i in range(0, len(all_chunks), batch):
        _col.upsert(
            ids       = all_ids[i : i + batch],
            documents = all_chunks[i : i + batch],
            metadatas = all_metas[i : i + batch],
        )

    log.info("KB ingested: %d chunks from %d files.", len(all_chunks), len(all_files))
    return len(all_chunks)


# ── Retrieve ──────────────────────────────────────────────────────
def get_relevant_knowledge(query: str, top_k: int = 3) -> str:
    """
    Embed query → cosine search → return top_k chunks joined as one string.
    Total output is ~300 words / ~420 tokens — fits inside 1.5B context window.
    Auto-ingests KB on first call if ChromaDB is empty.
    """
    if _col.count() == 0:
        loaded = ingest_kb()
        if loaded == 0:
            return ""

    k       = min(top_k, _col.count())
    results = _col.query(query_texts=[query], n_results=k, include=["documents", "metadatas"])
    docs    = results.get("documents", [[]])[0]
    metas   = results.get("metadatas", [[]])[0]

    # Join with source label so agent knows where each fact came from
    parts = []
    for doc, meta in zip(docs, metas):
        source = meta.get("source", "kb")
        parts.append(f"[{source}]\n{doc}")

    return "\n\n".join(parts)


def kb_stats() -> dict:
    return {"total_chunks": _col.count(), "db_path": DB_PATH}
```

---

## Phase 4 — Edit `src/crew/tasks.py`

Change only the responder task description to inject KB context.
Everything else in tasks.py stays the same.

```python
# Add at top of tasks.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledge import get_relevant_knowledge

# Inside your existing make_response_task() or equivalent function,
# replace the description string with this:

def make_response_task(message, classification_output, responder_agent):
    # Trim message body to 80 words to save tokens
    body_words = message.body.split()
    body_short = " ".join(body_words[:80]) + ("…" if len(body_words) > 80 else "")

    # Retrieve only the relevant KB chunks — NOT the whole KB
    kb_context = get_relevant_knowledge(message.body, top_k=3)
    kb_block   = kb_context if kb_context.strip() else "No specific product info found."

    return Task(
        description=(
            "You are a coconut oil product support agent.\n\n"
            "PRODUCT KNOWLEDGE:\n"
            f"{kb_block}\n\n"
            "CUSTOMER MESSAGE:\n"
            f"{body_short}\n\n"
            "CLASSIFICATION:\n"
            f"{classification_output}\n\n"
            "RULES:\n"
            "- Answer using ONLY the product knowledge above\n"
            "- If the answer is not in the knowledge base, say:\n"
            "  'I will check and get back to you shortly.'\n"
            "- Keep reply to 2–4 sentences\n"
            "- Be polite and professional\n"
        ),
        expected_output="A 2–4 sentence reply using only the provided knowledge base.",
        agent=responder_agent,
    )
```

---

## Phase 5 — Create `scripts/ingest_kb.py`

Run this once after adding or updating knowledge files.

```python
"""
scripts/ingest_kb.py
Run: python scripts/ingest_kb.py
Run with --force to reload after editing KB files.
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge import ingest_kb, kb_stats

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help="Re-ingest even if KB exists")
args = parser.parse_args()

print("Ingesting knowledge base…")
count = ingest_kb(force=args.force)
stats = kb_stats()
print(f"Done. {count} chunks stored in {stats['db_path']}")
```

---

## Phase 6 — Add KB Admin Page to `src/streamlit_app.py`

Add a new page to your existing Streamlit app for managing the KB.

```python
# Add to your existing page radio options:
# "🥥 Knowledge Base"

# Then add this page handler:
elif page == "🥥 Knowledge Base":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from knowledge import ingest_kb, get_relevant_knowledge, kb_stats

    st.title("🥥 Knowledge Base")

    stats = kb_stats()
    col1, col2 = st.columns(2)
    col1.metric("Chunks in ChromaDB", stats["total_chunks"])
    col2.metric("DB Path", stats["db_path"])

    st.divider()

    # Ingest button
    col_a, col_b = st.columns(2)
    if col_a.button("⚡ Ingest KB Files", use_container_width=True):
        with st.spinner("Chunking and embedding data/knowledge/ files…"):
            count = ingest_kb(force=True)
        st.success(f"Done — {count} chunks stored")

    # Test retrieval
    st.divider()
    st.subheader("Test Retrieval")
    query = st.text_input("Enter a test query", placeholder="What is the price of 500ml oil?")
    if query:
        result = get_relevant_knowledge(query, top_k=3)
        st.code(result or "No results — run Ingest first", language="markdown")
```

---

## Phase 7 — Startup Order

Update `start.sh` to auto-ingest KB on startup:

```bash
#!/bin/bash
set -e

echo "=== Ingesting knowledge base ==="
python scripts/ingest_kb.py

echo "=== Starting webhook server ==="
python src/webhook_server.py &

echo "=== Starting email poller ==="
python src/email_poller.py &

echo "=== Starting Streamlit dashboard ==="
streamlit run src/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

wait
```

---

## Runtime Flow After Implementation

```
Customer message arrives (WhatsApp or email)
        │
        ▼
  Pipeline.process(message)
        │
        ▼
  ClassifierAgent
  → intent: query / feedback / general
  → topic: pricing / hair care / cooking / delivery
        │
        ▼
  get_relevant_knowledge(message.body, top_k=3)
  → embed query with all-MiniLM-L6-v2  (~50ms)
  → cosine search ChromaDB              (~5ms)
  → returns 3 chunks × 150 words        (~300 words total)
        │
        ▼
  ResponderAgent prompt:
  [system ~50t] + [kb ~420t] + [message ~100t] + [rules ~30t]
  = ~600 tokens total → safely under 1024 token limit
        │
        ▼
  Local LLM generates reply             (~2–5 seconds)
        │
        ▼
  channel.send(reply)
  → WhatsApp API or Gmail SMTP
        │
        ▼
  store.save(message, reply, status)
```

---

## Checklist

```
[ ] pip install chromadb sentence-transformers
[ ] mkdir -p data/knowledge
[ ] Create coconut_oil_faq.md
[ ] Create coconut_oil_uses.md
[ ] Create coconut_oil_products.md  ← your real prices here
[ ] Create src/knowledge.py
[ ] Edit src/crew/tasks.py          ← inject kb_context into responder task
[ ] Create scripts/ingest_kb.py
[ ] Run: python scripts/ingest_kb.py
[ ] Test: python -c "import sys; sys.path.insert(0,'src'); from knowledge import get_relevant_knowledge; print(get_relevant_knowledge('price 500ml'))"
[ ] Add KB page to streamlit_app.py
[ ] Update start.sh to run ingest on startup
[ ] Run: bash start.sh
```

---

## When to Re-ingest

Re-run `python scripts/ingest_kb.py --force` after:

- Adding new product lines or prices
- Updating delivery or refund policy
- Adding new FAQ entries
- Any edit to files in `data/knowledge/`

The `--force` flag clears and rebuilds the ChromaDB collection.
Without `--force` the script skips if the collection already has data.

---

## Notes

- `chroma_db/` is auto-created in the project root. Add it to `.gitignore`.
- The embedding model downloads once (~90MB) on first run and is cached locally.
- `all-MiniLM-L6-v2` has a 512-token hard limit — the 150-word chunk size
  keeps every chunk under ~200 tokens, well within that limit.
- If you later switch to a 7B model with a 4096-token window, increase
  `top_k` to 8 and `chunk size` to 300 for richer context.
