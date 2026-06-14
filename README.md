# CrewAI Customer Bot

A lightweight WhatsApp + Email customer support bot built with CrewAI and a local llama.cpp-compatible LLM.

## Project Overview

This repository provides a simple customer support assistant that:
- receives WhatsApp messages through a Meta webhook,
- polls Gmail for unread emails,
- classifies each incoming message,
- generates a concise reply using a local LLM,
- sends the reply back through the original channel,
- stores message history in `data/messages.json`,
- exposes a Streamlit dashboard for monitoring and retries.

This project is designed for rapid prototyping, not production-grade scaling.

## Architecture

### Core components

- `src/webhook_server.py`
  - FastAPI server exposing `/webhook` for WhatsApp inbound events.
  - Creates unified `Message` objects and passes them to the `Pipeline`.

- `src/email_poller.py`
  - Polls Gmail every `EMAIL_POLL_INTERVAL` seconds.
  - Fetches unread emails, converts them to `Message` objects, and processes them through the same pipeline.

- `src/crew/pipeline.py`
  - Main message processing entry point.
  - Adds messages to `store`, creates classifier/responder agents, runs CrewAI, sends replies, and persists status.

- `src/crew/agents.py`
  - Defines two CrewAI agents:
    - `ClassifierAgent` for intent/tone/topic classification.
    - `ResponderAgent` for generating the reply.
  - Both agents use the local LLM via `src/crew/llm.py`.

- `src/crew/tasks.py`
  - Defines the classification and response tasks for CrewAI.
  - The response task uses the classifier output as context.

- `src/crew/llm.py`
  - Wraps CrewAI's `LLM` class for a local llama.cpp/OpenAI-compatible endpoint.
  - Reads `LLM_URL` and `LLM_MODEL` from `.env`.

- `src/channels/whatsapp.py`
  - Sends reply text to WhatsApp using the Meta Cloud API.

- `src/channels/email.py`
  - Fetches unread Gmail messages and sends email replies.

- `src/store.py`
  - Persists message state in `data/messages.json`.
  - Stores payload, reply, status, and timestamp.

- `src/streamlit_app.py`
  - Dashboard for viewing WhatsApp and email messages.
  - Allows retrying failed or pending replies manually.

## Installation

### Requirements

- Python 3.11+ recommended
- Docker + Docker Compose (optional)
- `requirements.txt` includes:
  - `crewai`
  - `crewai-tools`
  - `langchain-openai`
  - `fastapi`
  - `uvicorn`
  - `streamlit`
  - `python-dotenv`
  - `requests`

### Environment

Copy `.env.example` to `.env` and fill in credentials:

```bash
cp .env.example .env
```

Required values:
- `WHATSAPP_ACCESS_TOKEN`
- `WHATSAPP_PHONE_ID`
- `WHATSAPP_WEBHOOK_TOKEN`
- `GMAIL_USER`
- `GMAIL_APP_PASSWORD`
- `LLM_URL`
- `LLM_MODEL`

Optional:
- `BOT_SYSTEM_PROMPT`
- `EMAIL_POLL_INTERVAL`

### Run locally

Option 1: Use Docker Compose

```bash
docker compose up --build
```

Option 2: Run with Python directly

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/webhook_server.py
python src/email_poller.py
streamlit run src/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### Start script

`start.sh` launches the webhook server, email poller, and dashboard together.

```bash
bash start.sh
```

## Message Flow

1. Incoming message arrives via WhatsApp webhook or email poller.
2. `Pipeline.process()` persists the message to `store`.
3. A classifier agent analyzes the message intent, tone, and topics.
4. A responder agent writes a concise reply.
5. The reply is sent back through the same channel.
6. `store` is updated with reply text and status.

## What is currently supported

- WhatsApp inbound messages via Meta webhook
- Gmail inbound messages via polling
- Local LLM inference through a llama.cpp-compatible OpenAI-style endpoint
- Dashboard monitoring and manual retry

## Where to find the key code

- `src/crew/pipeline.py` — orchestration and routing
- `src/crew/agents.py` — AI agents definitions
- `src/crew/tasks.py` — classification and response tasks
- `src/crew/llm.py` — LLM backend configuration
- `src/channels/whatsapp.py` — WhatsApp output channel
- `src/channels/email.py` — Gmail input/output
- `src/store.py` — persisted message history
- `src/streamlit_app.py` — monitoring dashboard

## Adding a Coconut Oil Knowledge Base

This project currently generates replies from the LLM without a specialized retrieval layer. To make the bot answer coconut oil questions quickly and accurately, add a knowledge base and inject relevant content into the response prompt.

### Recommended approach

1. Create a knowledge folder:

```bash
mkdir -p data/knowledge
```

2. Add coconut oil content files:

- `data/knowledge/coconut_oil_faq.md`
- `data/knowledge/coconut_oil_uses.txt`
- `data/knowledge/coconut_oil_safety.md`

3. Build a simple retrieval helper.

Example idea for `src/knowledge.py`:

```python
import os
from pathlib import Path

def load_knowledge_text() -> str:
    base = Path(__file__).resolve().parent.parent / "data" / "knowledge"
    texts = []
    for path in sorted(base.glob("*.md")):
        texts.append(path.read_text(encoding="utf-8"))
    return "\n\n".join(texts)


def get_relevant_knowledge(query: str, top_k: int = 3) -> str:
    # For a first version, return the whole knowledge corpus.
    # Later you can replace this with semantic search or vector search.
    return load_knowledge_text()
```

4. Inject knowledge into the agent prompt.

Update `src/crew/tasks.py` or `src/crew/agents.py` so the `ResponderAgent` gets domain-specific context.

Example prompt change:

```python
from src.knowledge import get_relevant_knowledge

knowledge_snippet = get_relevant_knowledge(message.body)

return Task(
    description=(
        "Write a reply to this customer message.\n\n"
        f"Knowledge base: {knowledge_snippet}\n\n"
        "Original message: {message.body}\n\n"
        "Use the classification brief from the previous task as context.\n"
        "Rules: ..."
    ),
    ...
)
```

5. Keep the knowledge base concise and focused on coconut oil topics:
- benefits and uses
- extraction and purity
- storage and shelf life
- hair, skin, and cooking applications
- safety, allergies, and cosmetic use

### Faster retrieval

For better performance and accuracy, add a real retrieval layer instead of passing the entire document every time.

Options:
- simple keyword search in `data/knowledge`
- vector search with embeddings and top-k retrieval
- use `crewai-tools` or an embedding library if available

A typical retrieval flow is:

- embed every KB paragraph or FAQ entry,
- store vectors in a small local database,
- at runtime, compute query embedding,
- retrieve top 3–5 matching passages,
- pass those passages into the LLM prompt.

### Minimal working setup for coconut oil answers

If you want the fastest path without adding new dependencies:
- add a set of coconut oil FAQ text files,
- load them in `src/crew/tasks.py`,
- prepend the relevant knowledge to the `ResponderAgent` task description,
- keep the prompt explicit: "Use the following coconut oil knowledge base when answering."

### Example `data/knowledge/coconut_oil_faq.md`

```md
# Coconut Oil FAQ

- Coconut oil is commonly used for cooking, hair care, skin moisturization, and oil pulling.
- It can be virgin, refined, or fractionated; virgin coconut oil has a stronger aroma and more antioxidants.
- Use it for scalp massage, dry skin, and as a natural makeup remover.
- Store it in a cool, dark place; it solidifies below 24°C and liquefies above that.
- Always patch-test before topical use if the customer has sensitive skin.
```

### Updating the bot for coconut oil questions

1. Add coconut oil documents to `data/knowledge/`.
2. Create `src/knowledge.py` or extend `src/crew/llm.py`.
3. Add the knowledge snippet into the responder task prompt.
4. Test by sending coconut oil questions via WhatsApp or email.

## Notes

- `data/messages.json` holds the message audit trail and is reused by Streamlit.
- No dedicated KB retrieval exists yet; prompt injection is the simplest extension.
- The local LLM is configured using `LLM_URL` and `LLM_MODEL`.

## Useful commands

```bash
cp .env.example .env
docker compose up --build
bash start.sh
```

## Next steps for production

- add security validation for incoming webhooks,
- use a real vector search or embedding store,
- add validation / retry limits for email and WhatsApp replies,
- deploy the local llama server separately or replace with a hosted LLM provider.
