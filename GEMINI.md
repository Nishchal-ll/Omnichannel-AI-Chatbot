# Apex Auto Recondition Nepal — AI Engine Rules & Guidelines

## 1. Hardware Constraints & Model Selection
- **Target Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB VRAM), AMD Ryzen 7 5800H, 16 GB RAM.
- **Mandatory Generation Model**: `llama3.2:latest` (3B, 2.0 GB). Never load models $\ge 8\text{B}$ or `gemma4:e2b` to prevent PCIe RAM spilling and 40s latency spikes.
- **Mandatory Embedding Model**: `nomic-embed-text` (274 MB, 768-dim) via Ollama `/api/embeddings` or lightweight local sentence-transformers.

## 2. Architecture & RAG Pipeline Rules
- **No Heavyweight Frameworks**: Avoid bloated wrappers like LangChain `RetrievalQA` or LlamaIndex. Use direct lightweight vector retrieval and direct Ollama HTTP calls.
- **Connection Reuse**: Always use direct HTTP (`/api/chat`) with connection keep-alive rather than `subprocess.run(['ollama', 'run', ...])` to eliminate process-spawning cold-starts.
- **Cosine Distance Cutoff**: Reject retrieval results with cosine distance $> 0.60$ and trigger the domain deflection guard.
- **Confidence Rating**: Calculate turn confidence based on source count and distance (<0.35 = High, 0.35–0.60 = Medium, >0.60 = Low/Refusal).

## 3. Domain & Knowledge Invariants
- **Dealership Identity**: Apex Auto Recondition House Nepal (Tinkune, Kathmandu & Nayabazar, Pokhara).
- **Vehicle Models**: Real Nepali inventory (Hyundai Santro, Grand i10, Nissan Magnite, Suzuki Alto, Swift, Brezza, Creta).
- **Pricing**: Strictly in Nepali Rupees / Lakhs (never in foreign currencies or USD).
- **Policies**: 20-min spot exchange (साटासाट), 24-hr Bluebook ownership transfer (नामसारी) at Yatayat Ekantakuna, 6-month engine/gearbox warranty.
- **Customer Channels**: Inbound traffic originates from Meta WhatsApp Cloud API (v21.0) and Facebook Messenger. Web dashboard serves as the Executive Analytics Console.

