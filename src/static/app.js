/**
 * src/static/app.js
 * ─────────────────
 * Frontend logic for Lionel Messi Cognitive Brain Chat & Diagnostics Console:
 * 1. Multi-user profile switching & SQLite history sync
 * 2. Instant chat interface with typing indicator & millisecond badges
 * 3. Real-time telemetry inspector (latency waterfall, stage table, speed t/s)
 * 4. Brain Memory Manager (view SQLite memories, delete individual facts)
 * 5. ChromaDB RAG chunk visualizer with cosine distances
 */

let activeUser = localStorage.getItem("active_chat_user") || "Nishchal";
let currentDiagnostics = null;
let currentMemories = [];

// DOM Elements
const userSelect = document.getElementById("user-select");
const addUserBtn = document.getElementById("add-user-btn");
const clearHistoryBtn = document.getElementById("clear-history-btn");
const clearMemoriesBtn = document.getElementById("clear-memories-btn");
const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const promptChips = document.querySelectorAll(".chip");

// Diagnostic DOM Elements
const diagTabs = document.querySelectorAll(".diag-tab");
const tabContents = document.querySelectorAll(".diag-tab-content");
const memCountBadge = document.getElementById("mem-count-badge");
const chunksCountBadge = document.getElementById("chunks-count-badge");
const memoriesList = document.getElementById("memories-list");
const refreshMemBtn = document.getElementById("refresh-mem-btn");
const ragChunksList = document.getElementById("rag-chunks-list");
const ragQuerySubtitle = document.getElementById("rag-query-subtitle");

// Metrics
const metricTotalTime = document.getElementById("metric-total-time");
const metricPathType = document.getElementById("metric-path-type");
const metricSpeed = document.getElementById("metric-speed");
const metricTokens = document.getElementById("metric-tokens");
const waterfallBars = document.getElementById("waterfall-bars");
const diagStagesTable = document.getElementById("diag-stages-table");

// ── 1. Initialization ────────────────────────────────────────────────────────
async function initApp() {
  await loadUsersList();
  await loadUserHistory();
  await loadUserMemories();
  setupEventListeners();
  checkHealth();
}

// ── 2. User Profiles & Context ──────────────────────────────────────────────
async function loadUsersList() {
  try {
    const res = await fetch("/api/users");
    const data = await res.json();
    const users = data.users || [];
    
    if (!users.includes("Nishchal")) users.unshift("Nishchal");
    if (!users.includes(activeUser)) users.push(activeUser);

    userSelect.innerHTML = "";
    users.forEach(u => {
      const opt = document.createElement("option");
      opt.value = u;
      opt.textContent = u;
      if (u === activeUser) opt.selected = true;
      userSelect.appendChild(opt);
    });
  } catch (e) {
    console.error("Failed to load users:", e);
  }
}

async function switchUser(newUser) {
  activeUser = newUser;
  localStorage.setItem("active_chat_user", activeUser);
  chatMessages.innerHTML = "";
  await loadUserHistory();
  await loadUserMemories();
}

// ── 3. Load Dialogue History ────────────────────────────────────────────────
async function loadUserHistory() {
  try {
    const res = await fetch(`/api/history/${encodeURIComponent(activeUser)}`);
    const data = await res.json();
    const turns = data.turns || [];

    chatMessages.innerHTML = "";
    if (turns.length === 0) {
      appendSystemGreeting();
      return;
    }

    turns.forEach(t => {
      appendMessageBubble(t.role, t.content, null);
    });
    scrollToBottom();
  } catch (e) {
    console.error("Failed to load history:", e);
  }
}

function appendSystemGreeting() {
  const greeting = `Hello! I am Lionel Messi's Cognitive AI Knowledge Assistant.
I am grounded in Messi's career records, honors, World Cup 2022 stats, and club history.
I also have long-term Brain Memory: tell me facts like *"Remember that my name is Nishchal"* and I will remember you!`;
  appendMessageBubble("assistant", greeting, { path_type: "greeting", total_ms: 0 });
}

// ── 4. Chat Sending & Response Handling ─────────────────────────────────────
async function handleSendMessage(text) {
  const msg = (text || chatInput.value).trim();
  if (!msg) return;

  chatInput.value = "";
  appendMessageBubble("user", msg, null);
  scrollToBottom();

  // Show typing bubble
  const typingId = appendTypingIndicator();
  sendBtn.disabled = true;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: activeUser,
        message: msg
      })
    });

    const data = await res.json();
    removeTypingIndicator(typingId);

    if (data.reply) {
      appendMessageBubble("assistant", data.reply, data.metrics, data);
      renderDiagnostics(data.metrics, data.user_memories, data.retrieved_chunks);
      await loadUserMemories();
    } else {
      appendMessageBubble("assistant", "⚠️ Error processing response.", null);
    }
  } catch (err) {
    removeTypingIndicator(typingId);
    appendMessageBubble("assistant", `⚠️ Connection error: ${err.message}`, null);
  } finally {
    sendBtn.disabled = false;
    scrollToBottom();
  }
}

function appendMessageBubble(role, content, metrics, fullData = null) {
  const row = document.createElement("div");
  row.className = `chat-bubble-row ${role}`;

  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${role}`;
  bubble.textContent = content;
  row.appendChild(bubble);

  if (role === "assistant") {
    const meta = document.createElement("div");
    meta.className = "bubble-meta";

    if (metrics) {
      const badge = document.createElement("span");
      badge.className = "bubble-badge";
      const sec = (metrics.total_ms / 1000).toFixed(2);
      const tokens = metrics.tokens?.completion_tokens || 0;
      badge.textContent = `⚡ ${sec}s · ${tokens} tok · ${metrics.path_type || "rag"}`;
      meta.appendChild(badge);

      if (fullData) {
        const inspectLink = document.createElement("span");
        inspectLink.className = "inspect-trigger";
        inspectLink.textContent = "Inspect Telemetry";
        inspectLink.addEventListener("click", () => {
          renderDiagnostics(fullData.metrics, fullData.user_memories, fullData.retrieved_chunks);
          switchTab("tab-telemetry");
        });
        meta.appendChild(inspectLink);
      }
    }

    row.appendChild(meta);
  }

  chatMessages.appendChild(row);
  scrollToBottom();
}

function appendTypingIndicator() {
  const id = `typing_${Date.now()}`;
  const row = document.createElement("div");
  row.id = id;
  row.className = "chat-bubble-row assistant";
  row.innerHTML = `
    <div class="chat-bubble assistant">
      <div class="typing-indicator">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>
    </div>
  `;
  chatMessages.appendChild(row);
  scrollToBottom();
  return id;
}

function removeTypingIndicator(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ── 5. Diagnostics & Telemetry Rendering ────────────────────────────────────
const STAGE_NAMES = {
  memory_ms: "Memory MESH Interceptor",
  routing_ms: "Intent Routing & Anchors",
  rewrite_ms: "Context History Rewrite",
  rag_retrieval_ms: "ChromaDB RAG Retrieval",
  prompt_build_ms: "Cognitive Prompt Assembly",
  llm_inference_ms: "Gemma 4 E2B Generation",
  dispatch_ms: "Channel Dispatch",
};

const STAGE_COLORS = {
  memory_ms: "var(--color-mem)",
  routing_ms: "var(--color-route)",
  rewrite_ms: "var(--color-rewrite)",
  rag_retrieval_ms: "var(--color-rag)",
  prompt_build_ms: "var(--color-dispatch)",
  llm_inference_ms: "var(--color-llm)",
  dispatch_ms: "#60A5FA",
};

function renderDiagnostics(metrics, memories, chunks) {
  if (!metrics) return;
  currentDiagnostics = { metrics, memories, chunks };

  // Overview Cards
  metricTotalTime.textContent = `${(metrics.total_ms / 1000).toFixed(2)}s (${metrics.total_ms}ms)`;
  metricPathType.textContent = metrics.path_type || "rag_llm";
  metricSpeed.textContent = metrics.tokens?.tokens_per_sec ? `${metrics.tokens.tokens_per_sec.toFixed(1)} t/s` : "0 ms/tok";
  metricTokens.textContent = `${metrics.tokens?.completion_tokens || 0} tokens`;

  // Latency Waterfall
  const stages = metrics.stages || {};
  const total = metrics.total_ms || 1;
  waterfallBars.innerHTML = "";
  
  const tbody = diagStagesTable.querySelector("tbody");
  tbody.innerHTML = "";

  for (const [key, duration] of Object.entries(stages)) {
    const name = STAGE_NAMES[key] || key;
    const color = STAGE_COLORS[key] || "var(--accent-blue)";
    const pct = Math.min(100, Math.max(1, Math.round((duration / total) * 100)));

    // Waterfall bar
    const item = document.createElement("div");
    item.className = "waterfall-item";
    item.innerHTML = `
      <div class="wf-header">
        <span>${name}</span>
        <span><strong>${duration} ms</strong> (${pct}%)</span>
      </div>
      <div class="wf-bar-track">
        <div class="wf-bar-fill" style="width: ${pct}%; background: ${color};"></div>
      </div>
    `;
    waterfallBars.appendChild(item);

    // Table row
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="color: ${color}; font-weight: 600;">${name}</td>
      <td>${duration} ms</td>
      <td>${pct}%</td>
    `;
    tbody.appendChild(tr);
  }

  // Update RAG Chunks Tab
  renderChunks(chunks || metrics.rag?.chunks || [], metrics.standalone_query);
}

function renderChunks(chunks, standaloneQuery) {
  chunksCountBadge.textContent = chunks.length;
  if (standaloneQuery) {
    ragQuerySubtitle.textContent = `Rewritten Query: "${standaloneQuery}"`;
  } else {
    ragQuerySubtitle.textContent = `Retrieved for direct question`;
  }

  ragChunksList.innerHTML = "";
  if (!chunks || chunks.length === 0) {
    let explanation = `No RAG retrieval needed for path: <strong>${metricPathType.textContent}</strong>.`;
    if (metricPathType.textContent === "memory_recall" || metricPathType.textContent === "memory_store") {
      explanation = `⚡ <strong>Memory MESH Path:</strong> Resolved directly from SQLite User Profile (${activeUser}). Vector search was bypassed to achieve sub-millisecond response time.`;
    } else if (metricPathType.textContent === "fast_anchor") {
      explanation = `⚡ <strong>0ms Fast Anchor Path:</strong> Resolved instantly by canonical FAQ router. Vector search was bypassed to achieve near-zero latency.`;
    } else if (metricPathType.textContent === "out_of_domain") {
      explanation = `🛡️ <strong>Domain Guard Path:</strong> Query was outside Lionel Messi's football career, so RAG was bypassed.`;
    } else if (metricPathType.textContent === "rag_llm") {
      explanation = `ℹ️ <strong>ChromaDB Query:</strong> Returned 0 chunks because semantic distance exceeded the relevance threshold.`;
    }
    ragChunksList.innerHTML = `<div class="placeholder-text" style="line-height:1.6; text-align:left; padding:15px;">${explanation}</div>`;
    return;
  }

  chunks.forEach((c, idx) => {
    const card = document.createElement("div");
    card.className = "chunk-card";
    const dist = c.distance !== undefined ? c.distance.toFixed(4) : "—";
    card.innerHTML = `
      <div class="chunk-header">
        <span>#${idx + 1} ${c.metadata?.title || c.title || "Knowledge Chunk"}</span>
        <span class="chunk-distance">Cosine Dist: ${dist}</span>
      </div>
      <div class="chunk-text">${c.text || c.content || ""}</div>
    `;
    ragChunksList.appendChild(card);
  });
}

// ── 6. Long-Term Brain Memories (SQLite) ─────────────────────────────────────
async function loadUserMemories() {
  try {
    const res = await fetch(`/api/memories/${encodeURIComponent(activeUser)}`);
    const data = await res.json();
    currentMemories = data.memories || [];
    renderMemories(currentMemories);
  } catch (e) {
    console.error("Failed to load memories:", e);
  }
}

function renderMemories(memories) {
  memCountBadge.textContent = memories.length;
  memoriesList.innerHTML = "";

  if (memories.length === 0) {
    memoriesList.innerHTML = `
      <div class="placeholder-text">
        No long-term brain memories saved for <strong>${activeUser}</strong>.<br><br>
        Try saying:<br>
        • <em>"Remember that my name is ${activeUser}"</em><br>
        • <em>"Remember that my favorite club is Barcelona"</em>
      </div>`;
    return;
  }

  memories.forEach(m => {
    const item = document.createElement("div");
    item.className = "memory-item-card";
    const dt = new Date(m.created_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    item.innerHTML = `
      <div>
        <div class="memory-text">${m.memory_text}</div>
        <div class="memory-meta">Category: ${m.category} • Saved at ${dt}</div>
      </div>
      <button class="mem-del-btn" title="Forget this fact" data-id="${m.id}">✕</button>
    `;

    item.querySelector(".mem-del-btn").addEventListener("click", async () => {
      await deleteMemory(m.id);
    });

    memoriesList.appendChild(item);
  });
}

async function deleteMemory(memoryId) {
  try {
    await fetch(`/api/memories/${encodeURIComponent(activeUser)}/${encodeURIComponent(memoryId)}`, {
      method: "DELETE"
    });
    await loadUserMemories();
  } catch (e) {
    console.error("Failed to delete memory:", e);
  }
}

// ── 7. Tab Switching & Event Listeners ───────────────────────────────────────
function switchTab(targetId) {
  diagTabs.forEach(t => {
    if (t.dataset.tab === targetId) t.classList.add("active");
    else t.classList.remove("active");
  });

  tabContents.forEach(c => {
    if (c.id === targetId) c.classList.add("active");
    else c.classList.remove("active");
  });
}

function setupEventListeners() {
  // Tabs
  diagTabs.forEach(tab => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  });

  // User Selector
  userSelect.addEventListener("change", (e) => {
    switchUser(e.target.value);
  });

  // Add User
  addUserBtn.addEventListener("click", () => {
    const name = prompt("Enter new user name / phone number:");
    if (name && name.trim()) {
      switchUser(name.trim());
      loadUsersList();
    }
  });

  // Chat Form Submit
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    handleSendMessage();
  });

  // Prompt Chips
  promptChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const q = chip.dataset.query;
      if (q) handleSendMessage(q);
    });
  });

  // Clear Chat History
  clearHistoryBtn.addEventListener("click", async () => {
    if (confirm(`Clear conversation history for ${activeUser}?`)) {
      await fetch(`/api/history/${encodeURIComponent(activeUser)}`, { method: "DELETE" });
      chatMessages.innerHTML = "";
      appendSystemGreeting();
    }
  });

  // Clear Brain Memories
  clearMemoriesBtn.addEventListener("click", async () => {
    if (confirm(`Clear all long-term brain memories for ${activeUser}?`)) {
      await fetch(`/api/memories/${encodeURIComponent(activeUser)}`, { method: "DELETE" });
      await loadUserMemories();
    }
  });

  // Refresh Memories Button
  refreshMemBtn.addEventListener("click", () => {
    loadUserMemories();
  });
}

// ── 8. Health Check ─────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    
    const ollamaDot = document.querySelector("#ollama-status-badge .status-dot");
    const ollamaText = document.getElementById("ollama-status-text");
    if (data.ollama?.online) {
      ollamaDot.classList.add("active");
      ollamaText.textContent = `Ollama: ${data.ollama.model}`;
    } else {
      ollamaDot.classList.remove("active");
      ollamaText.textContent = "Ollama: Offline";
    }

    const ragText = document.getElementById("rag-status-text");
    if (data.rag?.total_chunks) {
      ragText.textContent = `RAG: ${data.rag.total_chunks} Chunks`;
    }
  } catch (e) {
    console.warn("Health check error:", e);
  }
}

// Start app
window.addEventListener("DOMContentLoaded", initApp);
