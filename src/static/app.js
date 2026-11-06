/**
 * src/static/app.js
 * ─────────────────
 * Lightweight client logic for Apex Auto Recondition Nepal Chatbox:
 * 1. Simple real-time chat with instant typing indicator
 * 2. Prominent Time Taken badges on every bot response
 * 3. Collapsible Knowledge Source inspector
 * 4. Quick prompt chips and history clearing
 */

let activeUser = localStorage.getItem("apex_chat_user") || "customer_" + Math.floor(1000 + Math.random() * 9000);
localStorage.setItem("apex_chat_user", activeUser);

const messagesContainer = document.getElementById("messages-container");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const clearBtn = document.getElementById("clear-btn");
const promptChips = document.querySelectorAll(".prompt-chip");
const modelStatusText = document.getElementById("model-status-text");

// ── Check Server Health on Load ─────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      const data = await res.json();
      if (data.ollama && data.ollama.online) {
        modelStatusText.textContent = `${data.ollama.model} (${data.rag.total_chunks || 31} chunks)`;
      }
    }
  } catch (e) {
    console.debug("Health check error:", e);
  }
}

// ── Append User Message ─────────────────────────────────────────────────────
function appendUserMessage(text) {
  const row = document.createElement("div");
  row.className = "message-row user-row";
  row.innerHTML = `
    <div class="msg-bubble user-bubble">
      <div class="msg-text">${escapeHtml(text)}</div>
    </div>
  `;
  messagesContainer.appendChild(row);
  scrollToBottom();
}

// ── Append Typing Indicator ─────────────────────────────────────────────────
function appendTypingIndicator() {
  const id = "typing-" + Date.now();
  const row = document.createElement("div");
  row.id = id;
  row.className = "message-row bot-row";
  row.innerHTML = `
    <div class="msg-avatar">🚗</div>
    <div class="msg-bubble bot-bubble">
      <div class="typing-dots">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>
  `;
  messagesContainer.appendChild(row);
  scrollToBottom();
  return id;
}

// ── Append Bot Message with Time Taken Badge ────────────────────────────────
function appendBotMessage(data, fallbackText) {
  const row = document.createElement("div");
  row.className = "message-row bot-row";

  const replyText = data.reply || fallbackText || "I could not find matching verified inventory.";
  const metrics = data.metrics || {};
  const totalMs = metrics.total_ms || 0;
  const pathType = metrics.path_type || "chat";
  const tps = (metrics.tokens && metrics.tokens.tokens_per_sec) || metrics.tokens_per_sec || 0;
  const chunks = (metrics.rag && metrics.rag.chunks) || data.chunks || [];

  // Format latency pill text
  let timePillClass = "time-badge";
  let timePillText = "";
  if (totalMs < 10) {
    timePillClass += " instant";
    timePillText = `⚡ ${totalMs} ms · Instant FAQ`;
  } else if (totalMs < 4000) {
    timePillText = `⏱️ ${(totalMs / 1000).toFixed(2)}s (${totalMs} ms) · Llama 3.2 3B`;
    if (tps > 0) timePillText += ` · ${tps} t/s`;
  } else {
    timePillClass += " medium";
    timePillText = `⏱️ ${(totalMs / 1000).toFixed(2)}s · CPU`;
  }

  // Sources accordion if chunks present
  let sourcesHtml = "";
  if (chunks && chunks.length > 0) {
    const detailId = "sources-" + Date.now() + Math.floor(Math.random() * 100);
    sourcesHtml = `
      <button class="sources-toggle" onclick="toggleSources('${detailId}')">
        📚 View Sources (${chunks.length} chunks)
      </button>
      <div id="${detailId}" class="sources-box" style="display: none;">
        ${chunks.map((c, i) => `
          <div class="source-item">
            <strong>[${i+1}] ${escapeHtml(c.source || 'KB')}</strong> — ${escapeHtml(c.title || '')} 
            <em>(Distance: ${c.distance !== undefined ? c.distance : 'N/A'})</em>
          </div>
        `).join("")}
      </div>
    `;
  }

  row.innerHTML = `
    <div class="msg-avatar">🚗</div>
    <div class="msg-bubble bot-bubble">
      <div class="msg-text">${formatMarkdown(replyText)}</div>
      <div class="msg-meta">
        <span class="${timePillClass}">${timePillText}</span>
        ${sourcesHtml}
      </div>
    </div>
  `;

  messagesContainer.appendChild(row);
  scrollToBottom();
}

// ── Toggle Sources Accordion ────────────────────────────────────────────────
window.toggleSources = function(id) {
  const el = document.getElementById(id);
  if (el) {
    el.style.display = el.style.display === "none" ? "block" : "none";
  }
};

// ── Handle Sending Query ────────────────────────────────────────────────────
async function handleSend(text) {
  const query = (text || userInput.value || "").trim();
  if (!query) return;

  userInput.value = "";
  appendUserMessage(query);

  const typingId = appendTypingIndicator();
  const startTime = performance.now();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: activeUser,
        message: query
      })
    });

    const typingEl = document.getElementById(typingId);
    if (typingEl) typingEl.remove();

    if (res.ok) {
      const data = await res.json();
      appendBotMessage(data);
    } else {
      const elapsed = Math.round(performance.now() - startTime);
      appendBotMessage({
        reply: "Error: Unable to connect to the showroom assistant. Please ensure the server is running.",
        metrics: { total_ms: elapsed, path_type: "error" }
      });
    }
  } catch (err) {
    const typingEl = document.getElementById(typingId);
    if (typingEl) typingEl.remove();
    appendBotMessage({
      reply: "Network connection error. Please verify the backend server.",
      metrics: { total_ms: 0, path_type: "network_error" }
    });
  }
}

// ── Clear Conversation History ──────────────────────────────────────────────
async function handleClear() {
  if (confirm("Clear conversation history?")) {
    try {
      await fetch(`/api/history/${encodeURIComponent(activeUser)}`, { method: "DELETE" });
    } catch (e) {}
    messagesContainer.innerHTML = `
      <div class="message-row bot-row">
        <div class="msg-avatar">🚗</div>
        <div class="msg-bubble bot-bubble">
          <div class="msg-text">
            History cleared! Namaste, how can I assist you with Apex Auto Recondition Nepal today?
          </div>
          <div class="msg-meta">
            <span class="time-badge instant">⚡ Ready</span>
          </div>
        </div>
      </div>
    `;
  }
}

// ── Helper Utilities ────────────────────────────────────────────────────────
function scrollToBottom() {
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function escapeHtml(str) {
  return (str || "").replace(/[&<>"']/g, function(m) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[m];
  });
}

function formatMarkdown(text) {
  let safe = escapeHtml(text);
  // Bold **text**
  safe = safe.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  // Bullet lines
  safe = safe.replace(/^\s*[-•]\s+(.*)$/gm, "• $1<br>");
  // Linebreaks
  safe = safe.replace(/\n\n/g, "<br><br>").replace(/\n/g, "<br>");
  return safe;
}

// ── Event Listeners ─────────────────────────────────────────────────────────
chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  handleSend();
});

clearBtn.addEventListener("click", handleClear);

promptChips.forEach(chip => {
  chip.addEventListener("click", () => {
    const q = chip.dataset.query;
    if (q) handleSend(q);
  });
});

// Run health check on startup
checkHealth();
