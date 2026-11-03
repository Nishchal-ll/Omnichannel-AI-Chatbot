/**
 * src/static/app.js
 * ─────────────────
 * Frontend logic for Apex Auto Recondition Nepal Dealership Reports & Analytics Console:
 * 1. Live KPI Analytics (Total Inquiries, Unique Leads, WhatsApp vs Messenger, Top Car Models)
 * 2. Customer Leads Directory with search & channel filtering
 * 3. Complete Chronological Conversation Transcript Viewer (Inbound queries & Bot responses)
 * 4. Customer Brain Memory Intelligence Inspector (SQLite preferences, budget, exchange facts)
 * 5. Inbound Message Simulator (Test WhatsApp & Messenger without external phones)
 */

let currentOverview = null;
let customersList = [];
let selectedSenderId = null;
let currentChannelFilter = "all";

// DOM Elements - KPIs
const kpiTotalQueries = document.getElementById("kpi-total-queries");
const kpiUniqueLeads = document.getElementById("kpi-unique-leads");
const kpiWaCount = document.getElementById("kpi-wa-count");
const kpiFbCount = document.getElementById("kpi-fb-count");
const kpiAvgLatency = document.getElementById("kpi-avg-latency");
const kpiTopicsCloud = document.getElementById("kpi-topics-cloud");
const waStatusText = document.getElementById("wa-status-text");
const fbStatusText = document.getElementById("fb-status-text");
const refreshBtn = document.getElementById("refresh-btn");

// DOM Elements - Leads Sidebar
const leadsList = document.getElementById("leads-list");
const leadsCountBadge = document.getElementById("leads-count-badge");
const leadSearchInput = document.getElementById("lead-search-input");
const filterTabs = document.querySelectorAll(".filter-tab");

// DOM Elements - Transcript Pane
const leadDetailHeader = document.getElementById("lead-detail-header");
const leadNameDisplay = document.getElementById("lead-name-display");
const leadChannelBadge = document.getElementById("lead-channel-badge");
const leadMetaDisplay = document.getElementById("lead-meta-display");
const leadActionsBar = document.getElementById("lead-actions-bar");
const clearLeadBtn = document.getElementById("clear-lead-btn");
const customerPreferencesBox = document.getElementById("customer-preferences-box");
const customerPrefItems = document.getElementById("customer-pref-items");
const prefCountBadge = document.getElementById("pref-count-badge");
const transcriptScrollArea = document.getElementById("transcript-scroll-area");
const transcriptSimulatorDock = document.getElementById("transcript-simulator-dock");
const simInlineForm = document.getElementById("sim-inline-form");
const simInlineInput = document.getElementById("sim-inline-input");

// DOM Elements - Modal Simulator
const openSimModalBtn = document.getElementById("open-sim-modal-btn");
const closeSimModalBtn = document.getElementById("close-sim-modal-btn");
const cancelSimBtn = document.getElementById("cancel-sim-btn");
const simModalBackdrop = document.getElementById("sim-modal-backdrop");
const simModal = document.getElementById("sim-modal");
const simModalForm = document.getElementById("sim-modal-form");
const simChannelSelect = document.getElementById("sim-channel-select");
const simSenderInput = document.getElementById("sim-sender-input");
const simMessageInput = document.getElementById("sim-message-input");

// ── 1. Initialization ────────────────────────────────────────────────────────
async function initApp() {
  setupEventListeners();
  await refreshReports();
}

// ── 2. Load Analytics Overview ──────────────────────────────────────────────
async function loadOverview() {
  try {
    const res = await fetch("/api/reports/overview");
    const data = await res.json();
    currentOverview = data;

    kpiTotalQueries.textContent = data.total_queries || 0;
    kpiUniqueLeads.textContent = data.unique_leads || 0;
    kpiWaCount.textContent = data.channels?.whatsapp || 0;
    kpiFbCount.textContent = data.channels?.messenger || 0;

    const latSec = data.avg_latency_ms ? (data.avg_latency_ms / 1000).toFixed(2) : "1.20";
    kpiAvgLatency.textContent = `${latSec}s`;

    // Status pills
    if (data.channel_status?.whatsapp_configured) {
      waStatusText.textContent = "WhatsApp (Cloud API)";
    } else {
      waStatusText.textContent = "WhatsApp (Ready)";
    }

    if (data.channel_status?.messenger_configured) {
      fbStatusText.textContent = "Messenger (Send API)";
    } else {
      fbStatusText.textContent = "Messenger (Ready)";
    }

    // Top Topics Cloud
    kpiTopicsCloud.innerHTML = "";
    const topics = data.top_topics || [];
    if (topics.length === 0) {
      kpiTopicsCloud.innerHTML = `<span class="topic-pill">No inquiries yet</span>`;
    } else {
      topics.forEach(t => {
        const pill = document.createElement("span");
        pill.className = "topic-pill";
        pill.innerHTML = `${escapeHtml(t.name)} <strong>(${t.count})</strong>`;
        kpiTopicsCloud.appendChild(pill);
      });
    }
  } catch (e) {
    console.error("Failed to load overview reports:", e);
  }
}

// ── 3. Load Customers Leads Directory ───────────────────────────────────────
async function loadCustomers() {
  try {
    const res = await fetch("/api/reports/customers");
    const data = await res.json();
    customersList = data.customers || [];
    renderLeadsList();

    // Auto-select first lead if none selected
    if (!selectedSenderId && customersList.length > 0) {
      selectCustomer(customersList[0].sender_id, customersList[0].channel);
    } else if (selectedSenderId) {
      const exists = customersList.find(c => c.sender_id === selectedSenderId);
      if (exists) selectCustomer(exists.sender_id, exists.channel, false);
    }
  } catch (e) {
    console.error("Failed to load customer directory:", e);
  }
}

function renderLeadsList() {
  const query = leadSearchInput.value.toLowerCase().trim();
  const filtered = customersList.filter(c => {
    const matchChan = (currentChannelFilter === "all") || (c.channel === currentChannelFilter);
    const matchQuery = !query || 
      c.sender_id.toLowerCase().includes(query) || 
      (c.last_message && c.last_message.toLowerCase().includes(query));
    return matchChan && matchQuery;
  });

  leadsCountBadge.textContent = filtered.length;
  leadsList.innerHTML = "";

  if (filtered.length === 0) {
    leadsList.innerHTML = `<div class="loading-state">No matching customer leads found.</div>`;
    return;
  }

  filtered.forEach(c => {
    const item = document.createElement("div");
    item.className = `lead-item ${c.sender_id === selectedSenderId ? "active" : ""}`;
    item.setAttribute("data-sender", c.sender_id);

    const channelClass = c.channel === "messenger" ? "messenger" : (c.channel === "whatsapp" ? "whatsapp" : "chat");
    const timeDisplay = c.last_active && c.last_active !== "Recent" ? formatTime(c.last_active) : "Recent";

    item.innerHTML = `
      <div class="lead-item-top">
        <span class="lead-sender-name">${escapeHtml(c.sender_id)}</span>
        <span class="lead-badge ${channelClass}">${c.channel}</span>
      </div>
      <div class="lead-snippet">${escapeHtml(c.last_message || "No message body")}</div>
      <div class="lead-item-footer">
        <span>${timeDisplay}</span>
        <span>${c.total_inquiries} turns</span>
      </div>
    `;

    item.addEventListener("click", () => {
      selectCustomer(c.sender_id, c.channel);
    });

    leadsList.appendChild(item);
  });
}

// ── 4. Select Customer & Load Transcript ────────────────────────────────────
async function selectCustomer(senderId, channel, scrollToBottom = true) {
  selectedSenderId = senderId;
  document.querySelectorAll(".lead-item").forEach(el => {
    if (el.getAttribute("data-sender") === senderId) {
      el.classList.add("active");
    } else {
      el.classList.remove("active");
    }
  });

  // Update detail header
  leadNameDisplay.textContent = senderId;
  leadChannelBadge.className = `channel-badge ${channel || "whatsapp"}`;
  leadChannelBadge.textContent = (channel || "whatsapp").toUpperCase();
  leadMetaDisplay.textContent = `Omnichannel Lead ID: ${senderId} · Multi-turn session active`;
  leadActionsBar.style.display = "block";
  transcriptSimulatorDock.style.display = "block";

  // Load details
  try {
    const res = await fetch(`/api/reports/customer/${encodeURIComponent(senderId)}`);
    const data = await res.json();
    renderCustomerPreferences(data.memories || []);
    renderTranscript(data.turns || [], data.raw_messages || [], scrollToBottom);
  } catch (e) {
    console.error("Failed to load customer details:", e);
  }
}

function renderCustomerPreferences(memories) {
  customerPrefItems.innerHTML = "";
  prefCountBadge.textContent = `${memories.length} facts`;

  if (memories.length === 0) {
    customerPreferencesBox.style.display = "none";
    return;
  }

  customerPreferencesBox.style.display = "flex";
  memories.forEach(m => {
    const tag = document.createElement("span");
    tag.className = "pref-fact-tag";
    tag.textContent = m.fact || m.memory_text;
    customerPrefItems.appendChild(tag);
  });
}

function renderTranscript(turns, rawMessages, shouldScroll = true) {
  transcriptScrollArea.innerHTML = "";

  if (turns.length === 0 && rawMessages.length === 0) {
    transcriptScrollArea.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">💬</div>
        <h3>No conversation turns recorded</h3>
        <p>This customer does not have any recorded dialogue turns yet. Use the simulator below to simulate an inbound query.</p>
      </div>
    `;
    return;
  }

  // If we have rawMessages from store, use them for rich latency data
  if (rawMessages.length > 0) {
    rawMessages.forEach(m => {
      // Inbound Customer Query
      const custTurn = document.createElement("div");
      custTurn.className = "transcript-turn customer";
      custTurn.innerHTML = `
        <div class="transcript-bubble">${escapeHtml(m.body || "")}</div>
        <div class="transcript-meta">
          <span>${formatTime(m.ts)}</span>
          <span>Customer (${m.channel})</span>
        </div>
      `;
      transcriptScrollArea.appendChild(custTurn);

      // Outbound Bot Reply
      if (m.reply) {
        const botTurn = document.createElement("div");
        botTurn.className = "transcript-turn bot";
        const metric = m.metrics || {};
        const latSec = metric.total_ms ? `${(metric.total_ms / 1000).toFixed(2)}s` : "0.5s";
        const pathType = metric.path_type || "rag";

        botTurn.innerHTML = `
          <div class="transcript-bubble">${escapeHtml(m.reply)}</div>
          <div class="transcript-meta">
            <span class="latency-pill">⚡ ${latSec} · ${pathType}</span>
            <span>Apex Support Bot</span>
          </div>
        `;
        transcriptScrollArea.appendChild(botTurn);
      }
    });
  } else {
    // Fallback to SQLite turns
    turns.forEach(t => {
      const row = document.createElement("div");
      const isUser = t.role === "user";
      row.className = `transcript-turn ${isUser ? "customer" : "bot"}`;
      row.innerHTML = `
        <div class="transcript-bubble">${escapeHtml(t.content)}</div>
        <div class="transcript-meta">
          <span>${isUser ? "Customer" : "Apex Support Bot"}</span>
        </div>
      `;
      transcriptScrollArea.appendChild(row);
    });
  }

  if (shouldScroll) {
    transcriptScrollArea.scrollTop = transcriptScrollArea.scrollHeight;
  }
}

// ── 5. Inbound Simulator Execution ──────────────────────────────────────────
async function handleSimulate(channel, sender, message) {
  if (!message || !message.trim()) return;

  try {
    const res = await fetch("/api/channels/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        channel: channel,
        sender: sender,
        message: message.trim()
      })
    });

    const data = await res.json();
    if (data.error) {
      alert(`Simulation error: ${data.error}`);
      return;
    }

    // Refresh directory and select this customer
    await loadOverview();
    await loadCustomers();
    selectCustomer(sender, channel, true);
  } catch (err) {
    alert(`Simulation network error: ${err.message}`);
  }
}

// ── 6. Refresh All Reports ──────────────────────────────────────────────────
async function refreshReports() {
  refreshBtn.disabled = true;
  refreshBtn.textContent = "Refreshing...";
  await loadOverview();
  await loadCustomers();
  refreshBtn.disabled = false;
  refreshBtn.textContent = "🔄 Refresh Reports";
}

// ── 7. Event Listeners ──────────────────────────────────────────────────────
function setupEventListeners() {
  refreshBtn.addEventListener("click", refreshReports);

  // Search input
  leadSearchInput.addEventListener("input", () => {
    renderLeadsList();
  });

  // Channel filter tabs
  filterTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      filterTabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      currentChannelFilter = tab.getAttribute("data-channel");
      renderLeadsList();
    });
  });

  // Clear customer history
  clearLeadBtn.addEventListener("click", async () => {
    if (!selectedSenderId) return;
    if (confirm(`Clear all conversation context for lead ${selectedSenderId}?`)) {
      await fetch(`/api/history/${encodeURIComponent(selectedSenderId)}`, { method: "DELETE" });
      await refreshReports();
      selectCustomer(selectedSenderId, "whatsapp");
    }
  });

  // Inline simulator form
  simInlineForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = simInlineInput.value;
    if (!text || !selectedSenderId) return;
    simInlineInput.value = "";
    
    // Detect channel from selected lead
    const currentLead = customersList.find(c => c.sender_id === selectedSenderId);
    const channel = currentLead ? currentLead.channel : "whatsapp";
    await handleSimulate(channel, selectedSenderId, text);
  });

  // Modal simulator controls
  openSimModalBtn.addEventListener("click", () => {
    simModal.classList.add("open");
    simModalBackdrop.classList.add("open");
  });

  const closeModal = () => {
    simModal.classList.remove("open");
    simModalBackdrop.classList.remove("open");
  };

  closeSimModalBtn.addEventListener("click", closeModal);
  cancelSimBtn.addEventListener("click", closeModal);
  simModalBackdrop.addEventListener("click", closeModal);

  simModalForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const chan = simChannelSelect.value;
    const sender = simSenderInput.value.trim() || "+977-9801234567";
    const msg = simMessageInput.value.trim();
    if (!msg) return;

    closeModal();
    simMessageInput.value = "";
    await handleSimulate(chan, sender, msg);
  });
}

function formatTime(isoStr) {
  if (!isoStr || isoStr === "Recent") return "Recent";
  try {
    const d = new Date(isoStr);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", month: "short", day: "numeric" });
  } catch (e) {
    return isoStr;
  }
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

document.addEventListener("DOMContentLoaded", initApp);
