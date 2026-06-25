const API_BASE = "http://127.0.0.1:8000";
const STORAGE_KEYS = {
  compareUrls: "compareUrls",
};

const state = {
  currentTab: null,
  currentSessionId: "default",
  indexed: false,
  latestSummary: null,
};

const $ = (selector) => document.querySelector(selector);

document.addEventListener("DOMContentLoaded", init);

async function init() {
  wireTabs();
  wireActions();
  await loadTheme();
  await loadCompareUrls();
  await loadCurrentTab();
}

function wireTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("is-active"));
      document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("is-active"));
      button.classList.add("is-active");
      $(`#${button.dataset.tab}`).classList.add("is-active");
      setStatus("");
    });
  });
}

function wireActions() {
  $("#summarizeBtn").addEventListener("click", summarizeCurrentPage);
  $("#compareBtn").addEventListener("click", comparePages);
  $("#indexBtn").addEventListener("click", indexCurrentPage);
  $("#clearChatBtn").addEventListener("click", () => {
    $("#chatLog").innerHTML = "";
  });
  $("#chatForm").addEventListener("submit", askQuestion);
  $("#copySummaryBtn").addEventListener("click", copySummary);
  $("#downloadTxtBtn").addEventListener("click", () => downloadSummary("txt"));
  $("#downloadMdBtn").addEventListener("click", () => downloadSummary("md"));
  $("#themeToggle").addEventListener("click", toggleTheme);
  $("#compareUrls").addEventListener("input", saveCompareUrls);
}

async function loadCurrentTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  state.currentTab = tab;
  state.currentSessionId = sessionIdFromUrl(tab?.url || "default");
  $("#pageTitle").textContent = tab?.title || "Current page";
}

async function summarizeCurrentPage() {
  if (!isWebUrl(state.currentTab?.url)) {
    setStatus("Open an http or https webpage before summarizing.", true);
    return;
  }

  await withBusy($("#summarizeBtn"), "Summarizing", async () => {
    const result = await postJson("/summarize", { url: state.currentTab.url });
    state.latestSummary = result;
    renderSummary(result);
    setStatus("Summary ready.");
  });
}

async function comparePages() {
  const urls = $("#compareUrls").value
    .split(/\n|,/)
    .map((url) => url.trim())
    .filter(Boolean);

  if (urls.length < 2) {
    setStatus("Add at least two URLs to compare.", true);
    return;
  }

  await withBusy($("#compareBtn"), "Comparing", async () => {
    const result = await postJson("/compare", { urls });
    renderList("#similarities", result.similarities);
    renderList("#differences", result.differences);
    $("#conclusion").textContent = result.conclusion || "No conclusion returned.";
    $("#conclusion").classList.toggle("empty", !result.conclusion);
    setStatus("Comparison ready.");
  });
}

async function loadCompareUrls() {
  const saved = await chrome.storage.local.get({ [STORAGE_KEYS.compareUrls]: "" });
  $("#compareUrls").value = saved[STORAGE_KEYS.compareUrls] || "";
}

async function saveCompareUrls() {
  await chrome.storage.local.set({
    [STORAGE_KEYS.compareUrls]: $("#compareUrls").value,
  });
}

async function indexCurrentPage() {
  if (!isWebUrl(state.currentTab?.url)) {
    setStatus("Open an http or https webpage before indexing.", true);
    return;
  }

  await withBusy($("#indexBtn"), "Indexing", async () => {
    const result = await postJson("/index-page", {
      url: state.currentTab.url,
      session_id: state.currentSessionId,
    });
    state.indexed = true;
    setStatus(`Indexed ${result.chunks_indexed} content chunks.`);
  });
}

async function askQuestion(event) {
  event.preventDefault();
  const question = $("#questionInput").value.trim();
  if (!question) {
    return;
  }

  appendMessage("user", question);
  $("#questionInput").value = "";

  try {
    if (!state.indexed) {
      await indexCurrentPage();
    }
    const result = await postJson("/chat", {
      question,
      session_id: state.currentSessionId,
    });
    appendMessage("ai", result.answer);
  } catch (error) {
    appendMessage("ai", error.message);
    setStatus(error.message, true);
  }
}

function renderSummary(summary) {
  $("#shortSummary").textContent = summary.short_summary || "No short summary returned.";
  $("#detailedSummary").textContent = summary.detailed_summary || "No detailed summary returned.";
  $("#shortSummary").classList.remove("empty");
  $("#detailedSummary").classList.remove("empty");
  renderList("#keyPoints", summary.key_points);
  renderChips("#keywords", summary.keywords);
}

function renderList(selector, items = []) {
  const list = $(selector);
  list.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
}

function renderChips(selector, items = []) {
  const wrap = $(selector);
  wrap.innerHTML = "";
  items.forEach((item) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = item;
    wrap.appendChild(chip);
  });
}

async function copySummary() {
  if (!state.latestSummary) {
    setStatus("Nothing to copy yet.", true);
    return;
  }
  await navigator.clipboard.writeText(summaryAsMarkdown(state.latestSummary));
  setStatus("Summary copied.");
}

function downloadSummary(format) {
  if (!state.latestSummary) {
    setStatus("Nothing to download yet.", true);
    return;
  }

  const isMarkdown = format === "md";
  const content = isMarkdown
    ? summaryAsMarkdown(state.latestSummary)
    : summaryAsText(state.latestSummary);
  const blob = new Blob([content], { type: isMarkdown ? "text/markdown" : "text/plain" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `webpage-summary.${format}`;
  link.click();
  URL.revokeObjectURL(url);
}

function summaryAsText(summary) {
  return [
    "Short Summary",
    summary.short_summary,
    "",
    "Detailed Summary",
    summary.detailed_summary,
    "",
    "Key Points",
    ...(summary.key_points || []).map((point) => `- ${point}`),
    "",
    "Keywords",
    (summary.keywords || []).join(", "),
  ].join("\n");
}

function summaryAsMarkdown(summary) {
  return [
    "# Webpage Summary",
    "",
    "## Short Summary",
    summary.short_summary,
    "",
    "## Detailed Summary",
    summary.detailed_summary,
    "",
    "## Key Points",
    ...(summary.key_points || []).map((point) => `- ${point}`),
    "",
    "## Keywords",
    (summary.keywords || []).map((keyword) => `\`${keyword}\``).join(", "),
  ].join("\n");
}

async function postJson(path, payload) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 90000);

  try {
    const response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || "The backend returned an error.");
    }
    return data;
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("The API request timed out.");
    }
    throw new Error(error.message || "Could not reach the backend.");
  } finally {
    clearTimeout(timeoutId);
  }
}

async function withBusy(button, label, task) {
  const previous = button.textContent;
  button.textContent = label;
  button.disabled = true;
  setStatus(`${label}...`);
  try {
    await task();
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    button.textContent = previous;
    button.disabled = false;
  }
}

function appendMessage(role, text) {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.textContent = text;
  $("#chatLog").appendChild(message);
  $("#chatLog").scrollTop = $("#chatLog").scrollHeight;
}

function setStatus(message, isError = false) {
  const status = $("#status");
  status.textContent = message;
  status.classList.toggle("is-visible", Boolean(message));
  status.classList.toggle("is-error", isError);
}

function isWebUrl(url = "") {
  return /^https?:\/\//i.test(url);
}

function sessionIdFromUrl(url) {
  let hash = 0;
  for (let index = 0; index < url.length; index += 1) {
    hash = (hash << 5) - hash + url.charCodeAt(index);
    hash |= 0;
  }
  return `tab-${Math.abs(hash)}`;
}

async function loadTheme() {
  const { darkMode } = await chrome.storage.local.get({ darkMode: false });
  document.body.classList.toggle("dark", darkMode);
}

async function toggleTheme() {
  const darkMode = !document.body.classList.contains("dark");
  document.body.classList.toggle("dark", darkMode);
  await chrome.storage.local.set({ darkMode });
}
