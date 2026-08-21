const state = {
  health: null,
  corpus: null,
  evaluations: [],
  busy: false,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const PUBLIC_SAMPLE_QUESTIONS = [
  {
    sourceId: "esp-idf-wifi-events",
    label: "Wi-Fi 重连",
    question: "ESP-IDF Wi-Fi 断开后如何重连，何时可以创建 socket？",
  },
  {
    sourceId: "esp-idf-nvs",
    label: "NVS 提交",
    question: "NVS 写入后为什么要调用 nvs_commit？",
  },
  {
    sourceId: "freertos-task-delay",
    label: "周期任务",
    question: "FreeRTOS 周期任务为什么适合使用 xTaskDelayUntil？",
  },
];

document.addEventListener("DOMContentLoaded", () => {
  bindViews();
  bindChat();
  bindUpload();
  void refreshWorkspace();
});

async function refreshWorkspace() {
  try {
    const [health, corpus, evaluations] = await Promise.all([
      fetchJson("/health"),
      fetchJson("/v1/corpus/sources"),
      fetchJson("/v1/evaluations"),
    ]);
    state.health = health;
    state.corpus = corpus;
    state.evaluations = evaluations;
    renderRuntime();
    renderCorpus();
    renderSuggestions(corpus.items);
    renderEvaluations();
  } catch (error) {
    $("#health-dot").className = "status-dot error";
    $("#health-label").textContent = "服务不可用";
    $("#runtime-label").textContent = readableError(error);
  }
}

function bindViews() {
  $$(".view-tab").forEach((button) => {
    button.addEventListener("click", () => {
      $$(".view-tab").forEach((item) => item.classList.remove("active"));
      $$(".view").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      $(`#${button.dataset.view}-view`).classList.add("active");
    });
  });
}

function bindChat() {
  const form = $("#question-form");
  const input = $("#question-input");
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    void askQuestion(input.value);
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  $$(".suggestions button").forEach((button) => {
    button.addEventListener("click", () => void askQuestion(button.dataset.question));
  });
  $("#clear-chat").addEventListener("click", () => {
    $("#conversation").innerHTML = "";
    $("#trace-list").className = "trace-list empty";
    $("#trace-list").textContent = "等待检索";
    $("#citation-list").innerHTML = "";
    setEvidenceStatus("等待问题", "neutral");
  });
}

async function askQuestion(rawQuestion) {
  const question = rawQuestion.trim();
  if (!question || state.busy) return;

  state.busy = true;
  $("#send-button").disabled = true;
  $("#question-input").value = "";
  $("#question-input").style.height = "auto";
  appendMessage("user", "Operator", question);
  const loading = appendMessage("assistant loading", "Knowledge Agent", "正在检索并核验证据...");
  setEvidenceStatus("执行中", "warning");

  try {
    const response = await fetchJson("/v1/agent/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: question, top_k: 4 }),
    });
    loading.remove();
    appendMessage(
      "assistant",
      "Knowledge Agent",
      response.answer,
      `${formatLatency(response.latency_ms)} · ${response.generation_mode}`,
    );
    renderEvidence(response);
  } catch (error) {
    loading.remove();
    appendMessage("assistant", "Knowledge Agent", `请求失败：${readableError(error)}`);
    setEvidenceStatus("请求失败", "warning");
  } finally {
    state.busy = false;
    $("#send-button").disabled = false;
    $("#question-input").focus();
  }
}

function appendMessage(kind, author, body, meta = currentTime()) {
  const article = document.createElement("article");
  article.className = `message ${kind}`;
  article.innerHTML = `
    <div class="message-meta">
      <strong>${escapeHtml(author)}</strong>
      <span>${escapeHtml(meta)}</span>
    </div>
    <div class="message-body">${escapeHtml(body)}</div>
  `;
  $("#conversation").append(article);
  article.scrollIntoView({ block: "end", behavior: "smooth" });
  return article;
}

function renderEvidence(response) {
  const trace = $("#trace-list");
  trace.className = "trace-list";
  trace.innerHTML = response.trace
    .map((step) => `<span class="trace-step">${escapeHtml(step)}</span>`)
    .join("");

  $("#citation-list").innerHTML = response.citations
    .map((citation, index) => {
      const link = safeSourceUrl(citation.source_url);
      const title = escapeHtml(citation.title);
      return `
        <article class="citation">
          <span class="citation-index">${index + 1}</span>
          ${link
            ? `<a href="${escapeAttribute(link)}" target="_blank" rel="noreferrer">${title}</a>`
            : `<strong>${title}</strong>`}
          <p>${escapeHtml(citation.section)} · ${escapeHtml(citation.source_id)}</p>
        </article>
      `;
    })
    .join("");

  if (response.status === "no_evidence") {
    setEvidenceStatus("证据不足", "warning");
  } else if (response.degraded) {
    setEvidenceStatus("已降级回答", "warning");
  } else {
    setEvidenceStatus("证据通过", "ok");
  }
}

function renderRuntime() {
  const health = state.health;
  $("#health-dot").className = "status-dot ok";
  $("#health-label").textContent = "服务正常";
  $("#runtime-label").textContent = `${health.retriever} · ${health.reranker} · ${health.generator}`;
  $("#source-count").textContent = health.sources;
  $("#chunk-count").textContent = health.chunks;
  $("#retriever-mode").textContent = health.retriever;
}

function renderSuggestions(items) {
  const sourceIds = new Set(items.map((item) => item.source_id));
  const sampleQuestions = PUBLIC_SAMPLE_QUESTIONS.filter((item) =>
    sourceIds.has(item.sourceId)
  );
  if (sampleQuestions.length < 2) {
    return;
  }
  $$(".suggestions button").forEach((button, index) => {
    const suggestion = sampleQuestions[index];
    button.hidden = !suggestion;
    if (suggestion) {
      button.textContent = suggestion.label;
      button.dataset.question = suggestion.question;
    }
  });
}

function renderCorpus() {
  const corpus = state.corpus;
  $("#component-count").textContent = Object.keys(corpus.components).length;
  $("#component-list").innerHTML = Object.entries(corpus.components)
    .map(
      ([component, count]) => `
        <div class="component-row">
          <span>${escapeHtml(component)}</span>
          <span>${count}</span>
        </div>
      `,
    )
    .join("");
  renderSourceRows(corpus.items);
  $("#source-filter").addEventListener("input", (event) => {
    const query = event.target.value.trim().toLowerCase();
    renderSourceRows(
      corpus.items.filter((item) =>
        [item.title, item.source_id, item.component]
          .join(" ")
          .toLowerCase()
          .includes(query),
      ),
    );
  });
}

function renderSourceRows(items) {
  $("#source-visible-count").textContent = `${items.length} 项`;
  $("#source-table").innerHTML = items
    .map(
      (item) => `
        <div class="source-row">
          <strong title="${escapeAttribute(item.title)}">${escapeHtml(item.title)}</strong>
          <span title="${escapeAttribute(item.source_id)}">${escapeHtml(item.component)} · ${escapeHtml(item.source_id)}</span>
          <span class="source-kind">${escapeHtml(item.source_type)}</span>
        </div>
      `,
    )
    .join("");
}

function renderEvaluations() {
  const grid = $("#evaluation-grid");
  if (!state.evaluations.length) {
    grid.innerHTML = '<div class="evaluation-item">暂无评测报告</div>';
    return;
  }
  grid.innerHTML = state.evaluations
    .map(
      (report) => `
        <article class="evaluation-item">
          <header>
            <h2>${escapeHtml(report.name)}</h2>
            <span>${escapeHtml(report.kind)} · ${report.question_count} questions</span>
          </header>
          <div class="metric-grid">
            ${Object.entries(report.metrics)
              .filter(([, value]) => value !== null)
              .map(
                ([key, value]) => `
                  <div>
                    <span>${escapeHtml(key)}</span>
                    <strong>${formatMetric(key, value)}</strong>
                  </div>
                `,
              )
              .join("")}
          </div>
        </article>
      `,
    )
    .join("");
}

function bindUpload() {
  const dialog = $("#upload-dialog");
  $("#open-upload").addEventListener("click", () => dialog.showModal());
  $("#close-upload").addEventListener("click", () => dialog.close());
  $("#cancel-upload").addEventListener("click", () => dialog.close());
  $("#upload-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const status = $("#upload-status");
    const file = $("#upload-file").files[0];
    if (!file) return;
    const body = new FormData();
    body.append("file", file);
    body.append("component", $("#upload-component").value.trim() || "general");
    const sourceId = $("#upload-source-id").value.trim();
    if (sourceId) body.append("source_id", sourceId);
    status.className = "form-status";
    status.textContent = "正在解析并重建索引...";
    try {
      const result = await fetchJson("/v1/corpus/upload", {
        method: "POST",
        body,
      });
      status.textContent = `已导入 ${result.title}，当前 ${result.sources} 篇文档。`;
      await refreshWorkspace();
      setTimeout(() => dialog.close(), 700);
    } catch (error) {
      status.className = "form-status error";
      status.textContent = readableError(error);
    }
  });
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `${response.status} ${response.statusText}`);
  }
  return payload;
}

function setEvidenceStatus(label, kind) {
  const element = $("#evidence-status");
  element.className = `badge ${kind}`;
  element.textContent = label;
}

function formatMetric(key, value) {
  if (key.endsWith("_ms")) return `${Number(value).toFixed(0)} ms`;
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatLatency(value) {
  return `${Number(value).toFixed(0)} ms`;
}

function safeSourceUrl(value) {
  return /^https?:\/\//.test(value) ? value : "";
}

function currentTime() {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date());
}

function readableError(error) {
  return error instanceof Error ? error.message : String(error);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return escapeHtml(value);
}
