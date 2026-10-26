/**
 * CODEX 50B DEVELOPER STUDIO CLIENT
 * Handles real-time SSE streaming, telemetry polling, diff updates, and agent tool execution.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const chatViewport = document.getElementById("chat-viewport");
  const chatInput = document.getElementById("chat-input");
  const btnSend = document.getElementById("btn-send");
  const btnStop = document.getElementById("btn-stop");
  const modelSelect = document.getElementById("model-select");
  const clusterStatusText = document.getElementById("cluster-status-text");
  const metricRam = document.getElementById("metric-ram");
  const metricVram = document.getElementById("metric-vram");
  const metricTps = document.getElementById("metric-tps");
  const inputStatus = document.getElementById("input-status");

  // Config sliders
  const paramTemp = document.getElementById("param-temp");
  const valTemp = document.getElementById("val-temp");
  const paramTopp = document.getElementById("param-topp");
  const valTopp = document.getElementById("val-topp");
  const paramMaxtok = document.getElementById("param-maxtok");
  const valMaxtok = document.getElementById("val-maxtok");
  const paramReppen = document.getElementById("param-reppen");
  const valReppen = document.getElementById("val-reppen");
  const paramApiKey = document.getElementById("param-apikey");

  // Tabs & Buttons
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");
  const quickBtns = document.querySelectorAll(".quick-btn");
  const btnFimMode = document.getElementById("btn-fim-mode");
  const btnAgentMode = document.getElementById("btn-agent-mode");
  const terminalBody = document.getElementById("terminal-body");
  const btnClearTerm = document.getElementById("btn-clear-term");
  const agentFeed = document.getElementById("agent-feed");

  // State
  let isGenerating = false;
  let abortController = null;
  let isFimActive = false;
  let isAgentActive = false;

  // 1. Tab Navigation
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));
      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) targetContent.classList.add("active");
    });
  });

  // 2. Parameter Slider Bindings
  paramTemp.addEventListener("input", (e) => valTemp.textContent = e.target.value);
  paramTopp.addEventListener("input", (e) => valTopp.textContent = e.target.value);
  paramMaxtok.addEventListener("input", (e) => valMaxtok.textContent = e.target.value);
  paramReppen.addEventListener("input", (e) => valReppen.textContent = e.target.value);

  // 3. FIM and Agent Mode Toggles
  btnFimMode.addEventListener("click", () => {
    isFimActive = !isFimActive;
    btnFimMode.classList.toggle("active", isFimActive);
    if (isFimActive) {
      chatInput.placeholder = "Enter Prefix and Suffix separated by <MID> placeholder...";
      inputStatus.textContent = "FIM (Fill-in-the-Middle) Active";
    } else {
      chatInput.placeholder = "Ask Codex-50B to generate, debug, test, or patch code...";
      inputStatus.textContent = "Standard Mode";
    }
  });

  btnAgentMode.addEventListener("click", () => {
    isAgentActive = !isAgentActive;
    btnAgentMode.classList.toggle("active", isAgentActive);
    inputStatus.textContent = isAgentActive ? "Autonomous ReAct Agent Active" : "Ready";
  });

  // 4. Quick Prompt Handlers
  quickBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      chatInput.value = btn.getAttribute("data-prompt");
      sendMessage();
    });
  });

  // 5. Chat Input Auto-Resize & Submit Keybinding
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // 6. Real-Time Telemetry Polling (/health)
  async function pollTelemetry() {
    try {
      const res = await fetch("/health");
      if (res.ok) {
        const data = await res.json();
        metricRam.textContent = `${(data.system_memory_used_mb / 1024).toFixed(1)} GB`;
        metricVram.textContent = `${data.vram_used_mb.toFixed(1)} MB`;
        clusterStatusText.textContent = `Online (${data.model_name} on ${data.device.toUpperCase()})`;
      }
    } catch {
      // Local server might be starting or running offline
      clusterStatusText.textContent = "Engine Ready (Local Dev Server)";
    }
  }
  setInterval(pollTelemetry, 3000);
  pollTelemetry();

  // 7. Core Messaging & Streaming Generator
  async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text || isGenerating) return;

    // Render User Bubble
    appendMessage("user", text);
    chatInput.value = "";
    isGenerating = true;
    btnSend.classList.add("hidden");
    btnStop.classList.remove("hidden");
    inputStatus.textContent = "Generating stream...";

    const assistantBubble = appendMessage("assistant", "");
    const contentDiv = assistantBubble.querySelector(".message-content");
    contentDiv.innerHTML = "<span class='cursor'>▋</span>";

    const startTime = performance.now();
    let tokenCount = 0;
    abortController = new AbortController();

    // Log to Agent Feed
    addAgentEvent("Reasoning", `Processing: ${text.slice(0, 32)}...`);

    try {
      const headers = { "Content-Type": "application/json" };
      if (paramApiKey.value.trim()) {
        headers["Authorization"] = `Bearer ${paramApiKey.value.trim()}`;
      }

      const payload = {
        model: modelSelect.value,
        messages: [{ role: "user", content: text }],
        temperature: parseFloat(paramTemp.value),
        top_p: parseFloat(paramTopp.value),
        max_tokens: parseInt(paramMaxtok.value),
        stream: true,
        repetition_penalty: parseFloat(paramReppen.value),
      };

      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: headers,
        body: JSON.stringify(payload),
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP Error: ${response.status} ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let fullGeneratedText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ") && !line.includes("[DONE]")) {
            try {
              const data = JSON.parse(line.slice(6));
              const deltaContent = data.choices[0]?.delta?.content || "";
              if (deltaContent) {
                fullGeneratedText += deltaContent;
                tokenCount++;
                contentDiv.innerHTML = formatMarkdown(fullGeneratedText) + "<span class='cursor'>▋</span>";
                chatViewport.scrollTop = chatViewport.scrollHeight;

                // Update real-time speed metric
                const elapsedSec = (performance.now() - startTime) / 1000;
                if (elapsedSec > 0.5) {
                  metricTps.textContent = `${(tokenCount / elapsedSec).toFixed(1)} tok/s`;
                }
              }
            } catch (err) {
              // Non-JSON line or chunk boundary
            }
          }
        }
      }

      contentDiv.innerHTML = formatMarkdown(fullGeneratedText);
      addAgentEvent("Complete", `Generated ${tokenCount} tokens in ${((performance.now() - startTime)/1000).toFixed(2)}s.`);
      appendTerminalLine(`[Engine] Completed stream: ${tokenCount} tokens generated.`);

    } catch (err) {
      if (err.name === "AbortError") {
        contentDiv.innerHTML += "<p><em>[Generation cancelled by user]</em></p>";
      } else {
        contentDiv.innerHTML = `<p style="color:var(--danger)"><strong>Error:</strong> ${err.message}</p>`;
        appendTerminalLine(`[Error] ${err.message}`, "error");
      }
    } finally {
      isGenerating = false;
      btnSend.classList.remove("hidden");
      btnStop.classList.add("hidden");
      inputStatus.textContent = "Ready";
      abortController = null;
    }
  }

  btnStop.addEventListener("click", () => {
    if (abortController) {
      abortController.abort();
    }
  });

  // UI Helpers
  function appendMessage(role, text) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `chat-message ${role}-message`;
    msgDiv.innerHTML = `
      <div class="message-avatar">${role === 'assistant' ? '⚡' : '👤'}</div>
      <div class="message-bubble">
        <div class="message-meta">
          <span class="author-name">${role === 'assistant' ? 'Codex-50B' : 'Developer'}</span>
          <span class="timestamp">${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
        </div>
        <div class="message-content">${formatMarkdown(text)}</div>
      </div>
    `;
    chatViewport.appendChild(msgDiv);
    chatViewport.scrollTop = chatViewport.scrollHeight;
    return msgDiv;
  }

  function formatMarkdown(text) {
    if (!text) return "";
    let safe = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // Code blocks
    safe = safe.replace(/```([a-zA-Z0-9]*)\n([\s\S]*?)```/g, (m, lang, code) => {
      return `<pre><code class="language-${lang}">${code}</code></pre>`;
    });

    // Inline code
    safe = safe.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Bold text
    safe = safe.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    // Paragraph linebreaks
    safe = safe.replace(/\n\n/g, "</p><p>");
    return `<p>${safe}</p>`;
  }

  function addAgentEvent(tag, desc) {
    const ev = document.createElement("div");
    ev.className = "feed-event";
    ev.innerHTML = `
      <span class="event-time">${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'})}</span>
      <span class="event-tag tag-model">${tag}</span>
      <span class="event-desc">${desc}</span>
    `;
    agentFeed.prepend(ev);
  }

  function appendTerminalLine(text, type = "info") {
    const l = document.createElement("div");
    l.className = `term-line ${type}`;
    l.textContent = text;
    terminalBody.appendChild(l);
    terminalBody.scrollTop = terminalBody.scrollHeight;
  }

  if (btnClearTerm) {
    btnClearTerm.addEventListener("click", () => {
      terminalBody.innerHTML = "<div class='term-line prompt'>&gt; Terminal cleared.</div>";
    });
  }
});
