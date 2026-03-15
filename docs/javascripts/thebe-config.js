// Live code execution for mkdocs-jupyter notebook pages
// Connects to FUES Jupyter server on Digital Ocean droplet
// No dependency on thebelab — uses Jupyter REST + WebSocket API

(function () {
  var JUPYTER_URL = "https://key-literary-age-valuable.trycloudflare.com";
  var TOKEN = "fues-thebe-2026";
  var WS_URL = "wss://key-literary-age-valuable.trycloudflare.com";

  var kernel = null;
  var ws = null;

  function init() {
    if (document.getElementById("thebe-banner")) return;
    var wrapper = document.querySelector(".jupyter-wrapper");
    if (!wrapper) return;

    var banner = document.createElement("div");
    banner.id = "thebe-banner";
    banner.style.cssText =
      "margin:0 0 1.2rem 0;display:flex;align-items:center;" +
      "gap:8px;font-size:12px;color:#78909c;";

    var btn = document.createElement("button");
    btn.id = "thebe-launch-btn";
    btn.textContent = "\u25B6 Launch Live";
    btn.style.cssText =
      "background:none;color:#00897b;border:1px solid #00897b;" +
      "padding:3px 10px;border-radius:3px;cursor:pointer;" +
      "font-weight:600;font-size:12px;";

    var msg = document.createElement("span");
    msg.id = "thebe-msg";
    msg.textContent = "";

    banner.appendChild(btn);
    banner.appendChild(msg);

    var h1 = wrapper.querySelector("h1");
    if (h1 && h1.nextSibling) {
      h1.parentNode.insertBefore(banner, h1.nextSibling);
    } else {
      var article = document.querySelector("article") || wrapper;
      article.insertBefore(banner, article.firstChild);
    }

    btn.onclick = function () {
      btn.textContent = "Connecting\u2026";
      btn.disabled = true;
      startKernel(btn, msg);
    };
  }

  function getCodeCells() {
    return Array.from(document.querySelectorAll(".jp-Cell.jp-CodeCell"));
  }

  function autoSize(ta) {
    ta.style.height = "auto";
    ta.style.height = ta.scrollHeight + "px";
  }

  function ensureOutputArea(cell) {
    var outputArea = cell.querySelector(".jp-OutputArea");
    if (outputArea) return outputArea;
    var outputWrapper = cell.querySelector(".jp-Cell-outputWrapper");
    if (!outputWrapper) {
      outputWrapper = document.createElement("div");
      outputWrapper.className = "jp-Cell-outputWrapper";
      cell.appendChild(outputWrapper);
    }
    outputArea = document.createElement("div");
    outputArea.className = "jp-OutputArea jp-Cell-outputArea";
    outputWrapper.appendChild(outputArea);
    cell.classList.remove("jp-mod-noOutputs");
    return outputArea;
  }

  function activateCells() {
    var cells = getCodeCells();
    cells.forEach(function (cell, idx) {
      var pre = cell.querySelector(".highlight-ipynb pre");
      if (!pre) return;
      var code = pre.textContent;

      var ta = document.createElement("textarea");
      ta.className = "thebe-input";
      ta.value = code;
      ta.style.cssText =
        "width:100%;box-sizing:border-box;font-family:'JetBrains Mono'," +
        "monospace;font-size:13px;padding:8px;border:1px solid " +
        "#00897b;border-radius:4px;background:#263238;color:#eee;" +
        "resize:vertical;tab-size:4;line-height:1.5;overflow:hidden;";
      ta.spellcheck = false;

      ta.addEventListener("keydown", function (e) {
        if (e.key === "Tab") {
          e.preventDefault();
          var s = this.selectionStart;
          var end = this.selectionEnd;
          this.value = this.value.substring(0, s) + "    " + this.value.substring(end);
          this.selectionStart = this.selectionEnd = s + 4;
        }
      });
      ta.addEventListener("input", function () { autoSize(this); });

      var editorDiv = cell.querySelector(".jp-CodeMirrorEditor");
      if (editorDiv) {
        editorDiv.innerHTML = "";
        editorDiv.appendChild(ta);
      }
      autoSize(ta);

      var runBtn = document.createElement("button");
      runBtn.textContent = "\u25B6 Run";
      runBtn.className = "thebe-run-btn";
      runBtn.style.cssText =
        "background:#00897b;color:#fff;border:none;" +
        "padding:4px 12px;border-radius:3px;cursor:pointer;" +
        "font-size:12px;font-weight:600;margin:4px 0;";
      runBtn.onclick = function () { executeCell(ta.value, cell); };

      var inputArea = cell.querySelector(".jp-InputArea");
      if (inputArea) inputArea.appendChild(runBtn);
    });
  }

  function startKernel(btn, msg) {
    fetch(JUPYTER_URL + "/api/kernels?token=" + TOKEN, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: "fues" }),
    })
      .then(function (r) {
        if (!r.ok) throw new Error("Kernel start failed: " + r.status);
        return r.json();
      })
      .then(function (data) {
        kernel = data;
        connectWebSocket(data.id, btn, msg);
      })
      .catch(function (err) {
        msg.textContent = "Connection failed: " + err.message;
        btn.textContent = "Retry";
        btn.disabled = false;
      });
  }

  function connectWebSocket(kernelId, btn, msg) {
    var wsUrl = WS_URL + "/api/kernels/" + kernelId + "/channels?token=" + TOKEN;
    ws = new WebSocket(wsUrl);

    ws.onopen = function () {
      btn.textContent = "\u2713 Live";
      btn.style.background = "#e8f5e9";
      btn.style.color = "#2e7d32";
      msg.textContent = "Kernel running. Edit cells and click \u25B6 Run.";
      activateCells();
    };
    ws.onmessage = function (event) { handleMessage(JSON.parse(event.data)); };
    ws.onerror = function () { msg.textContent = "WebSocket error."; };
    ws.onclose = function () {
      msg.textContent = "Kernel disconnected.";
      btn.textContent = "Reconnect";
      btn.style.background = "#fff";
      btn.style.color = "#00897b";
      btn.disabled = false;
      btn.onclick = function () {
        btn.textContent = "Connecting\u2026";
        btn.disabled = true;
        startKernel(btn, msg);
      };
    };
  }

  var pendingCells = {};

  function executeCell(code, cellEl) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      alert("Kernel not connected. Click Launch Live first.");
      return;
    }
    var outputArea = ensureOutputArea(cellEl);
    outputArea.innerHTML = "";
    var msgId = "exec_" + Math.random().toString(36).substr(2, 9);
    pendingCells[msgId] = outputArea;

    ws.send(JSON.stringify({
      header: { msg_id: msgId, msg_type: "execute_request", username: "", session: "", version: "5.3" },
      parent_header: {},
      metadata: {},
      content: { code: code, silent: false, store_history: true, user_expressions: {}, allow_stdin: false, stop_on_error: true },
      channel: "shell",
    }));

    var ri = document.createElement("div");
    ri.className = "thebe-running";
    ri.textContent = "Running\u2026";
    ri.style.cssText = "color:#00897b;font-style:italic;padding:4px 8px;font-size:12px;";
    outputArea.appendChild(ri);
  }

  function handleMessage(message) {
    var parentId = message.parent_header && message.parent_header.msg_id;
    var outputArea = pendingCells[parentId];
    if (!outputArea) return;
    var msgType = message.msg_type || message.header.msg_type;
    var running = outputArea.querySelector(".thebe-running");

    if (msgType === "stream") {
      if (running) running.remove();
      var pre = document.createElement("pre");
      pre.style.cssText = "margin:2px 8px;font-size:13px;line-height:1.4;white-space:pre-wrap;color:#cfd8dc;";
      pre.textContent = message.content.text;
      outputArea.appendChild(pre);
    } else if (msgType === "execute_result" || msgType === "display_data") {
      if (running) running.remove();
      var content = message.content.data;
      if (content["image/png"]) {
        var img = document.createElement("img");
        img.src = "data:image/png;base64," + content["image/png"];
        img.style.maxWidth = "100%";
        outputArea.appendChild(img);
      } else if (content["text/html"]) {
        var div = document.createElement("div");
        div.innerHTML = content["text/html"];
        outputArea.appendChild(div);
        div.querySelectorAll("script").forEach(function (oldScript) {
          var newScript = document.createElement("script");
          if (oldScript.src) { newScript.src = oldScript.src; }
          else { newScript.textContent = oldScript.textContent; }
          oldScript.parentNode.replaceChild(newScript, oldScript);
        });
      } else if (content["text/plain"]) {
        var pre = document.createElement("pre");
        pre.style.cssText = "margin:2px 8px;font-size:13px;color:#cfd8dc;";
        pre.textContent = content["text/plain"];
        outputArea.appendChild(pre);
      }
    } else if (msgType === "error") {
      if (running) running.remove();
      var pre = document.createElement("pre");
      pre.style.cssText = "margin:2px 8px;font-size:13px;color:#ef5350;white-space:pre-wrap;";
      pre.textContent = (message.content.traceback || []).join("\n").replace(/\x1b\[[0-9;]*m/g, "");
      outputArea.appendChild(pre);
    } else if (msgType === "status") {
      if (message.content.execution_state === "idle" && running) running.remove();
    }
  }

  if (document.readyState === "complete" || document.readyState === "interactive") init();
  document.addEventListener("DOMContentLoaded", function () { init(); });
  if (typeof document$ !== "undefined") {
    document$.subscribe(function () { init(); });
  }
})();
