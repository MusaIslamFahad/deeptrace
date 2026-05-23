/**
 * DeepTrace Popup Script
 * Loads/saves settings, checks API connectivity.
 */

const apiKeyInput  = document.getElementById("api-key");
const apiBaseInput = document.getElementById("api-base");
const saveBtn      = document.getElementById("save-btn");
const testBtn      = document.getElementById("test-btn");
const apiDot       = document.getElementById("api-dot");
const apiStatus    = document.getElementById("api-status");
const saveFeedback = document.getElementById("save-feedback");

// ---------------------------------------------------------------------------
// Load stored settings
// ---------------------------------------------------------------------------

chrome.runtime.sendMessage({ action: "GET_SETTINGS" }, (settings) => {
  if (settings) {
    apiKeyInput.value  = settings.apiKey  || "";
    apiBaseInput.value = settings.apiBase || "http://localhost:8000";
  }
  checkConnection();
});

// ---------------------------------------------------------------------------
// Save settings
// ---------------------------------------------------------------------------

saveBtn.addEventListener("click", () => {
  const settings = {
    apiKey:  apiKeyInput.value.trim(),
    apiBase: apiBaseInput.value.trim() || "http://localhost:8000",
  };

  chrome.runtime.sendMessage({ action: "SAVE_SETTINGS", settings }, () => {
    saveFeedback.style.display = "block";
    setTimeout(() => { saveFeedback.style.display = "none"; }, 2000);
    checkConnection();
  });
});

// ---------------------------------------------------------------------------
// Test connection
// ---------------------------------------------------------------------------

testBtn.addEventListener("click", checkConnection);

async function checkConnection() {
  setStatus("checking", "Connecting…");
  const base = apiBaseInput.value.trim() || "http://localhost:8000";

  try {
    const res = await fetch(`${base}/health`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (data.status === "healthy") {
      setStatus("green", `Connected · v${data.version}`);
    } else if (data.status === "degraded") {
      setStatus("yellow", `Degraded · ${data.model_loaded ? "model ok" : "model error"}`);
    } else {
      setStatus("red", `Unhealthy — check server`);
    }
  } catch (err) {
    setStatus("red", `Offline — ${err.message}`);
  }
}

function setStatus(state, text) {
  apiDot.className = "dot";
  if (state === "green")    apiDot.classList.add("green");
  if (state === "yellow")   apiDot.classList.add("yellow");
  if (state === "red")      apiDot.classList.add("red");
  apiStatus.textContent = text;
}
