/**
 * DeepTrace Background Service Worker (Manifest V3)
 * Handles context menu registration, API calls, and notifications.
 */

const API_BASE = "http://localhost:8000/api/v1";
const CONTEXT_MENU_ID = "deeptrace-check-image";

// ---------------------------------------------------------------------------
// Context menu setup
// ---------------------------------------------------------------------------

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: CONTEXT_MENU_ID,
    title: "Check with DeepTrace",
    contexts: ["image"],
  });
  console.log("[DeepTrace] Extension installed. Context menu created.");
});

// ---------------------------------------------------------------------------
// Context menu click handler
// ---------------------------------------------------------------------------

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== CONTEXT_MENU_ID) return;

  // Get the actual src from content script (handles lazy-loaded images)
  let imageSrc = info.srcUrl;

  try {
    const response = await chrome.tabs.sendMessage(tab.id, { action: "GET_IMAGE_SRC" });
    if (response?.src) imageSrc = response.src;
  } catch (_) {
    // content script may not be loaded; fall back to info.srcUrl
  }

  if (!imageSrc) {
    showNotification("No image found", "Could not extract image URL from this element.");
    return;
  }

  // Show "analyzing" notification
  showNotification("Analyzing image…", imageSrc.substring(0, 60) + "...");

  try {
    const result = await analyzeImage(imageSrc);
    const label = SOURCE_LABELS[result.predicted_source] || result.predicted_source;
    const conf = (result.confidence * 100).toFixed(1);

    showNotification(
      `DeepTrace: ${label}`,
      `${conf}% confidence · ${result.is_ai_generated ? "AI Generated ⚡" : "Real Photo ✓"}`
    );

    // Show inline badge on the page
    try {
      await chrome.tabs.sendMessage(tab.id, {
        action: "SHOW_RESULT_BADGE",
        result,
      });
    } catch (_) {}

  } catch (err) {
    console.error("[DeepTrace] Analysis failed:", err);
    showNotification(
      "DeepTrace: Analysis failed",
      err.message || "Could not connect to the DeepTrace API."
    );
  }
});

// ---------------------------------------------------------------------------
// API call
// ---------------------------------------------------------------------------

async function analyzeImage(imageSrc) {
  const { apiKey } = await chrome.storage.sync.get({ apiKey: "dev-key-123" });

  // Fetch the image as a blob
  const imgResponse = await fetch(imageSrc);
  if (!imgResponse.ok) throw new Error(`Failed to fetch image: ${imgResponse.status}`);

  const blob = await imgResponse.blob();

  // Convert to supported MIME type if needed
  const mimeType = blob.type.startsWith("image/") ? blob.type : "image/jpeg";

  const formData = new FormData();
  formData.append("file", blob, "image.jpg");

  const response = await fetch(`${API_BASE}/predict?explain=true`, {
    method: "POST",
    headers: { "X-API-Key": apiKey },
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  return response.json();
}

// ---------------------------------------------------------------------------
// Notification helper
// ---------------------------------------------------------------------------

function showNotification(title, message) {
  chrome.notifications.create({
    type: "basic",
    iconUrl: "icons/icon48.png",
    title,
    message,
    priority: 1,
  });
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SOURCE_LABELS = {
  stable_diffusion: "Stable Diffusion",
  midjourney: "Midjourney",
  dalle3: "DALL·E 3",
  flux: "Flux",
  real: "Real Photo",
};

// ---------------------------------------------------------------------------
// Popup message handler (settings)
// ---------------------------------------------------------------------------

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.action === "GET_SETTINGS") {
    chrome.storage.sync.get(
      { apiKey: "dev-key-123", apiBase: API_BASE },
      (settings) => sendResponse(settings)
    );
    return true;
  }
  if (message.action === "SAVE_SETTINGS") {
    chrome.storage.sync.set(message.settings, () => sendResponse({ ok: true }));
    return true;
  }
});
