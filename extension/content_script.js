/**
 * DeepTrace Content Script
 * Injects context menu targeting on right-clicked images.
 * Communicates with background.js via chrome.runtime.sendMessage.
 */

let lastRightClickedImg = null;

// Track last right-clicked image element
document.addEventListener(
  "contextmenu",
  (event) => {
    const target = event.target;
    if (target && target.tagName === "IMG") {
      lastRightClickedImg = target;
    } else {
      // Check if click is inside a picture element with img
      const img = target.closest && target.closest("picture")?.querySelector("img");
      lastRightClickedImg = img || null;
    }
  },
  true
);

// Listen for messages from background.js
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.action === "GET_IMAGE_SRC") {
    if (lastRightClickedImg) {
      // Prefer the full-resolution src, fall back to currentSrc
      const src = lastRightClickedImg.src || lastRightClickedImg.currentSrc;
      sendResponse({ src: src || null });
    } else {
      sendResponse({ src: null });
    }
  }

  if (message.action === "SHOW_RESULT_BADGE") {
    showResultBadge(message.result, lastRightClickedImg);
    sendResponse({ ok: true });
  }

  return true; // keep channel open for async
});

// ---------------------------------------------------------------------------
// Result badge overlay
// ---------------------------------------------------------------------------

function showResultBadge(result, imgElement) {
  if (!imgElement) return;

  // Remove any existing badge
  const existing = document.getElementById("deeptrace-badge");
  if (existing) existing.remove();

  const { predicted_source, confidence, is_ai_generated } = result;
  const SOURCE_LABELS = {
    stable_diffusion: "Stable Diffusion",
    midjourney: "Midjourney",
    dalle3: "DALL·E 3",
    flux: "Flux",
    real: "Real Photo",
  };
  const SOURCE_COLORS = {
    stable_diffusion: "#7F77DD",
    midjourney: "#1D9E75",
    dalle3: "#D85A30",
    flux: "#378ADD",
    real: "#639922",
  };

  const badge = document.createElement("div");
  badge.id = "deeptrace-badge";

  const rect = imgElement.getBoundingClientRect();
  const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
  const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;

  Object.assign(badge.style, {
    position: "absolute",
    top: `${rect.top + scrollTop + 8}px`,
    left: `${rect.left + scrollLeft + 8}px`,
    zIndex: "2147483647",
    background: "rgba(0,0,0,0.85)",
    color: "#fff",
    borderRadius: "10px",
    padding: "10px 14px",
    fontSize: "13px",
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    boxShadow: "0 4px 20px rgba(0,0,0,0.3)",
    backdropFilter: "blur(8px)",
    maxWidth: "220px",
    lineHeight: "1.5",
    cursor: "pointer",
    border: `2px solid ${SOURCE_COLORS[predicted_source] || "#fff"}`,
  });

  badge.innerHTML = `
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
      <span style="
        background:${SOURCE_COLORS[predicted_source] || "#888"};
        color:#fff;border-radius:6px;padding:2px 8px;
        font-size:11px;font-weight:600;letter-spacing:0.03em;
      ">${SOURCE_LABELS[predicted_source] || predicted_source}</span>
      ${is_ai_generated
        ? '<span style="font-size:11px;color:#fbbf24;">⚡ AI Generated</span>'
        : '<span style="font-size:11px;color:#4ade80;">✓ Real</span>'}
    </div>
    <div style="font-size:20px;font-weight:700;color:${SOURCE_COLORS[predicted_source]};">
      ${(confidence * 100).toFixed(1)}%
    </div>
    <div style="font-size:11px;color:#aaa;">confidence · DeepTrace</div>
    <div style="font-size:10px;color:#666;margin-top:6px;">Click to dismiss</div>
  `;

  badge.addEventListener("click", () => badge.remove());

  // Auto-remove after 8 seconds
  setTimeout(() => badge?.remove(), 8000);

  document.body.appendChild(badge);
}
