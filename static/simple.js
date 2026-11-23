import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const FINGER_3BONE_CONNECTIONS = [
  [1, 2], [2, 3], [3, 4],
  [5, 6], [6, 7], [7, 8],
  [9, 10], [10, 11], [11, 12],
  [13, 14], [14, 15], [15, 16],
  [17, 18], [18, 19], [19, 20],
];

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

// Drawer & Theme DOM
const leftDrawer = document.getElementById("leftDrawer");
const drawerBackdrop = document.getElementById("drawerBackdrop");
const btnOpenDrawer = document.getElementById("btnOpenDrawer");
const btnCloseDrawer = document.getElementById("btnCloseDrawer");
const btnThemeToggle = document.getElementById("btnThemeToggle");
const btnQuickTheme = document.getElementById("btnQuickTheme");

// Guide Modal DOM
const guideModal = document.getElementById("guideModal");
const btnZoomGuide = document.getElementById("btnZoomGuide");
const guideImg = document.getElementById("guideImg");
const btnCloseGuideModal = document.getElementById("btnCloseGuideModal");

// Camera & Prediction DOM
const webcamEl = document.getElementById("webcam");
const canvasEl = document.getElementById("overlayCanvas");
const ctx = canvasEl.getContext("2d");
const camPlaceholder = document.getElementById("camPlaceholder");
const btnToggleCam = document.getElementById("btnToggleCam");
const chkMirror = document.getElementById("chkMirror");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const activeModelInfo = document.getElementById("activeModelInfo");
const handStateText = document.getElementById("handStateText");

const livePredChar = document.getElementById("livePredChar");
const livePredConf = document.getElementById("livePredConf");
const confBar = document.getElementById("confBar");
const holdProgressBar = document.getElementById("holdProgressBar");
const holdStatusText = document.getElementById("holdStatusText");

// Sign-to-Text DOM
const sentenceOutput = document.getElementById("sentenceOutput");
const charCountInfo = document.getElementById("charCountInfo");
const btnAddCurrent = document.getElementById("btnAddCurrent");
const btnAddSpace = document.getElementById("btnAddSpace");
const btnBackspace = document.getElementById("btnBackspace");
const btnSpeak = document.getElementById("btnSpeak");
const btnClearText = document.getElementById("btnClearText");

// Settings DOM
const chkAutoHold = document.getElementById("chkAutoHold");
const rngHoldMs = document.getElementById("rngHoldMs");
const lblHoldMs = document.getElementById("lblHoldMs");
const rngConfThresh = document.getElementById("rngConfThresh");
const lblConfThresh = document.getElementById("lblConfThresh");
const selCropMode = document.getElementById("selCropMode");

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");
const resnetInputCanvas = document.createElement("canvas");
resnetInputCanvas.width = 224;
resnetInputCanvas.height = 224;
const resnetInputCtx = resnetInputCanvas.getContext("2d");

let handLandmarker = null;
let ws = null;
let isCamRunning = false;
let animFrameId = null;

let lastInferenceSentAt = 0;
let awaitingWsResponse = false;
let currentHandCount = 0;

let smoothedProbs = {};
const EMA_ALPHA = 0.55;
let latestPrediction = null;
let latestConfidence = 0;

let candidateLabel = null;
let candidateHoldStart = 0;
let cooldownUntil = 0;

// 1. Drawer Open / Close
function openDrawer() {
  leftDrawer.classList.remove("-translate-x-full");
  drawerBackdrop.classList.remove("hidden");
}

function closeDrawer() {
  leftDrawer.classList.add("-translate-x-full");
  drawerBackdrop.classList.add("hidden");
}

btnOpenDrawer.addEventListener("click", openDrawer);
btnCloseDrawer.addEventListener("click", closeDrawer);
drawerBackdrop.addEventListener("click", closeDrawer);

// 2. Light / Dark Mode (Default: Light Mode)
function applyTheme(isDark) {
  if (isDark) {
    document.documentElement.classList.add("dark");
    btnThemeToggle.textContent = "Mode: Gelap";
  } else {
    document.documentElement.classList.remove("dark");
    btnThemeToggle.textContent = "Mode: Terang";
  }
  localStorage.setItem("bisindo_theme", isDark ? "dark" : "light");
}

function toggleTheme() {
  const isDark = !document.documentElement.classList.contains("dark");
  applyTheme(isDark);
}

// Inisialisasi tema (default terang)
const savedTheme = localStorage.getItem("bisindo_theme");
applyTheme(savedTheme === "dark");

btnThemeToggle.addEventListener("click", toggleTheme);
btnQuickTheme.addEventListener("click", toggleTheme);

// 3. Guide Modal Zoom
function openGuideModal() {
  guideModal.classList.remove("hidden");
  guideModal.classList.add("flex");
}
function closeGuideModal() {
  guideModal.classList.add("hidden");
  guideModal.classList.remove("flex");
}
btnZoomGuide.addEventListener("click", openGuideModal);
guideImg.addEventListener("click", openGuideModal);
btnCloseGuideModal.addEventListener("click", closeGuideModal);
guideModal.addEventListener("click", (e) => {
  if (e.target === guideModal) closeGuideModal();
});

// 4. MediaPipe & WebSocket
async function initMediaPipe() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "/models/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.35,
      minHandPresenceConfidence: 0.35,
      minTrackingConfidence: 0.35,
    });
    updateStatus();
  } catch (err) {
    console.error("MediaPipe Error:", err);
    statusDot.className = "w-2 h-2 rounded-full bg-red-600";
    statusText.textContent = "Gagal memuat tracker";
  }
}

function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${window.location.host}/ws/predict`);

  ws.onopen = () => updateStatus();
  ws.onclose = () => {
    awaitingWsResponse = false;
    updateStatus();
    setTimeout(initWebSocket, 2000);
  };
  ws.onmessage = (event) => {
    awaitingWsResponse = false;
    const data = JSON.parse(event.data);
    if (!data.error) {
      handlePrediction(data);
    }
  };
}

function updateStatus() {
  const wsReady = ws && ws.readyState === WebSocket.OPEN;
  const mpReady = Boolean(handLandmarker);
  if (wsReady && mpReady) {
    statusDot.className = "w-2 h-2 rounded-full bg-emerald-600";
    statusText.textContent = "Aktif";
  } else {
    statusDot.className = "w-2 h-2 rounded-full bg-amber-500";
    statusText.textContent = "Menghubungkan...";
  }
}

function handlePrediction(data) {
  if (currentHandCount === 0) return;

  if (data.all_probs) {
    const keys = Object.keys(data.all_probs);
    if (Object.keys(smoothedProbs).length === 0) {
      smoothedProbs = { ...data.all_probs };
    } else {
      keys.forEach((k) => {
        smoothedProbs[k] =
          EMA_ALPHA * data.all_probs[k] + (1 - EMA_ALPHA) * (smoothedProbs[k] || 0);
      });
    }
    const sorted = keys
      .map((k) => ({ label: k, confidence: smoothedProbs[k] }))
      .sort((a, b) => b.confidence - a.confidence);
    latestPrediction = sorted[0].label;
    latestConfidence = sorted[0].confidence;
  } else {
    latestPrediction = data.prediction;
    latestConfidence = data.confidence;
  }

  const pct = (latestConfidence * 100).toFixed(1);
  livePredChar.textContent = latestPrediction;
  livePredConf.textContent = `${pct}%`;
  confBar.style.width = `${Math.min(100, latestConfidence * 100)}%`;
}

function updateCharCount() {
  charCountInfo.textContent = `${sentenceOutput.value.length} karakter`;
}

function updateHoldToType(now) {
  if (!chkAutoHold.checked) {
    holdProgressBar.style.width = "0%";
    holdStatusText.textContent = "Manual";
    return;
  }

  const confThreshold = Number(rngConfThresh.value) / 100;
  const requiredHoldMs = Number(rngHoldMs.value);

  if (now < cooldownUntil) {
    holdStatusText.textContent = "Terketik ✓";
    holdProgressBar.style.width = "100%";
    return;
  }

  if (
    currentHandCount === 0 ||
    !latestPrediction ||
    latestConfidence < confThreshold
  ) {
    candidateLabel = null;
    candidateHoldStart = 0;
    holdProgressBar.style.width = "0%";
    holdStatusText.textContent = "0%";
    return;
  }

  if (candidateLabel !== latestPrediction) {
    candidateLabel = latestPrediction;
    candidateHoldStart = now;
    holdProgressBar.style.width = "5%";
    holdStatusText.textContent = "5%";
    return;
  }

  const elapsed = now - candidateHoldStart;
  const progressPct = Math.min(100, (elapsed / requiredHoldMs) * 100);
  holdProgressBar.style.width = `${progressPct.toFixed(0)}%`;
  holdStatusText.textContent = `${Math.round(progressPct)}%`;

  if (elapsed >= requiredHoldMs) {
    sentenceOutput.value += candidateLabel;
    updateCharCount();
    cooldownUntil = now + 750;
    candidateLabel = null;
    candidateHoldStart = 0;
  }
}

function computeCropBox(rawLandmarksList, width, height) {
  const mode = selCropMode.value;
  if (mode === "full_frame") {
    return { x: 0, y: 0, w: width, h: height };
  }
  if (mode === "center_square" || !rawLandmarksList || rawLandmarksList.length === 0) {
    const side = Math.min(width, height);
    return {
      x: Math.round((width - side) / 2),
      y: Math.round((height - side) / 2),
      w: side,
      h: side,
    };
  }

  let minX = width,
    maxX = 0,
    minY = height,
    maxY = 0;

  rawLandmarksList.forEach((hand) => {
    hand.forEach((lm) => {
      const px = lm.x * width;
      const py = lm.y * height;
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
    });
  });

  const maxSpan = Math.max(40, maxX - minX, maxY - minY);
  let side = Math.round(maxSpan * 1.7);
  side = Math.min(Math.min(width, height), Math.max(160, side));

  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;

  const x0 = Math.max(0, Math.min(width - side, Math.round(cx - side / 2)));
  const y0 = Math.max(0, Math.min(height - side, Math.round(cy - side / 2)));

  return { x: x0, y: y0, w: side, h: side };
}

function drawHands(results, width, height, isMirrored) {
  if (!results || !results.landmarks) return;
  const is3BonesOnly = selCropMode.value === "finger_3bones";
  const connections = is3BonesOnly ? FINGER_3BONE_CONNECTIONS : HAND_CONNECTIONS;

  results.landmarks.forEach((rawLandmarks) => {
    const points = rawLandmarks.map((lm) => {
      const normX = isMirrored ? 1 - lm.x : lm.x;
      return {
        x: Math.min(width - 1, Math.max(0, Math.round(normX * width))),
        y: Math.min(height - 1, Math.max(0, Math.round(lm.y * height))),
      };
    });

    const activePoints = is3BonesOnly ? points.slice(1) : points;
    const xs = activePoints.map((p) => p.x);
    const ys = activePoints.map((p) => p.y);
    const pad = 16;
    const xMin = Math.max(0, Math.min(...xs) - pad);
    const xMax = Math.min(width - 1, Math.max(...xs) + pad);
    const yMin = Math.max(0, Math.min(...ys) - pad);
    const yMax = Math.min(height - 1, Math.max(...ys) + pad);

    ctx.strokeStyle = "#16a34a";
    ctx.lineWidth = 2;
    ctx.strokeRect(xMin, yMin, xMax - xMin, yMax - yMin);

    ctx.strokeStyle = "#22c55e";
    ctx.lineWidth = 2;
    connections.forEach(([start, end]) => {
      ctx.beginPath();
      ctx.moveTo(points[start].x, points[start].y);
      ctx.lineTo(points[end].x, points[end].y);
      ctx.stroke();
    });

    ctx.fillStyle = "#dc2626";
    activePoints.forEach((pt) => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    });
  });
}

function renderLoop() {
  if (!isCamRunning) return;
  const now = performance.now();

  if (webcamEl.readyState >= 2) {
    const width = webcamEl.videoWidth || 640;
    const height = webcamEl.videoHeight || 480;

    if (canvasEl.width !== width || canvasEl.height !== height) {
      canvasEl.width = width;
      canvasEl.height = height;
      captureCanvas.width = width;
      captureCanvas.height = height;
    }

    captureCtx.drawImage(webcamEl, 0, 0, width, height);

    const isMirrored = chkMirror.checked;
    ctx.save();
    if (isMirrored) {
      ctx.translate(width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(webcamEl, 0, 0, width, height);
    ctx.restore();

    let results = null;
    if (handLandmarker) {
      results = handLandmarker.detectForVideo(captureCanvas, now);
    }

    currentHandCount = results?.landmarks?.length || 0;

    if (currentHandCount > 0) {
      handStateText.textContent = `${currentHandCount} tangan terdeteksi`;
      drawHands(results, width, height, isMirrored);

      if (
        ws &&
        ws.readyState === WebSocket.OPEN &&
        !awaitingWsResponse &&
        now - lastInferenceSentAt >= 120
      ) {
        awaitingWsResponse = true;
        lastInferenceSentAt = now;

        if (selCropMode.value === "finger_3bones") {
          ws.send(
            JSON.stringify({
              model_type: "finger_3bones",
              landmarks: results.landmarks,
              client_ts: now,
            })
          );
        } else {
          const cropBox = computeCropBox(results.landmarks, width, height);
          resnetInputCtx.drawImage(
            captureCanvas,
            cropBox.x,
            cropBox.y,
            cropBox.w,
            cropBox.h,
            0,
            0,
            224,
            224
          );
          ws.send(
            JSON.stringify({
              model_type: "resnet50",
              image: resnetInputCanvas.toDataURL("image/jpeg", 0.88),
              client_ts: now,
            })
          );
        }
      }
    } else {
      handStateText.textContent = "Menunggu tangan";
      smoothedProbs = {};
      latestPrediction = null;
      latestConfidence = 0;
      livePredChar.textContent = "-";
      livePredConf.textContent = "0%";
      confBar.style.width = "0%";
    }

    updateHoldToType(now);
  }

  animFrameId = requestAnimationFrame(renderLoop);
}

btnToggleCam.addEventListener("click", async () => {
  if (isCamRunning) {
    isCamRunning = false;
    if (animFrameId) cancelAnimationFrame(animFrameId);
    const stream = webcamEl.srcObject;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    webcamEl.srcObject = null;
    camPlaceholder.classList.remove("hidden");
    btnToggleCam.textContent = "Mulai Kamera";
    btnToggleCam.className =
      "px-3.5 py-1 rounded bg-emerald-600 hover:bg-emerald-700 text-white font-medium text-xs transition";
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
    webcamEl.srcObject = stream;
    await webcamEl.play();
    camPlaceholder.classList.add("hidden");
    isCamRunning = true;
    btnToggleCam.textContent = "Hentikan Kamera";
    btnToggleCam.className =
      "px-3.5 py-1 rounded bg-red-600 hover:bg-red-700 text-white font-medium text-xs transition";
    renderLoop();
  } catch (err) {
    alert("Tidak dapat mengakses kamera: " + err.message);
  }
});

btnAddCurrent.addEventListener("click", () => {
  if (latestPrediction) {
    sentenceOutput.value += latestPrediction;
    updateCharCount();
  }
});
btnAddSpace.addEventListener("click", () => {
  sentenceOutput.value += " ";
  updateCharCount();
});
btnBackspace.addEventListener("click", () => {
  sentenceOutput.value = sentenceOutput.value.slice(0, -1);
  updateCharCount();
});
btnClearText.addEventListener("click", () => {
  sentenceOutput.value = "";
  updateCharCount();
});
sentenceOutput.addEventListener("input", updateCharCount);

btnSpeak.addEventListener("click", () => {
  const txt = sentenceOutput.value.trim();
  if (!txt || !("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(txt);
  utter.lang = "id-ID";
  window.speechSynthesis.speak(utter);
});

rngHoldMs.addEventListener("input", () => {
  lblHoldMs.textContent = `${(Number(rngHoldMs.value) / 1000).toFixed(1)} dtk`;
});
rngConfThresh.addEventListener("input", () => {
  lblConfThresh.textContent = `${rngConfThresh.value}%`;
});
selCropMode.addEventListener("change", () => {
  smoothedProbs = {};
  const labelText = selCropMode.options[selCropMode.selectedIndex].text;
  activeModelInfo.textContent = `Model: ${labelText}`;
});

initMediaPipe();
initWebSocket();
