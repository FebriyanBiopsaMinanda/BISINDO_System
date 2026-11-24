import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// 3 tulang jari (tanpa koneksi ke titik 0 / Wrist)
const FINGER_3BONE_CONNECTIONS = [
  [1, 2], [2, 3], [3, 4],
  [5, 6], [6, 7], [7, 8],
  [9, 10], [10, 11], [11, 12],
  [13, 14], [14, 15], [15, 16],
  [17, 18], [18, 19], [19, 20],
];

const FINGER_CHAINS = [
  [4, 3, 2, 1],
  [8, 7, 6, 5],
  [12, 11, 10, 9],
  [16, 15, 14, 13],
  [20, 19, 18, 17],
];

// Drawer & Theme DOM
const leftDrawer = document.getElementById("leftDrawer");
const drawerBackdrop = document.getElementById("drawerBackdrop");
const btnOpenDrawer = document.getElementById("btnOpenDrawer");
const btnCloseDrawer = document.getElementById("btnCloseDrawer");
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
const handStateText = document.getElementById("handStateText");

const livePredChar = document.getElementById("livePredChar");
const livePredConf = document.getElementById("livePredConf");
const confBar = document.getElementById("confBar");
const holdProgressBar = document.getElementById("holdProgressBar");
const holdStatusText = document.getElementById("holdStatusText");

// Sign-to-Text DOM
const sentenceOutput = document.getElementById("sentenceOutput");
const charCountInfo = document.getElementById("charCountInfo");
const btnAddSpace = document.getElementById("btnAddSpace");
const btnBackspace = document.getElementById("btnBackspace");
const btnSpeak = document.getElementById("btnSpeak");
const btnClearText = document.getElementById("btnClearText");

// Settings DOM
const rngHoldMs = document.getElementById("rngHoldMs");
const lblHoldMs = document.getElementById("lblHoldMs");
const rngConfThresh = document.getElementById("rngConfThresh");
const lblConfThresh = document.getElementById("lblConfThresh");

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

let handLandmarker = null;
let fingerModelData = null;
let isCamRunning = false;
let animFrameId = null;

let lastInferenceTime = 0;
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
  } else {
    document.documentElement.classList.remove("dark");
  }
  localStorage.setItem("bisindo_theme", isDark ? "dark" : "light");
}

function toggleTheme() {
  const isDark = !document.documentElement.classList.contains("dark");
  applyTheme(isDark);
}

const savedTheme = localStorage.getItem("bisindo_theme");
applyTheme(savedTheme === "dark");
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

// 4. Load MediaPipe & Exported 3-Bone Finger Model JSON (100% Client-Side for Vercel)
async function initModels() {
  try {
    const [modelRes, vision] = await Promise.all([
      fetch("/models/finger_model.json"),
      FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      ),
    ]);
    fingerModelData = await modelRes.json();

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
  } catch (err) {
    console.error("Initialization Error:", err);
  }
}

// 5. Client-Side 3-Bone Feature Extraction & MLP Forward Pass (Identical to Python train_finger_landmarks.py)
function extractSingleHandFeatures(landmarks) {
  const pts = landmarks.map((lm) => [lm.x, lm.y, lm.z || 0.0]);
  const mcpIndices = [1, 5, 9, 13, 17];
  const mcpCenter = [0, 0, 0];
  mcpIndices.forEach((idx) => {
    mcpCenter[0] += pts[idx][0] / 5;
    mcpCenter[1] += pts[idx][1] / 5;
    mcpCenter[2] += pts[idx][2] / 5;
  });

  // 20 titik jari (indeks 1..20, tanpa Wrist 0)
  const centered = [];
  let maxNorm2D = 1e-6;
  for (let i = 1; i <= 20; i++) {
    const cx = pts[i][0] - mcpCenter[0];
    const cy = pts[i][1] - mcpCenter[1];
    const cz = pts[i][2] - mcpCenter[2];
    centered.push([cx, cy, cz]);
    const norm2D = Math.hypot(cx, cy);
    if (norm2D > maxNorm2D) maxNorm2D = norm2D;
  }

  const scale = maxNorm2D + 1e-6;
  const normCoords = [];
  centered.forEach(([cx, cy, cz]) => {
    normCoords.push(cx / scale, cy / scale, cz / scale);
  });

  const boneFeatures = [];
  const angleFeatures = [];

  FINGER_CHAINS.forEach(([tip, j1, j2, base]) => {
    const b1 = [
      (pts[tip][0] - pts[j1][0]) / scale,
      (pts[tip][1] - pts[j1][1]) / scale,
      (pts[tip][2] - pts[j1][2]) / scale,
    ];
    const b2 = [
      (pts[j1][0] - pts[j2][0]) / scale,
      (pts[j1][1] - pts[j2][1]) / scale,
      (pts[j1][2] - pts[j2][2]) / scale,
    ];
    const b3 = [
      (pts[j2][0] - pts[base][0]) / scale,
      (pts[j2][1] - pts[base][1]) / scale,
      (pts[j2][2] - pts[base][2]) / scale,
    ];

    [b1, b2, b3].forEach((b) => {
      const len = Math.hypot(b[0], b[1], b[2]);
      boneFeatures.push(b[0], b[1], b[2], len);
    });

    [
      [b1, b2],
      [b2, b3],
    ].forEach(([va, vb]) => {
      const na = Math.hypot(va[0], va[1], va[2]) + 1e-6;
      const nb = Math.hypot(vb[0], vb[1], vb[2]) + 1e-6;
      const dot = va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2];
      const cosTheta = Math.max(-1.0, Math.min(1.0, dot / (na * nb)));
      angleFeatures.push(cosTheta);
    });
  });

  return [...normCoords, ...boneFeatures, ...angleFeatures]; // 130 features
}

function predictClientSide(handsLandmarks) {
  if (!fingerModelData || !handsLandmarks || handsLandmarks.length === 0) return;

  const handsSorted = [...handsLandmarks].sort((a, b) => {
    let meanA = 0,
      meanB = 0;
    for (let i = 1; i <= 20; i++) {
      meanA += a[i].x;
      meanB += b[i].x;
    }
    return meanA - meanB;
  });

  const featH1 = extractSingleHandFeatures(handsSorted[0]);
  let featH2 = new Array(130).fill(0);
  let interTipDists = new Array(25).fill(0);
  let handCountFlag = 1.0;

  if (handsSorted.length > 1) {
    featH2 = extractSingleHandFeatures(handsSorted[1]);
    const tips = [4, 8, 12, 16, 20];
    interTipDists = [];
    tips.forEach((t1) => {
      tips.forEach((t2) => {
        const dx = handsSorted[0][t1].x - handsSorted[1][t2].x;
        const dy = handsSorted[0][t1].y - handsSorted[1][t2].y;
        interTipDists.push(Math.hypot(dx, dy));
      });
    });
    handCountFlag = 2.0;
  }

  const rawFeat = [...featH1, ...featH2, ...interTipDists, handCountFlag];

  // StandardScaler transform
  const mean = fingerModelData.scaler_mean;
  const scale = fingerModelData.scaler_scale;
  let x = rawFeat.map((val, i) => (val - mean[i]) / (scale[i] + 1e-8));

  // Forward pass MLP layers (Dense + BatchNormalization)
  for (const layer of fingerModelData.layers) {
    if (layer.type === "dense") {
      const W = layer.W;
      const b = layer.b;
      const outDim = b.length;
      const nextX = new Array(outDim).fill(0);
      for (let j = 0; j < outDim; j++) {
        let sum = b[j];
        for (let i = 0; i < x.length; i++) {
          sum += x[i] * W[i][j];
        }
        nextX[j] = layer.activation === "relu" ? Math.max(0, sum) : sum;
      }
      x = nextX;
    } else if (layer.type === "bn") {
      const { gamma, beta, mean: bnMean, var: bnVar, eps } = layer;
      x = x.map(
        (val, i) =>
          (gamma[i] * (val - bnMean[i])) / Math.sqrt(bnVar[i] + eps) + beta[i]
      );
    }
  }

  // Softmax output
  const maxLogit = Math.max(...x);
  const exps = x.map((v) => Math.exp(v - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((v) => v / sumExps);

  const classes = fingerModelData.classes;
  if (Object.keys(smoothedProbs).length === 0) {
    classes.forEach((c, i) => {
      smoothedProbs[c] = probs[i];
    });
  } else {
    classes.forEach((c, i) => {
      smoothedProbs[c] = EMA_ALPHA * probs[i] + (1 - EMA_ALPHA) * (smoothedProbs[c] || 0);
    });
  }

  let bestClass = classes[0];
  let bestConf = 0;
  classes.forEach((c) => {
    if (smoothedProbs[c] > bestConf) {
      bestConf = smoothedProbs[c];
      bestClass = c;
    }
  });

  latestPrediction = bestClass;
  latestConfidence = bestConf;

  const pct = (latestConfidence * 100).toFixed(1);
  livePredChar.textContent = latestPrediction;
  livePredConf.textContent = `${pct}%`;
  confBar.style.width = `${Math.min(100, latestConfidence * 100)}%`;
}

function updateCharCount() {
  charCountInfo.textContent = `${sentenceOutput.value.length} karakter`;
}

function updateHoldToType(now) {
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

function drawHands(results, width, height, isMirrored) {
  if (!results || !results.landmarks) return;

  results.landmarks.forEach((rawLandmarks) => {
    const points = rawLandmarks.map((lm) => {
      const normX = isMirrored ? 1 - lm.x : lm.x;
      return {
        x: Math.min(width - 1, Math.max(0, Math.round(normX * width))),
        y: Math.min(height - 1, Math.max(0, Math.round(lm.y * height))),
      };
    });

    const activePoints = points.slice(1);
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
    FINGER_3BONE_CONNECTIONS.forEach(([start, end]) => {
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

      if (now - lastInferenceTime >= 80) {
        lastInferenceTime = now;
        predictClientSide(results.landmarks);
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

initModels();
