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

// DOM Elements
const webcamEl = document.getElementById("webcam");
const canvasEl = document.getElementById("overlayCanvas");
const ctx = canvasEl.getContext("2d");
const cropPreviewCanvas = document.getElementById("cropPreviewCanvas");
const cropPreviewCtx = cropPreviewCanvas.getContext("2d");

const camPlaceholder = document.getElementById("camPlaceholder");
const btnToggleCam = document.getElementById("btnToggleCam");
const camBtnText = document.getElementById("camBtnText");
const btnSnapshot = document.getElementById("btnSnapshot");
const selCropMode = document.getElementById("selCropMode");
const chkMirror = document.getElementById("chkMirror");
const chkShowROI = document.getElementById("chkShowROI");

// Status Indicators
const wsDot = document.getElementById("wsDot");
const wsStatus = document.getElementById("wsStatus");
const mpDot = document.getElementById("mpDot");
const mpStatus = document.getElementById("mpStatus");
const tfDevice = document.getElementById("tfDevice");
const telemetryStateBadge = document.getElementById("telemetryStateBadge");

// HUD & Live Prediction
const hudFps = document.getElementById("hudFps");
const hudHands = document.getElementById("hudHands");
const hudInferMs = document.getElementById("hudInferMs");
const livePredChar = document.getElementById("livePredChar");
const livePredConf = document.getElementById("livePredConf");
const holdProgressBar = document.getElementById("holdProgressBar");
const holdStatusText = document.getElementById("holdStatusText");

// Sentence Builder
const sentenceOutput = document.getElementById("sentenceOutput");
const tokenHistoryContainer = document.getElementById("tokenHistoryContainer");
const charCountBadge = document.getElementById("charCountBadge");
const btnAddCurrent = document.getElementById("btnAddCurrent");
const btnAddSpace = document.getElementById("btnAddSpace");
const btnBackspace = document.getElementById("btnBackspace");
const btnSpeak = document.getElementById("btnSpeak");
const btnClearText = document.getElementById("btnClearText");

// Telemetry Cards
const metricConf = document.getElementById("metricConf");
const metricMeanConf = document.getElementById("metricMeanConf");
const metricBackendMs = document.getElementById("metricBackendMs");
const metricRttMs = document.getElementById("metricRttMs");
const metricHandScore = document.getElementById("metricHandScore");
const metricHandedness = document.getElementById("metricHandedness");
const metricEntropy = document.getElementById("metricEntropy");
const top5Bars = document.getElementById("top5Bars");
const totalFramesPredicted = document.getElementById("totalFramesPredicted");

// Controls
const rngConfThresh = document.getElementById("rngConfThresh");
const lblConfThresh = document.getElementById("lblConfThresh");
const rngHoldMs = document.getElementById("rngHoldMs");
const lblHoldMs = document.getElementById("lblHoldMs");
const rngCropPad = document.getElementById("rngCropPad");
const lblCropPad = document.getElementById("lblCropPad");
const rngInferInterval = document.getElementById("rngInferInterval");
const lblInferInterval = document.getElementById("lblInferInterval");

// Logs & Evaluation
const selGroundTruth = document.getElementById("selGroundTruth");
const sessionAccuracy = document.getElementById("sessionAccuracy");
const logTableBody = document.getElementById("logTableBody");
const btnExportCSV = document.getElementById("btnExportCSV");
const btnExportJSON = document.getElementById("btnExportJSON");
const btnClearLogs = document.getElementById("btnClearLogs");

// Offscreen canvases:
// 1. captureCanvas menyimpan frame kamera mentah (tanpa garis overlay)
const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

// 2. resnetInputCanvas berukuran tepat 224x224 untuk dikirim ke backend ResNet50
const resnetInputCanvas = document.createElement("canvas");
resnetInputCanvas.width = 224;
resnetInputCanvas.height = 224;
const resnetInputCtx = resnetInputCanvas.getContext("2d");

// Application State
let handLandmarker = null;
let ws = null;
let isCamRunning = false;
let animFrameId = null;

// Tracking & Inference State
let fpsCounter = 0;
let currentFps = 0;
let lastFpsUpdate = performance.now();

let lastInferenceSentAt = 0;
let awaitingWsResponse = false;
let currentHandCount = 0;
let currentHandScore = 0;
let currentHandedness = "-";

// Temporal Smoothing (Exponential Moving Average) untuk menstabilkan prediksi
let smoothedProbs = {};
const EMA_ALPHA = 0.55; // Bobot frame terbaru (55% baru, 45% histori)

let latestPrediction = null;
let latestConfidence = 0;
let latestEntropy = 0;

// Hold-to-Type State
let candidateLabel = null;
let candidateHoldStart = 0;
let cooldownUntil = 0;
const tokenHistory = [];

// Research Logs
const researchLogs = [];
let sumConfidence = 0;
let targetMatchCount = 0;
let targetTotalCount = 0;

// 1. Fetch Backend Model Info
async function loadBackendInfo() {
  try {
    const res = await fetch("/api/info");
    const info = await res.json();
    tfDevice.textContent = `${info.devices.join(", ")} (TF ${info.tensorflow_version})`;

    info.classes.forEach((cls) => {
      const opt = document.createElement("option");
      opt.value = cls;
      opt.textContent = `Kelas: ${cls}`;
      selGroundTruth.appendChild(opt);
    });
  } catch (err) {
    tfDevice.textContent = "Error memuat info";
  }
}

// 2. Initialize MediaPipe HandLandmarker
async function initMediaPipe() {
  try {
    mpStatus.textContent = "Memuat WASM...";
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
    mpDot.className = "w-2 h-2 rounded-full bg-emerald-400";
    mpStatus.textContent = "Siap (GPU 2 Tangan)";
  } catch (err) {
    console.error("MediaPipe Init Error:", err);
    mpDot.className = "w-2 h-2 rounded-full bg-rose-500";
    mpStatus.textContent = "Gagal memuat";
  }
}

// 3. Initialize WebSocket for ResNet50 Inference
function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${window.location.host}/ws/predict`);

  ws.onopen = () => {
    wsDot.className = "w-2 h-2 rounded-full bg-emerald-400";
    wsStatus.textContent = "Terhubung";
  };

  ws.onclose = () => {
    wsDot.className = "w-2 h-2 rounded-full bg-rose-500";
    wsStatus.textContent = "Terputus (Reconnect...)";
    awaitingWsResponse = false;
    setTimeout(initWebSocket, 2000);
  };

  ws.onmessage = (event) => {
    awaitingWsResponse = false;
    const data = JSON.parse(event.data);
    if (data.error) {
      console.warn("Backend inference error:", data.error);
      return;
    }
    handlePredictionResult(data);
  };
}

// 4. Handle ResNet50 Prediction + Temporal EMA Smoothing
function handlePredictionResult(data) {
  if (currentHandCount === 0) return;

  const rttMs = Math.round(performance.now() - (data.client_ts || performance.now()));
  const backendMs = data.metrics?.total_backend_ms ?? 0;

  // Terapkan Exponential Moving Average pada distribusi probabilitas 26 kelas
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
    renderTop5(sorted.slice(0, 5));
  } else {
    latestPrediction = data.prediction;
    latestConfidence = data.confidence;
    renderTop5(data.top5);
  }

  latestEntropy = data.entropy;

  // Update HUD & Cards
  livePredChar.textContent = latestPrediction;
  livePredConf.textContent = `${(latestConfidence * 100).toFixed(1)}%`;
  hudInferMs.textContent = `${backendMs.toFixed(0)} ms`;

  metricConf.textContent = `${(latestConfidence * 100).toFixed(2)}%`;
  metricBackendMs.textContent = `${backendMs.toFixed(1)} ms`;
  metricRttMs.textContent = `RTT: ${rttMs} ms`;
  metricEntropy.textContent = latestEntropy.toFixed(3);

  recordResearchSample(data, rttMs);
}

function renderTop5(top5) {
  if (!top5 || top5.length === 0) return;
  top5Bars.innerHTML = top5
    .map((item, idx) => {
      const pct = (item.confidence * 100).toFixed(1);
      const barColor =
        idx === 0
          ? "bg-emerald-500"
          : idx === 1
          ? "bg-cyan-500/80"
          : "bg-slate-600";
      return `
        <div class="flex items-center gap-2">
          <span class="w-8 font-bold ${idx === 0 ? "text-emerald-400" : "text-slate-300"}">${item.label}</span>
          <div class="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
            <div class="h-full ${barColor}" style="width: ${pct}%"></div>
          </div>
          <span class="w-14 text-right text-slate-300">${pct}%</span>
        </div>
      `;
    })
    .join("");
}

function recordResearchSample(data, rttMs) {
  // Hanya catat ke dalam log jika akurasi (confidence) di atas 95% (> 0.95)
  if (latestConfidence <= 0.95) {
    return;
  }

  const groundTruth = selGroundTruth.value;
  let isMatch = null;
  if (groundTruth) {
    targetTotalCount++;
    isMatch = latestPrediction === groundTruth;
    if (isMatch) targetMatchCount++;
    const acc = ((targetMatchCount / targetTotalCount) * 100).toFixed(1);
    sessionAccuracy.textContent = `${acc}% (${targetMatchCount}/${targetTotalCount})`;
  }

  sumConfidence += latestConfidence;
  const sampleIndex = researchLogs.length + 1;
  const meanConf = ((sumConfidence / sampleIndex) * 100).toFixed(1);
  metricMeanConf.textContent = `Avg: ${meanConf}%`;
  totalFramesPredicted.textContent = `${sampleIndex} sampel`;

  const nowStr = new Date().toLocaleTimeString("id-ID", { hour12: false });
  const entry = {
    id: sampleIndex,
    timestamp: new Date().toISOString(),
    time_display: nowStr,
    crop_mode: selCropMode.value,
    prediction: latestPrediction,
    raw_prediction: data.prediction,
    confidence: Number(latestConfidence.toFixed(4)),
    entropy: data.entropy,
    hand_count: currentHandCount,
    hand_score: Number(currentHandScore.toFixed(4)),
    handedness: currentHandedness,
    backend_ms: data.metrics.total_backend_ms,
    inference_ms: data.metrics.inference_ms,
    rtt_ms: rttMs,
    ground_truth: groundTruth || "-",
    correct: isMatch === null ? "-" : isMatch ? "BENAR" : "SALAH",
  };
  researchLogs.push(entry);

  if (researchLogs.length === 1) {
    logTableBody.innerHTML = "";
  }
  const tr = document.createElement("tr");
  tr.className = "hover:bg-slate-900/60";
  const statusBadge =
    isMatch === null
      ? `<span class="text-slate-500">${selCropMode.value === "hand_square" ? "1:1 CROP" : "RAW"}</span>`
      : isMatch
      ? `<span class="text-emerald-400 font-semibold">BENAR</span>`
      : `<span class="text-rose-400 font-semibold">SALAH (${groundTruth})</span>`;

  tr.innerHTML = `
    <td class="py-1.5 px-2.5 text-slate-500">${entry.id}</td>
    <td class="py-1.5 px-2.5">${entry.time_display}</td>
    <td class="py-1.5 px-2.5 font-bold text-emerald-400">${entry.prediction}</td>
    <td class="py-1.5 px-2.5">${(entry.confidence * 100).toFixed(1)}%</td>
    <td class="py-1.5 px-2.5 text-cyan-400">${entry.hand_count} (${(entry.hand_score * 100).toFixed(0)}%)</td>
    <td class="py-1.5 px-2.5 text-indigo-300">${entry.backend_ms}</td>
    <td class="py-1.5 px-2.5">${statusBadge}</td>
  `;
  logTableBody.prepend(tr);
  while (logTableBody.children.length > 60) {
    logTableBody.removeChild(logTableBody.lastChild);
  }
}

// 5. Hold-to-Type Logic
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
    holdStatusText.textContent =
      currentHandCount === 0 ? "Menunggu..." : "Conf Rendah";
    return;
  }

  if (candidateLabel !== latestPrediction) {
    candidateLabel = latestPrediction;
    candidateHoldStart = now;
    holdProgressBar.style.width = "5%";
    holdStatusText.textContent = `Tahan '${candidateLabel}'`;
    return;
  }

  const elapsed = now - candidateHoldStart;
  const progressPct = Math.min(100, (elapsed / requiredHoldMs) * 100);
  holdProgressBar.style.width = `${progressPct.toFixed(0)}%`;
  holdStatusText.textContent = `${Math.round(progressPct)}% (${candidateLabel})`;

  if (elapsed >= requiredHoldMs) {
    appendTokenToSentence(candidateLabel);
    cooldownUntil = now + 750;
    candidateLabel = null;
    candidateHoldStart = 0;
  }
}

function appendTokenToSentence(token) {
  if (!token) return;
  sentenceOutput.value += token;
  tokenHistory.push({
    token,
    time: new Date().toLocaleTimeString("id-ID", { hour12: false }),
    conf: (latestConfidence * 100).toFixed(0),
  });
  updateSentenceStats();
}

function updateSentenceStats() {
  const text = sentenceOutput.value;
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  charCountBadge.textContent = `${text.length} karakter | ${words} kata`;

  if (tokenHistory.length === 0) {
    tokenHistoryContainer.innerHTML = `<span class="text-slate-500 italic">Belum ada token huruf yang diketik...</span>`;
    return;
  }

  tokenHistoryContainer.innerHTML = tokenHistory
    .slice(-12)
    .map(
      (item) => `
      <span class="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-emerald-300 mono text-[11px] shrink-0">
        <b>${item.token === " " ? "␣" : item.token}</b> <span class="text-slate-500">${item.conf}%</span>
      </span>
    `
    )
    .join("");
}

// 6. Compute Combined Square Bounding Box (1:1 Aspect Ratio) for ResNet50
function computeResNetCropBox(rawLandmarksList, width, height, padRatio) {
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

  // Mode "hand_square": Gabungkan semua titik dari 1 atau 2 tangan yang terdeteksi
  let minX = width;
  let maxX = 0;
  let minY = height;
  let maxY = 0;

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

  const boxW = Math.max(40, maxX - minX);
  const boxH = Math.max(40, maxY - minY);
  const maxSpan = Math.max(boxW, boxH);
  // Tambahkan margin (default 35%) agar proporsi tangan terhadap latar belakang sama dengan foto dataset Data/
  let side = Math.round(maxSpan * (1 + padRatio * 2));
  side = Math.min(Math.min(width, height), Math.max(160, side));

  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;

  let x0 = Math.round(cx - side / 2);
  let y0 = Math.round(cy - side / 2);

  // Jaga agar kotak crop 1:1 tidak keluar dari batas frame kamera
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x0 + side > width) x0 = width - side;
  if (y0 + side > height) y0 = height - side;

  return { x: x0, y: y0, w: side, h: side };
}

// 7. Draw Hand Landmarks, Per-Hand Box & Cyan ResNet Crop Box
function drawTrackingOverlay(results, cropBox, width, height, isMirrored) {
  if (chkShowROI.checked) {
    const yOffset = Math.floor(height * 0.2);
    ctx.save();
    ctx.strokeStyle = "rgba(245, 158, 11, 0.55)";
    ctx.setLineDash([6, 6]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, yOffset);
    ctx.lineTo(width, yOffset);
    ctx.stroke();
    ctx.restore();
  }

  // Gambar kotak Area Crop ResNet50 (Cyan putus-putus) jika tangan terdeteksi
  if (cropBox && results?.landmarks?.length > 0) {
    ctx.save();
    const drawX = isMirrored ? width - (cropBox.x + cropBox.w) : cropBox.x;
    ctx.strokeStyle = "rgba(34, 211, 238, 0.85)";
    ctx.setLineDash([8, 5]);
    ctx.lineWidth = 2;
    ctx.strokeRect(drawX, cropBox.y, cropBox.w, cropBox.h);
    ctx.fillStyle = "rgba(34, 211, 238, 0.9)";
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillText(
      `RESNET INPUT CROP (${cropBox.w}x${cropBox.h} -> 224x224)`,
      drawX + 6,
      Math.min(height - 8, cropBox.y + cropBox.h - 6)
    );
    ctx.restore();
  }

  if (!results || !results.landmarks || results.landmarks.length === 0) {
    return;
  }

  const is3BonesOnly = selCropMode.value === "finger_3bones";
  const connections = is3BonesOnly ? FINGER_3BONE_CONNECTIONS : HAND_CONNECTIONS;

  results.landmarks.forEach((rawLandmarks, handIdx) => {
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
    const padding = 18;
    const xMin = Math.max(0, Math.min(...xs) - padding);
    const xMax = Math.min(width - 1, Math.max(...xs) + padding);
    const yMin = Math.max(0, Math.min(...ys) - padding);
    const yMax = Math.min(height - 1, Math.max(...ys) + padding);

    // Kotak Hijau Bounding Box Tangan
    ctx.strokeStyle = "#00ff00";
    ctx.lineWidth = 2;
    ctx.strokeRect(xMin, yMin, xMax - xMin, yMax - yMin);

    // Garis Koneksi Landmark Kuning
    ctx.strokeStyle = "#ffff00";
    ctx.lineWidth = 2.5;
    connections.forEach(([start, end]) => {
      ctx.beginPath();
      ctx.moveTo(points[start].x, points[start].y);
      ctx.lineTo(points[end].x, points[end].y);
      ctx.stroke();
    });

    // Titik Landmark Merah
    ctx.fillStyle = "#ff2222";
    activePoints.forEach((pt) => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
      ctx.fill();
    });

    const handInfo = results.handednesses?.[handIdx]?.[0];
    const handLabel = handInfo
      ? `HAND TRACKING (${handInfo.categoryName} ${(handInfo.score * 100).toFixed(0)}%)`
      : "HAND TRACKING";

    ctx.fillStyle = "#00ff00";
    ctx.font = "bold 12px 'JetBrains Mono', monospace";
    ctx.fillText(handLabel, xMin, Math.max(20, yMin - 8));
  });
}

// 8. Main Real-time Loop
function renderLoop() {
  if (!isCamRunning) return;

  const now = performance.now();
  fpsCounter++;
  if (now - lastFpsUpdate >= 500) {
    currentFps = Math.round((fpsCounter * 1000) / (now - lastFpsUpdate));
    hudFps.textContent = currentFps;
    fpsCounter = 0;
    lastFpsUpdate = now;
  }

  if (webcamEl.readyState >= 2) {
    const width = webcamEl.videoWidth || 640;
    const height = webcamEl.videoHeight || 480;

    if (canvasEl.width !== width || canvasEl.height !== height) {
      canvasEl.width = width;
      canvasEl.height = height;
      captureCanvas.width = width;
      captureCanvas.height = height;
    }

    // 1. Simpan frame kamera asli tanpa mirror di captureCanvas
    captureCtx.drawImage(webcamEl, 0, 0, width, height);

    // 2. Gambar frame ke layar pengguna (dengan mirror jika dicentang)
    const isMirrored = chkMirror.checked;
    ctx.save();
    if (isMirrored) {
      ctx.translate(width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(webcamEl, 0, 0, width, height);
    ctx.restore();

    // 3. Deteksi tangan dengan MediaPipe pada captureCanvas
    let results = null;
    if (handLandmarker) {
      results = handLandmarker.detectForVideo(captureCanvas, now);
    }

    currentHandCount = results?.landmarks?.length || 0;
    hudHands.textContent = currentHandCount;

    const padRatio = Number(rngCropPad.value) / 100;
    const cropBox = computeResNetCropBox(results?.landmarks, width, height, padRatio);

    if (currentHandCount > 0) {
      const firstHandInfo = results.handednesses?.[0]?.[0];
      currentHandScore = firstHandInfo?.score || 0;
      currentHandedness = results.handednesses
        .map((h) => h[0]?.categoryName)
        .filter(Boolean)
        .join(", ");

      metricHandScore.textContent = `${(currentHandScore * 100).toFixed(1)}%`;
      metricHandedness.textContent = `Tangan: ${currentHandedness}`;
      telemetryStateBadge.textContent = "TRACKING AKTIF";
      telemetryStateBadge.className =
        "text-[11px] mono px-2.5 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/30";

      // Gambar overlay landmark & kotak crop
      drawTrackingOverlay(results, cropBox, width, height, isMirrored);

      // Siapkan gambar 224x224 dari area cropBox (dari captureCanvas yang bersih tanpa garis)
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

      // Tampilkan preview 224x224 di pojok kanan bawah video agar peneliti melihat persis apa yang masuk ke ResNet50
      cropPreviewCtx.drawImage(resnetInputCanvas, 0, 0, 112, 112);

      // Kirim ke backend via WebSocket (Landmark 3 Tulang Jari atau Gambar 224x224 ResNet50)
      const inferInterval = Number(rngInferInterval.value);
      if (
        ws &&
        ws.readyState === WebSocket.OPEN &&
        !awaitingWsResponse &&
        now - lastInferenceSentAt >= inferInterval
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
          const jpegB64 = resnetInputCanvas.toDataURL("image/jpeg", 0.88);
          ws.send(
            JSON.stringify({
              model_type: "resnet50",
              image: jpegB64,
              client_ts: now,
            })
          );
        }
      }
    } else {
      // Reset smoothing saat tangan keluar dari kamera
      smoothedProbs = {};
      latestPrediction = null;
      latestConfidence = 0;
      livePredChar.textContent = "-";
      livePredConf.textContent = "Posisikan tangan";
      metricHandScore.textContent = "0.00%";
      metricHandedness.textContent = "Tangan: -";
      telemetryStateBadge.textContent = "MENUNGGU TANGAN";
      telemetryStateBadge.className =
        "text-[11px] mono px-2.5 py-0.5 rounded-full bg-amber-500/15 text-amber-300 border border-amber-500/30";
    }

    updateHoldToType(now);
  }

  animFrameId = requestAnimationFrame(renderLoop);
}

// 9. Camera Start / Stop
async function toggleCamera() {
  if (isCamRunning) {
    isCamRunning = false;
    if (animFrameId) cancelAnimationFrame(animFrameId);
    const stream = webcamEl.srcObject;
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
    }
    webcamEl.srcObject = null;
    camPlaceholder.classList.remove("hidden");
    camBtnText.textContent = "Mulai Kamera";
    btnToggleCam.className =
      "px-3.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs transition flex items-center gap-2 shadow";
    telemetryStateBadge.textContent = "IDLE";
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
    camBtnText.textContent = "Hentikan Kamera";
    btnToggleCam.className =
      "px-3.5 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-medium text-xs transition flex items-center gap-2 shadow";
    renderLoop();
  } catch (err) {
    alert("Tidak dapat mengakses kamera: " + err.message);
  }
}

// 10. Event Listeners
btnToggleCam.addEventListener("click", toggleCamera);

btnSnapshot.addEventListener("click", () => {
  if (!isCamRunning) return;
  const link = document.createElement("a");
  link.download = `bisindo_capture_${Date.now()}.jpg`;
  link.href = canvasEl.toDataURL("image/jpeg", 0.95);
  link.click();
});

selCropMode.addEventListener("change", () => {
  smoothedProbs = {};
});

btnAddCurrent.addEventListener("click", () => {
  if (latestPrediction) appendTokenToSentence(latestPrediction);
});

btnAddSpace.addEventListener("click", () => {
  sentenceOutput.value += " ";
  updateSentenceStats();
});

btnBackspace.addEventListener("click", () => {
  sentenceOutput.value = sentenceOutput.value.slice(0, -1);
  tokenHistory.pop();
  updateSentenceStats();
});

btnClearText.addEventListener("click", () => {
  sentenceOutput.value = "";
  tokenHistory.length = 0;
  updateSentenceStats();
});

btnSpeak.addEventListener("click", () => {
  const txt = sentenceOutput.value.trim();
  if (!txt || !("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(txt);
  utter.lang = "id-ID";
  window.speechSynthesis.speak(utter);
});

sentenceOutput.addEventListener("input", updateSentenceStats);

rngConfThresh.addEventListener("input", () => {
  lblConfThresh.textContent = `${rngConfThresh.value}%`;
});
rngHoldMs.addEventListener("input", () => {
  lblHoldMs.textContent = `${rngHoldMs.value} ms`;
});
rngCropPad.addEventListener("input", () => {
  lblCropPad.textContent = `${rngCropPad.value}%`;
});
rngInferInterval.addEventListener("input", () => {
  lblInferInterval.textContent = `${rngInferInterval.value} ms`;
});

selGroundTruth.addEventListener("change", () => {
  targetMatchCount = 0;
  targetTotalCount = 0;
  sessionAccuracy.textContent = selGroundTruth.value ? "0.0% (0/0)" : "-";
});

btnExportCSV.addEventListener("click", () => {
  if (researchLogs.length === 0) {
    alert("Belum ada data log penelitian untuk diekspor.");
    return;
  }
  const headers = [
    "id",
    "timestamp",
    "crop_mode",
    "prediction",
    "raw_prediction",
    "confidence",
    "entropy",
    "hand_count",
    "hand_score",
    "handedness",
    "backend_ms",
    "inference_ms",
    "rtt_ms",
    "ground_truth",
    "correct",
  ];
  const rows = researchLogs.map((r) =>
    headers.map((h) => JSON.stringify(r[h] ?? "")).join(",")
  );
  const csvContent = [headers.join(","), ...rows].join("\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `log_penelitian_bisindo_${Date.now()}.csv`;
  link.click();
});

btnExportJSON.addEventListener("click", () => {
  if (researchLogs.length === 0) {
    alert("Belum ada data log penelitian untuk diekspor.");
    return;
  }
  const payload = {
    exported_at: new Date().toISOString(),
    built_sentence: sentenceOutput.value,
    total_samples: researchLogs.length,
    mean_confidence: sumConfidence / researchLogs.length,
    logs: researchLogs,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `log_penelitian_bisindo_${Date.now()}.json`;
  link.click();
});

btnClearLogs.addEventListener("click", () => {
  researchLogs.length = 0;
  sumConfidence = 0;
  targetMatchCount = 0;
  targetTotalCount = 0;
  metricMeanConf.textContent = "Avg: 0.0%";
  totalFramesPredicted.textContent = "0 sampel";
  sessionAccuracy.textContent = "-";
  logTableBody.innerHTML = `<tr><td colspan="7" class="py-6 text-center text-slate-500 font-sans">Belum ada data log inferensi.</td></tr>`;
});

// Boot
loadBackendInfo();
initMediaPipe();
initWebSocket();
