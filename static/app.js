const SAMPLE_RATE = 16000;
const WINDOW_SECONDS = 3;

const els = {
  systemStatus: document.getElementById("systemStatus"),
  tabUpload: document.getElementById("tabUpload"),
  tabLive: document.getElementById("tabLive"),
  panelUpload: document.getElementById("panelUpload"),
  panelLive: document.getElementById("panelLive"),
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("fileInput"),
  fileMeta: document.getElementById("fileMeta"),
  fileName: document.getElementById("fileName"),
  clearFile: document.getElementById("clearFile"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  startMic: document.getElementById("startMic"),
  stopMic: document.getElementById("stopMic"),
  micOrb: document.getElementById("micOrb"),
  micHint: document.getElementById("micHint"),
  levelFill: document.getElementById("levelFill"),
  verdict: document.getElementById("verdict"),
  verdictLabel: document.getElementById("verdictLabel"),
  verdictConf: document.getElementById("verdictConf"),
  resultStamp: document.getElementById("resultStamp"),
  historyList: document.getElementById("historyList"),
};

let selectedFile = null;
let live = {
  active: false,
  stream: null,
  audioCtx: null,
  processor: null,
  source: null,
  buffer: [],
  anim: null,
};

function setMode(mode) {
  const upload = mode === "upload";
  els.tabUpload.classList.toggle("is-active", upload);
  els.tabLive.classList.toggle("is-active", !upload);
  els.tabUpload.setAttribute("aria-selected", String(upload));
  els.tabLive.setAttribute("aria-selected", String(!upload));
  els.panelUpload.hidden = !upload;
  els.panelLive.hidden = upload;
  if (upload) stopMicrophone();
}

els.tabUpload.addEventListener("click", () => setMode("upload"));
els.tabLive.addEventListener("click", () => setMode("live"));

function setFile(file) {
  selectedFile = file || null;
  if (!selectedFile) {
    els.fileMeta.hidden = true;
    els.analyzeBtn.disabled = true;
    els.fileInput.value = "";
    return;
  }
  els.fileMeta.hidden = false;
  els.fileName.textContent = selectedFile.name;
  els.analyzeBtn.disabled = false;
}

els.dropzone.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", () => {
  const file = els.fileInput.files?.[0];
  if (file) setFile(file);
});

["dragenter", "dragover"].forEach((evt) => {
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    els.dropzone.classList.add("is-drag");
  });
});
["dragleave", "drop"].forEach((evt) => {
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    els.dropzone.classList.remove("is-drag");
  });
});
els.dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files?.[0];
  if (file) setFile(file);
});
els.clearFile.addEventListener("click", () => setFile(null));

els.analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  els.analyzeBtn.disabled = true;
  els.analyzeBtn.textContent = "Analyzing…";
  try {
    const form = new FormData();
    form.append("file", selectedFile, selectedFile.name);
    const res = await fetch("/api/classify", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Classification failed");
    renderResult(data, selectedFile.name);
  } catch (err) {
    renderError(err.message || String(err));
  } finally {
    els.analyzeBtn.disabled = !selectedFile;
    els.analyzeBtn.textContent = "Analyze recording";
  }
});

function renderError(message) {
  els.verdict.dataset.state = "uncertain";
  els.verdictLabel.textContent = "Error";
  els.verdictConf.textContent = message;
  els.resultStamp.textContent = "Failed";
}

function renderResult(data, sourceLabel) {
  const state = data.uncertain ? "uncertain" : data.label;
  const resultClass = data.label; // healthy | unhealthy | noise
  const title = data.uncertain
    ? `Uncertain · ${data.label.charAt(0).toUpperCase() + data.label.slice(1)}`
    : data.label.charAt(0).toUpperCase() + data.label.slice(1);

  const confPct = Math.round((data.confidence || 0) * 100);

  els.verdict.dataset.state = state;
  els.verdictLabel.textContent = title;
  els.verdictConf.textContent = `Confidence ${confPct}% · ${sourceLabel}`;
  els.resultStamp.textContent = new Date().toLocaleTimeString();

  // Show only the final result bar (Healthy OR Unhealthy OR Noise)
  document.querySelectorAll(".bar-row").forEach((row) => {
    const cls = row.dataset.class;
    const isWinner = cls === resultClass;
    row.hidden = !isWinner;
    if (isWinner) {
      row.querySelector(".bar-fill").style.width = `${confPct}%`;
      row.querySelector("strong").textContent = `${confPct}%`;
    } else {
      row.querySelector(".bar-fill").style.width = "0%";
      row.querySelector("strong").textContent = "0%";
    }
  });

  const li = document.createElement("li");
  li.innerHTML = `<span class="h-label">${title}</span><span class="h-time">${confPct}% · ${new Date().toLocaleTimeString()}</span>`;
  els.historyList.prepend(li);
  while (els.historyList.children.length > 8) {
    els.historyList.lastElementChild.remove();
  }
}

function floatTo16BitPCM(float32Array) {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < float32Array.length; i++) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return view;
}

function encodeWav(samples, sampleRate) {
  const pcm = floatTo16BitPCM(samples);
  const buffer = new ArrayBuffer(44 + pcm.byteLength);
  const view = new DataView(buffer);
  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + pcm.byteLength, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, pcm.byteLength, true);
  new Uint8Array(buffer, 44).set(new Uint8Array(pcm.buffer));
  return new Blob([buffer], { type: "audio/wav" });
}

function downsampleBuffer(buffer, fromRate, toRate) {
  if (fromRate === toRate) return buffer;
  const ratio = fromRate / toRate;
  const newLen = Math.floor(buffer.length / ratio);
  const result = new Float32Array(newLen);
  for (let i = 0; i < newLen; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const a = buffer[idx] || 0;
    const b = buffer[Math.min(idx + 1, buffer.length - 1)] || 0;
    result[i] = a + (b - a) * frac;
  }
  return result;
}

let liveBusy = false;
let liveBirdStreak = 0;
let livePendingBird = null;
const LIVE_CONFIRM_WINDOWS = 3; // need 3 bird windows in a row before showing health

async function classifyLiveChunk(samples) {
  if (liveBusy) return;
  liveBusy = true;
  try {
    const wav = encodeWav(samples, SAMPLE_RATE);
    const form = new FormData();
    form.append("file", wav, "chunk.wav");
    const res = await fetch("/api/classify-chunk", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Live classification failed");

    const poultry = Boolean(
      data.windows?.[0]?.poultry_detected || data.flock_detected || data.poultry_windows > 0
    );
    const isBirdLabel = data.label === "healthy" || data.label === "unhealthy";

    if (poultry && isBirdLabel) {
      liveBirdStreak += 1;
      livePendingBird = data;
      if (liveBirdStreak < LIVE_CONFIRM_WINDOWS) {
        // Hold as background until confirmed — stops brief talk from flipping to Healthy
        const holding = {
          ...data,
          label: "noise",
          uncertain: true,
          confidence: Math.max(data.distribution?.noise || 0.55, 0.55),
          flock_detected: false,
          poultry_windows: 0,
          _holding: true,
        };
        if (holding.windows?.[0]) {
          holding.windows = [
            { ...holding.windows[0], poultry_detected: false, label: "noise" },
          ];
        }
        renderResult(holding, "microphone (confirming…)");
        els.micHint.textContent = `Possible poultry — confirming ${liveBirdStreak}/${LIVE_CONFIRM_WINDOWS}`;
        return;
      }
      renderResult(livePendingBird, "microphone");
      els.micHint.textContent = "Listening — poultry voice confirmed";
    } else {
      liveBirdStreak = 0;
      livePendingBird = null;
      renderResult(data, "microphone");
      els.micHint.textContent = "Listening — background/talk ignored";
    }
  } finally {
    liveBusy = false;
  }
}

async function startMicrophone() {
  if (live.active) return;
  try {
    // Keep poultry chirps intact — browser noise suppression can erase them
    live.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: true,
        channelCount: 1,
      },
    });
  } catch (err) {
    renderError("Microphone permission denied or unavailable.");
    return;
  }

  live.audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
  if (live.audioCtx.state === "suspended") {
    await live.audioCtx.resume();
  }
  live.source = live.audioCtx.createMediaStreamSource(live.stream);
  live.processor = live.audioCtx.createScriptProcessor(4096, 1, 1);
  live.silent = live.audioCtx.createGain();
  live.silent.gain.value = 0;
  live.buffer = [];
  live.active = true;
  liveBusy = false;
  liveBirdStreak = 0;
  livePendingBird = null;

  const need = SAMPLE_RATE * WINDOW_SECONDS;

  live.processor.onaudioprocess = (event) => {
    if (!live.active) return;
    const input = event.inputBuffer.getChannelData(0);
    const down = downsampleBuffer(input, live.audioCtx.sampleRate, SAMPLE_RATE);
    for (let i = 0; i < down.length; i++) live.buffer.push(down[i]);

    let sum = 0;
    for (let i = 0; i < input.length; i++) sum += Math.abs(input[i]);
    const level = Math.min(100, (sum / input.length) * 450);
    els.levelFill.style.width = `${level}%`;

    if (!liveBusy && live.buffer.length >= need) {
      const chunk = Float32Array.from(live.buffer.slice(0, need));
      // keep a small overlap so we don't drop chirps between windows
      live.buffer = live.buffer.slice(Math.floor(need * 0.5));
      classifyLiveChunk(chunk).catch((err) => renderError(err.message || String(err)));
    }
  };

  live.source.connect(live.processor);
  // Mute monitor path — prevents speaker feedback into the mic
  live.processor.connect(live.silent);
  live.silent.connect(live.audioCtx.destination);

  els.micOrb.dataset.state = "listening";
  els.micHint.textContent = "Listening — classifying every ~3 seconds";
  els.startMic.disabled = true;
  els.stopMic.disabled = false;
}

function stopMicrophone() {
  live.active = false;
  liveBusy = false;
  liveBirdStreak = 0;
  livePendingBird = null;
  if (live.processor) {
    live.processor.disconnect();
    live.processor.onaudioprocess = null;
  }
  if (live.silent) live.silent.disconnect();
  if (live.source) live.source.disconnect();
  if (live.audioCtx) live.audioCtx.close().catch(() => {});
  if (live.stream) live.stream.getTracks().forEach((t) => t.stop());
  live.processor = null;
  live.silent = null;
  live.source = null;
  live.audioCtx = null;
  live.stream = null;
  live.buffer = [];
  els.levelFill.style.width = "0%";
  els.micOrb.dataset.state = "idle";
  els.micHint.textContent = "Microphone idle";
  els.startMic.disabled = false;
  els.stopMic.disabled = true;
}

els.startMic.addEventListener("click", startMicrophone);
els.stopMic.addEventListener("click", stopMicrophone);
window.addEventListener("beforeunload", stopMicrophone);

async function boot() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.model_loaded) {
      els.systemStatus.textContent = "Model ready";
      els.systemStatus.classList.add("is-ready");
    } else {
      els.systemStatus.textContent = "Model missing";
    }
  } catch {
    els.systemStatus.textContent = "Server offline";
  }
}

boot();
