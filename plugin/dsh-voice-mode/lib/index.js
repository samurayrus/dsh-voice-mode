// src/index.ts
import z from "@deepseek-ai/schemastery";
import { dirname as dirname2, join as join5 } from "node:path";
import { homedir as homedir2 } from "node:os";
import { rm } from "node:fs/promises";

// src/asr-host.ts
import { statSync } from "node:fs";
import { join as join2 } from "node:path";
import { fileURLToPath } from "node:url";
import { Worker } from "node:worker_threads";

// src/sense-worker.ts
import { parentPort, workerData } from "node:worker_threads";
function createSenseWorkerClient(worker) {
  let counter = 0;
  const pending = /* @__PURE__ */ new Map();
  let dead = false;
  const deathFns = /* @__PURE__ */ new Set();
  const die = () => {
    if (dead) return;
    dead = true;
    for (const fn of deathFns) {
      try {
        fn();
      } catch {
      }
    }
  };
  worker.on?.("message", (msg) => {
    const p = pending.get(msg?.id);
    if (!p) return;
    pending.delete(msg.id);
    if (!msg.ok) {
      p.resolve(null);
      return;
    }
    p.resolve(p.op === "create" ? true : msg.text ?? "");
  });
  worker.on?.("error", (e) => {
    die();
    const err = new Error("sense worker error: " + String(e?.message ?? e));
    for (const [, p] of pending) p.reject(err);
    pending.clear();
  });
  worker.on?.("exit", () => {
    die();
    const err = new Error("sense worker exited");
    for (const [, p] of pending) p.reject(err);
    pending.clear();
  });
  const request = (op, samples) => {
    if (dead) return Promise.reject(new Error("sense worker dead"));
    const id = counter++;
    return new Promise((resolve, reject) => {
      pending.set(id, { op, resolve, reject });
      const msg = { id, op };
      if (samples) msg.samples = samples;
      try {
        worker.postMessage(msg);
      } catch (e) {
        pending.delete(id);
        reject(e instanceof Error ? e : new Error(String(e)));
      }
    });
  };
  return {
    request,
    onDeath(fn) {
      deathFns.add(fn);
    },
    terminate: async () => {
      dead = true;
      const err = new Error("sense worker terminated");
      for (const [, p] of pending) p.reject(err);
      pending.clear();
      try {
        await worker.terminate?.();
      } catch {
      }
    }
  };
}
function startSenseWorker(data) {
  const port = parentPort;
  if (!port) return;
  let recognizer = null;
  let sherpa = null;
  port.on("message", async (msg) => {
    try {
      if (msg.op === "create" || msg.op === "decode") {
        if (!sherpa) {
          sherpa = await import(data.sherpaModule);
        }
        if (!recognizer) {
          recognizer = sherpa.createOfflineRecognizer({
            featConfig: { sampleRate: 16e3, featureDim: 80 },
            modelConfig: {
              senseVoice: {
                model: data.modelDir + "/model.int8.onnx",
                language: data.language,
                useInverseTextNormalization: data.useITN
              },
              tokens: data.modelDir + "/tokens.txt",
              provider: "cpu",
              debug: 0
            }
          });
        }
        if (msg.op === "decode" && msg.samples) {
          const stream = recognizer.createStream();
          try {
            stream.acceptWaveform(16e3, msg.samples);
            recognizer.decode(stream);
            const text = recognizer.getResult(stream).text.trim();
            port.postMessage({ id: msg.id, ok: true, text });
          } finally {
            try {
              stream.free();
            } catch {
            }
          }
          return;
        }
        port.postMessage({ id: msg.id, ok: true, text: "" });
        return;
      }
      port.postMessage({ id: msg.id, ok: false, error: "unknown op: " + msg.op });
    } catch (e) {
      port.postMessage({ id: msg.id, ok: false, error: String(e) });
    }
  });
}
if (parentPort) {
  startSenseWorker(workerData);
}

// src/asr-host.ts
import sherpa_onnx from "sherpa-onnx";

// src/models.ts
import { createHash } from "node:crypto";
import { createWriteStream } from "node:fs";
import { mkdir, readdir, readFile, rename, stat, unlink, writeFile } from "node:fs/promises";
import { join } from "node:path";
var HOST_PRIMARY = "https://huggingface.co";
var HOST_FALLBACK = "https://hf-mirror.com";
var ALLOWED_MODEL_HOSTNAMES = ["huggingface.co", "hf.co", "hf-mirror.com"];
function validateModelHost(raw, allowCustomHost) {
  if (!raw) return null;
  let u;
  try {
    u = new URL(raw);
  } catch {
    return null;
  }
  if (u.protocol !== "https:") return null;
  const hostname = u.hostname.toLowerCase();
  if (!ALLOWED_MODEL_HOSTNAMES.includes(hostname) && !allowCustomHost) {
    return null;
  }
  return `${u.protocol}//${u.hostname}${u.port ? `:${u.port}` : ""}`;
}
function redirectHostAllowed(finalUrl, allowCustomHost) {
  try {
    const u = new URL(finalUrl);
    if (u.protocol !== "https:") return false;
    const hostname = u.hostname.toLowerCase();
    if (allowCustomHost) return true;
    return hostname === "huggingface.co" || hostname.endsWith(".huggingface.co") || hostname === "hf.co" || hostname.endsWith(".hf.co") || hostname === "hf-mirror.com" || hostname.endsWith(".hf-mirror.com");
  } catch {
    return false;
  }
}
var verifiedCache = /* @__PURE__ */ new Map();
var cacheKey = (st) => `${st.size}:${st.ino}:${st.ctimeMs}`;
async function sha256OfFile(path) {
  const hash = createHash("sha256");
  const { createReadStream } = await import("node:fs");
  await new Promise((resolve, reject) => {
    const stream = createReadStream(path);
    stream.on("data", (c) => hash.update(c));
    stream.on("error", reject);
    stream.on("end", () => resolve());
  });
  return hash.digest("hex");
}
var MODEL_FETCH_HEADERS_TIMEOUT_MS = 6e4;
var MODEL_BODY_IDLE_TIMEOUT_MS = 12e4;
function makeIdleWatch(ac, ms) {
  let t;
  const stop = () => clearTimeout(t);
  const kick = () => {
    stop();
    t = setTimeout(() => ac.abort(new Error("model download idle timeout")), ms);
  };
  return { kick, stop };
}
async function fetchWithHeadersTimeout(url, init, ac = new AbortController()) {
  const timer = setTimeout(() => ac.abort(new Error("model fetch headers timeout")), MODEL_FETCH_HEADERS_TIMEOUT_MS);
  try {
    return await fetch(url, { ...init, signal: ac.signal });
  } finally {
    clearTimeout(timer);
  }
}
async function ensureModelFile(opts) {
  const { repoDir, spec, primaryHost, broadcast } = opts;
  const localPath = join(repoDir, spec.file);
  const partPath = `${localPath}.part`;
  const existing = await stat(localPath).catch(() => null);
  if (existing?.isFile()) {
    if (verifiedCache.get(localPath) === `${cacheKey(existing)}:${spec.sha256}`) return true;
    const ok = await sha256OfFile(localPath).catch(() => "") === spec.sha256;
    if (ok) {
      verifiedCache.set(localPath, `${cacheKey(existing)}:${spec.sha256}`);
      return true;
    }
    await unlink(localPath).catch(() => void 0);
  }
  await mkdir(join(repoDir, spec.file.includes("/") ? spec.file.slice(0, spec.file.lastIndexOf("/")) : ""), {
    recursive: true
  }).catch(() => void 0);
  const hosts = [...new Set([primaryHost, HOST_PRIMARY, HOST_FALLBACK].filter(Boolean))];
  let lastError = "no upstream reachable";
  for (const host of hosts) {
    if (opts.signal?.aborted) break;
    try {
      const done = await downloadVerified({ ...opts, host, partPath, localPath });
      if (done) return true;
    } catch (e) {
      lastError = String(e);
    }
  }
  if (opts.signal?.aborted) return false;
  if (lastError) console.warn(`[dsh-voice-mode] model download failed: ${spec.file}: ${lastError}`);
  broadcast("asr-error", { file: spec.file, reason: "checksum_or_download_failed" });
  return false;
}
async function downloadVerified(opts) {
  const { repo, spec, host, allowCustomHost, partPath, localPath, broadcast } = opts;
  const ac = new AbortController();
  const onAbort = () => ac.abort(opts.signal?.reason);
  if (opts.signal?.aborted) onAbort();
  else opts.signal?.addEventListener("abort", onAbort, { once: true });
  const idle = makeIdleWatch(ac, opts.idleTimeoutMs ?? MODEL_BODY_IDLE_TIMEOUT_MS);
  try {
    return await downloadBody();
  } finally {
    idle.stop();
    opts.signal?.removeEventListener("abort", onAbort);
  }
  async function downloadBody() {
    const url = `${host}/${repo}/resolve/${spec.revision ?? "main"}/${spec.file}`;
    const partSt = await stat(partPath).catch(() => null);
    const resumeFrom = partSt?.isFile() ? partSt.size : 0;
    const headers = { "user-agent": "dsh-voice-mode" };
    if (resumeFrom > 0) headers.range = `bytes=${resumeFrom}-`;
    const res = await fetchWithHeadersTimeout(url, { headers, redirect: "follow" }, ac);
    idle.kick();
    if (!redirectHostAllowed(res.url, allowCustomHost)) return false;
    if (res.status === 416) {
      if (await sha256OfFile(partPath).catch(() => "") === spec.sha256) {
        await rename(partPath, localPath);
        return true;
      }
      await unlink(partPath).catch(() => void 0);
      return false;
    }
    if (res.status !== 200 && res.status !== 206) return false;
    const resume = res.status === 206 ? resumeFrom : 0;
    const total = Number(res.headers.get("content-length") ?? 0) + resume;
    const src = res.body;
    if (!src) return false;
    const sink = createWriteStream(partPath, resume > 0 ? { flags: "a" } : {});
    const reader = src.getReader();
    let received = resume;
    await new Promise((resolve, reject) => {
      sink.on("error", (e) => reject(e));
      sink.on("finish", () => resolve());
      void (async () => {
        try {
          for (; ; ) {
            const { done, value } = await reader.read();
            if (done) break;
            idle.kick();
            received += value.byteLength;
            if (!sink.write(value)) {
              await new Promise((r) => sink.once("drain", r));
              idle.kick();
            }
            if (total > 0) {
              broadcast("asr-progress", {
                file: spec.file,
                percent: Math.min(100, Math.round(received / total * 100))
              });
            }
          }
          sink.end();
        } catch (e) {
          sink.destroy(e);
          reject(e);
        }
      })();
    });
    idle.stop();
    const actual = await sha256OfFile(partPath).catch(() => "");
    if (actual !== spec.sha256) {
      await unlink(partPath).catch(() => void 0);
      return false;
    }
    await rename(partPath, localPath);
    const st = await stat(localPath).catch(() => null);
    if (st) verifiedCache.set(localPath, `${cacheKey(st)}:${spec.sha256}`);
    return true;
  }
}
function isSafeModelRel(rel, subdir) {
  if (!rel.startsWith(subdir + "/")) return false;
  if (rel.includes("\\") || rel.includes("\0")) return false;
  return !rel.split("/").some((seg) => seg === "" || seg === "." || seg === "..");
}
var TREE_MANIFEST = ".model-tree.json";
async function countTreeFiles(dir) {
  const ents = await readdir(dir, { recursive: true, withFileTypes: true }).catch(() => []);
  return ents.filter((e) => e.isFile() && !e.name.endsWith(".part") && e.name !== TREE_MANIFEST).length;
}
async function treeCompleteLocally(repoDir, subdir, sentinel, minFiles) {
  const subRoot = join(repoDir, subdir);
  const raw = await readFile(join(subRoot, TREE_MANIFEST), "utf8").catch(() => null);
  if (raw !== null) {
    try {
      const m = JSON.parse(raw);
      const files = m.files ?? [];
      if (files.length > 0) {
        for (const f of files) {
          const st = await stat(join(repoDir, f.rel)).catch(() => null);
          if (!st?.isFile() || st.size !== f.size) return false;
        }
        return true;
      }
    } catch {
    }
  }
  if (!sentinel || !minFiles) return false;
  if (!(await stat(join(repoDir, sentinel)).catch(() => null))?.isFile()) return false;
  return await countTreeFiles(subRoot) >= minFiles;
}
async function ensureModelTree(opts) {
  const { repo, repoDir, subdir, primaryHost, allowCustomHost, broadcast } = opts;
  const hosts = [...new Set([primaryHost, HOST_PRIMARY, HOST_FALLBACK].filter(Boolean))];
  const subRoot = join(repoDir, subdir);
  if (await treeCompleteLocally(repoDir, subdir, opts.sentinel, opts.minFiles)) return true;
  await mkdir(subRoot, { recursive: true }).catch(() => void 0);
  let tree = [];
  for (const host of hosts) {
    try {
      const res = await fetchWithHeadersTimeout(host + "/api/models/" + repo + "?blobs=true", {
        headers: { "user-agent": "dsh-voice-mode" }
      });
      if (!redirectHostAllowed(res.url, allowCustomHost)) continue;
      if (res.ok) {
        const j = await res.json();
        tree = (j.siblings ?? []).map((s) => s.rfilename ?? "").filter((f) => isSafeModelRel(f, subdir));
        if (tree.length > 0) break;
      }
    } catch {
    }
  }
  if (tree.length === 0) return false;
  let allOk = true;
  let done = 0;
  const queue = [...tree];
  const worker = async () => {
    for (; ; ) {
      const rel = queue.shift();
      if (rel === void 0) return;
      const localPath = join(repoDir, rel);
      const partPath = localPath + ".part";
      if ((await stat(localPath).catch(() => null))?.isFile() && (await stat(localPath)).size > 0) {
        done++;
        continue;
      }
      await mkdir(join(repoDir, rel.slice(0, rel.lastIndexOf("/"))), { recursive: true }).catch(() => void 0);
      let ok = false;
      for (const host of hosts) {
        try {
          const url = host + "/" + repo + "/resolve/main/" + encodeURIComponent(rel);
          const ac = new AbortController();
          const onAbort = () => ac.abort(opts.signal?.reason);
          if (opts.signal?.aborted) onAbort();
          else opts.signal?.addEventListener("abort", onAbort, { once: true });
          const idle = makeIdleWatch(ac, opts.idleTimeoutMs ?? MODEL_BODY_IDLE_TIMEOUT_MS);
          try {
            const res = await fetchWithHeadersTimeout(url, { headers: { "user-agent": "dsh-voice-mode" }, redirect: "follow" }, ac);
            if (!redirectHostAllowed(res.url, allowCustomHost)) continue;
            if (res.status !== 200) continue;
            idle.kick();
            const reader = res.body?.getReader();
            if (!reader) continue;
            const sink = createWriteStream(partPath);
            let size = 0;
            await new Promise((resolve, reject) => {
              sink.on("error", reject);
              sink.on("finish", resolve);
              void (async () => {
                try {
                  for (; ; ) {
                    const r = await reader.read();
                    if (r.done) break;
                    idle.kick();
                    size += r.value.byteLength;
                    if (!sink.write(r.value)) {
                      await new Promise((r2) => sink.once("drain", r2));
                      idle.kick();
                    }
                  }
                  sink.end();
                } catch (e) {
                  sink.destroy(e);
                  reject(e);
                }
              })();
            });
            if (size > 0) {
              await rename(partPath, localPath);
              ok = true;
              break;
            }
          } finally {
            idle.stop();
            opts.signal?.removeEventListener("abort", onAbort);
          }
        } catch {
          if (opts.signal?.aborted) break;
        }
      }
      if (ok) {
        done++;
        broadcast("asr-progress", { file: rel, percent: Math.round(done / tree.length * 100) });
      } else {
        allOk = false;
      }
    }
  };
  await Promise.all(Array.from({ length: 8 }, () => worker()));
  if (allOk) {
    const files = [];
    for (const rel of tree) {
      const st = await stat(join(repoDir, rel)).catch(() => null);
      if (st?.isFile()) files.push({ rel, size: st.size });
    }
    if (files.length === tree.length) {
      await writeFile(join(subRoot, TREE_MANIFEST), JSON.stringify({ files })).catch(() => void 0);
    }
  }
  return allOk;
}

// src/asr-host.ts
var { createOnlineRecognizer, createVad } = sherpa_onnx;
var MODEL_REPO = "csukuangfj/sherpa-onnx-streaming-zipformer-small-ru-vosk-int8-2025-08-16";
var MODEL_FILES = [
  { file: "encoder.int8.onnx", sha256: "e0db705e94ec35d803b1df4f40cda23d064e1142977c80ab288430b109777a9d" },
  { file: "decoder.onnx", sha256: "89b3088a9e20e1ef7f2e85ce1a3478afe6a9c4ac57369cabcc4beb8e95328ea0" },
  { file: "joiner.int8.onnx", sha256: "b55784b071ab7512eab4c7c44e4f5478284ef33c83562cc6a249b972515a31e5" },
  { file: "tokens.txt", sha256: "93bbbc0bae6b78c0bbb743d4aa9fded3bb5ff3aac5f0200e3a769a5a05e0fdf6" }
];
var VAD_REPO = "csukuangfj/vad";
var VAD_FILES = [
  { file: "silero_vad.onnx", sha256: "a35ebf52fd3ce5f1469b2a36158dba761bc47b973ea3382b3186ca15b1f5af28" }
];
var SENSE_REPO = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17";
var SENSE_FILES = [
  { file: "model.int8.onnx", sha256: "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51" },
  { file: "tokens.txt", sha256: "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc" }
];
function pcmToSamples(buf) {
  if (buf.length % 4 !== 0) return null;
  return new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
}
var MAX_ASR_BYTES = 4 * 1024 * 1024;
var SEGMENT_IDLE_MS = 9e4;
var VAD_CONTINUE_RMS = 0.02;
var CONFIRM_CONJUNCTION_MS = 800;
var CONFIRM_LONG_SENTENCE_MS = 350;
var CONFIRM_LONG_SENTENCE_S = 8;
var CONFIRM_MIN_MS = 200;
var CONJUNCTION_TAIL = /(然后|还有|以及|并且|而且|此外|再说|接着|然后呢|比方说|比如说|比如|例如|等等|或者|或是|还有呢)$/;
function endpointConfirmMs(text, spokenMs) {
  const tail = text.trimEnd();
  if (CONJUNCTION_TAIL.test(tail)) return CONFIRM_CONJUNCTION_MS;
  if (spokenMs > CONFIRM_LONG_SENTENCE_S * 1e3) return CONFIRM_LONG_SENTENCE_MS;
  return CONFIRM_MIN_MS;
}
function rmsOf(samples) {
  if (samples.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}
function createAsrRuntime(options) {
  const { cacheDir, modelHost, broadcast, senseVoice, silenceMs, senseITN, allowCustomHost, bargeInMode } = options;
  let lastProgress = null;
  const localBroadcast = (event, payload) => {
    if (event === "asr-progress") lastProgress = payload;
    broadcast(event, payload);
  };
  const repoDir = join2(cacheDir, MODEL_REPO);
  const vadDir = join2(cacheDir, VAD_REPO);
  const senseDir = join2(cacheDir, SENSE_REPO);
  const normalizedModelHost = () => validateModelHost(modelHost(), allowCustomHost) ?? HOST_PRIMARY;
  const segments = /* @__PURE__ */ new Map();
  const finalized = /* @__PURE__ */ new Map();
  const finalizing = /* @__PURE__ */ new Map();
  const resetGen = /* @__PURE__ */ new Map();
  let recognizer = null;
  let recognizerDisposed = false;
  let modelsReady = false;
  let modelsLoading = null;
  let asrFailAt = 0;
  const ensureModels = async () => {
    if (modelsReady) return true;
    if (Date.now() < asrFailAt) return false;
    if (!modelsLoading) {
      modelsLoading = (async () => {
        for (const f of MODEL_FILES) {
          if (!await ensureModelFile({ repo: MODEL_REPO, repoDir, spec: f, primaryHost: normalizedModelHost(), allowCustomHost, broadcast: localBroadcast })) {
            asrFailAt = Date.now() + 6e4;
            broadcast("asr-error", { file: f.file });
            return false;
          }
        }
        modelsReady = true;
        broadcast("asr-ready", {});
        return true;
      })().finally(() => {
        modelsLoading = null;
        lastProgress = null;
      });
    }
    return modelsLoading;
  };
  const getRecognizer = async () => {
    if (recognizerDisposed) return null;
    if (!await ensureModels()) return null;
    if (recognizerDisposed) return null;
    if (recognizer) return recognizer;
    const t = (f) => join2(repoDir, f);
    recognizer = createOnlineRecognizer({
      modelConfig: {
        transducer: {
          encoder: t("encoder.int8.onnx"),
          decoder: t("decoder.onnx"),
          joiner: t("joiner.int8.onnx")
        },
        tokens: t("tokens.txt"),
        numThreads: 4,
        provider: "cpu",
        debug: 0
      }
    });
    return recognizer;
  };
  let vadModelReady = false;
  let vadLoading = null;
  let vadFailAt = 0;
  const ensureVadModel = async () => {
    if (vadModelReady) return join2(vadDir, VAD_FILES[0].file);
    if (Date.now() < vadFailAt) return null;
    if (!vadLoading) {
      vadLoading = (async () => {
        for (const f of VAD_FILES) {
          if (!await ensureModelFile({ repo: VAD_REPO, repoDir: vadDir, spec: f, primaryHost: normalizedModelHost(), allowCustomHost, broadcast: localBroadcast })) {
            vadFailAt = Date.now() + 6e4;
            return null;
          }
        }
        vadModelReady = true;
        return join2(vadDir, VAD_FILES[0].file);
      })().finally(() => {
        vadLoading = null;
        lastProgress = null;
      });
    }
    return vadLoading;
  };
  const newVad = (vadPath, threshold = 0.5, minSilenceDuration = 0.5) => createVad({
    sileroVad: {
      model: vadPath,
      threshold,
      minSilenceDuration,
      minSpeechDuration: 0.25,
      maxSpeechDuration: 20,
      windowSize: 512
    },
    sampleRate: 16e3,
    numThreads: 1,
    provider: "cpu",
    debug: 0,
    bufferSizeInSeconds: 30
  });
  const ensureSessionVad = async (seg) => {
    if (seg.vad) return seg.vad;
    const vadPath = await ensureVadModel();
    if (!vadPath) return null;
    seg.vad = newVad(vadPath, 0.5, silenceMs() / 1e3);
    return seg.vad;
  };
  const detectVads = /* @__PURE__ */ new Map();
  const detectVadLastUse = /* @__PURE__ */ new Map();
  const ensureDetectVad = async (sessionId) => {
    const existing = detectVads.get(sessionId);
    if (existing) return existing;
    const vadPath = await ensureVadModel();
    if (!vadPath) return null;
    const vad = newVad(vadPath, 0.35);
    detectVads.set(sessionId, vad);
    return vad;
  };
  let senseModelReady = false;
  let senseLoading = null;
  let senseFailAt = 0;
  const ensureSenseModel = async () => {
    if (senseModelReady) return join2(senseDir, SENSE_FILES[0].file);
    if (Date.now() < senseFailAt) return null;
    if (!senseLoading) {
      senseLoading = (async () => {
        for (const f of SENSE_FILES) {
          if (!await ensureModelFile({ repo: SENSE_REPO, repoDir: senseDir, spec: f, primaryHost: normalizedModelHost(), allowCustomHost, broadcast: localBroadcast })) {
            senseFailAt = Date.now() + 6e4;
            return null;
          }
        }
        senseModelReady = true;
        return join2(senseDir, SENSE_FILES[0].file);
      })().finally(() => {
        senseLoading = null;
        lastProgress = null;
      });
    }
    return senseLoading;
  };
  let senseWorker = null;
  let senseWorkerSyncing = null;
  let senseWorkerLangKey = "";
  let senseDisposed = false;
  const getSenseWorker = async () => {
    if (!senseVoice()) return null;
    const langKey = `auto\0${senseITN() ? "1" : "0"}`;
    if (senseWorker && langKey !== senseWorkerLangKey) {
      void senseWorker.terminate();
      senseWorker = null;
      senseWorkerSyncing = null;
    }
    senseWorkerLangKey = langKey;
    if (senseWorker) return senseWorker;
    if (senseWorkerSyncing) return senseWorkerSyncing;
    senseWorkerSyncing = (async () => {
      if (senseDisposed) return null;
      const sensePath = await ensureSenseModel();
      if (!sensePath) return null;
      try {
        const workerPath = fileURLToPath(new URL("./sense-worker.mjs", import.meta.url));
        const w = new Worker(workerPath, {
          workerData: {
            sherpaModule: "sherpa-onnx",
            modelDir: senseDir,
            language: "auto",
            useITN: senseITN() ? 1 : 0
          }
        });
        const client = createSenseWorkerClient(w);
        client.onDeath(() => {
          senseWorker = null;
          senseWorkerSyncing = null;
          senseWorkerLangKey = "";
        });
        if (!await client.request("create")) {
          await client.terminate();
          return null;
        }
        if (senseDisposed) {
          await client.terminate();
          return null;
        }
        senseWorker = client;
        return client;
      } catch (e) {
        console.warn("[dsh-voice-mode] SenseVoice worker init failed: " + String(e));
        return null;
      }
    })().finally(() => {
      if (!senseWorker || senseDisposed) senseWorkerSyncing = null;
    });
    return senseWorkerSyncing;
  };
  const senseTranscribe = async (allSamples) => {
    try {
      const worker = await getSenseWorker();
      if (!worker) return null;
      const total = allSamples.reduce((acc, c) => acc + c.length, 0);
      if (total === 0) return null;
      const buf = new Float32Array(total);
      let off = 0;
      for (const c of allSamples) {
        buf.set(c, off);
        off += c.length;
      }
      return await worker.request("decode", buf);
    } catch (e) {
      console.warn("[dsh-voice-mode] SenseVoice re-transcribe failed: " + String(e));
      return null;
    }
  };
  const feed = async (sessionId, samples, final, offset = 0, epoch = 0, manualPressed = false) => {
    if (bargeInMode() === "manual" && !manualPressed) {
      return { text: "" };
    }
    const rec = await getRecognizer();
    if (!rec) return { text: "", loading: true };
    if (!final && senseVoice()) {
      void getSenseWorker().catch(() => {
      });
    }
    let finMap = finalized.get(sessionId);
    const myGen = resetGen.get(sessionId) ?? 0;
    const cached2 = finMap?.get(epoch);
    if (cached2 !== void 0) return { text: cached2 };
    let sessSegs = segments.get(sessionId);
    if (!sessSegs) {
      sessSegs = /* @__PURE__ */ new Map();
      segments.set(sessionId, sessSegs);
    }
    let seg = sessSegs.get(epoch);
    if (!seg) {
      if (samples.length === 0 && final) return { text: "" };
      seg = { stream: rec.createStream(), fed: 0, vad: null, pendingEndpoint: null, lastText: "", allSamples: [], lastActivity: Date.now() };
      sessSegs.set(epoch, seg);
    }
    seg.lastActivity = Date.now();
    let endpoint = false;
    let text = "";
    let isSpeech;
    if (offset + samples.length > seg.fed) {
      const skip = Math.max(seg.fed - offset, 0);
      const inc = samples.subarray(skip);
      seg.stream.acceptWaveform(rec.config.featConfig.sampleRate, inc);
      seg.fed = offset + samples.length;
      if (seg.fed <= rec.config.featConfig.sampleRate * 60) seg.allSamples.push(inc);
      while (rec.isReady(seg.stream)) rec.decode(seg.stream);
      text = rec.getResult(seg.stream).text;
      seg.lastText = text;
      if (!final) {
        const vad = await ensureSessionVad(seg);
        if (vad) {
          if (seg.pendingEndpoint) {
            const now = Date.now();
            const rms = rmsOf(inc);
            if (rms > VAD_CONTINUE_RMS) {
              seg.pendingEndpoint = null;
            } else if (now - seg.pendingEndpoint.at >= CONFIRM_MIN_MS && text === seg.pendingEndpoint.textAtPending) {
              seg.pendingEndpoint = null;
              endpoint = true;
            } else if (now - seg.pendingEndpoint.at >= seg.pendingEndpoint.confirmMs) {
              seg.pendingEndpoint = null;
              endpoint = true;
            }
          }
          vad.acceptWaveform(inc);
          isSpeech = vad.isDetected();
          if (!vad.isEmpty()) {
            let spokenMs = 0;
            while (!vad.isEmpty()) {
              const sp = vad.front();
              spokenMs = sp.samples.length / 16e3 * 1e3;
              vad.pop();
            }
            const confirmMs = endpointConfirmMs(seg.lastText, spokenMs);
            if (confirmMs <= 0) {
              endpoint = true;
            } else {
              seg.pendingEndpoint = { at: Date.now(), confirmMs, textAtPending: seg.lastText };
            }
          }
        }
      }
    }
    if (!final) return { text, endpoint, isSpeech };
    const inflightMap = finalizing.get(sessionId);
    const inflightP = inflightMap?.get(epoch);
    if (inflightP) return { text: await inflightP };
    sessSegs.delete(epoch);
    if (sessSegs.size === 0) segments.delete(sessionId);
    const finalizeP = (async () => {
      const all = seg.allSamples;
      let senseTimer = null;
      const senseP = all.length > 0 ? Promise.race([
        senseTranscribe(all),
        new Promise((resolve) => {
          senseTimer = setTimeout(() => resolve(null), 2e4);
        })
      ]).finally(() => {
        if (senseTimer) clearTimeout(senseTimer);
      }) : Promise.resolve(null);
      const pad = new Float32Array(rec.config.featConfig.sampleRate / 2);
      seg.stream.acceptWaveform(rec.config.featConfig.sampleRate, pad);
      while (rec.isReady(seg.stream)) rec.decode(seg.stream);
      const settled = rec.getResult(seg.stream).text;
      try {
        seg.vad?.free?.();
      } catch {
      }
      seg.stream.free();
      const sense = await senseP;
      return (sense && sense.trim() ? sense : settled) || "";
    })().then((finalText) => {
      if ((resetGen.get(sessionId) ?? 0) !== myGen) return finalText;
      let fm = finalized.get(sessionId);
      if (!fm) {
        fm = /* @__PURE__ */ new Map();
        finalized.set(sessionId, fm);
      }
      fm.set(epoch, finalText);
      if (fm.size > 32) {
        const first = fm.keys().next().value;
        if (first !== void 0) fm.delete(first);
      }
      const ff = finalizing.get(sessionId);
      ff?.delete(epoch);
      if (ff && ff.size === 0) finalizing.delete(sessionId);
      return finalText;
    }).catch((e) => {
      const ff = finalizing.get(sessionId);
      ff?.delete(epoch);
      if (ff && ff.size === 0) finalizing.delete(sessionId);
      console.warn("[dsh-voice-mode] finalize failed: " + String(e));
      return "";
    });
    if (!inflightMap) {
      finalizing.set(sessionId, /* @__PURE__ */ new Map());
    }
    finalizing.get(sessionId).set(epoch, finalizeP);
    return { text: await finalizeP };
  };
  const sweep = () => {
    const now = Date.now();
    for (const [sid, sessSegs] of segments) {
      for (const [epoch, s] of sessSegs) {
        if (now - s.lastActivity > SEGMENT_IDLE_MS) {
          try {
            s.vad?.free?.();
          } catch {
          }
          try {
            s.stream.free();
          } catch {
          }
          sessSegs.delete(epoch);
        }
      }
      if (sessSegs.size === 0) {
        segments.delete(sid);
        finalized.delete(sid);
      }
    }
    for (const sid of finalized.keys()) {
      if (segments.has(sid) || finalizing.has(sid)) continue;
      finalized.delete(sid);
      resetGen.delete(sid);
    }
    for (const sid of resetGen.keys()) {
      if (segments.has(sid) || finalizing.has(sid)) continue;
      resetGen.delete(sid);
    }
    for (const [sid, at] of detectVadLastUse) {
      if (now - at > SEGMENT_IDLE_MS) {
        try {
          detectVads.get(sid)?.free?.();
        } catch {
        }
        detectVads.delete(sid);
        detectVadLastUse.delete(sid);
      }
    }
  };
  const sweepTimer = setInterval(sweep, 3e4);
  return {
    feed,
    detect: async (sessionId, samples) => {
      const vad = await ensureDetectVad(sessionId);
      if (!vad) return { isSpeech: false };
      detectVadLastUse.set(sessionId, Date.now());
      if (samples.length > 0) vad.acceptWaveform(samples);
      const speech = vad.isDetected();
      while (!vad.isEmpty()) vad.pop();
      return { isSpeech: speech };
    },
    reset: (sessionId) => {
      const sessSegs = segments.get(sessionId);
      if (sessSegs) {
        for (const [, s] of sessSegs) {
          try {
            s.vad?.free?.();
          } catch {
          }
          try {
            s.stream.free();
          } catch {
          }
        }
        segments.delete(sessionId);
      }
      finalized.delete(sessionId);
      resetGen.set(sessionId, (resetGen.get(sessionId) ?? 0) + 1);
      finalizing.delete(sessionId);
      const dv = detectVads.get(sessionId);
      if (dv) {
        try {
          dv.free?.();
        } catch {
        }
        detectVads.delete(sessionId);
      }
      detectVadLastUse.delete(sessionId);
    },
    dispose: () => {
      clearInterval(sweepTimer);
      senseDisposed = true;
      recognizerDisposed = true;
      let w = senseWorker;
      senseWorker = null;
      senseWorkerSyncing = null;
      if (w) void w.terminate();
      for (const [, sessSegs] of segments) {
        for (const [, s] of sessSegs) {
          try {
            s.vad?.free?.();
          } catch {
          }
          try {
            s.stream.free();
          } catch {
          }
        }
      }
      segments.clear();
      finalized.clear();
      finalizing.clear();
      resetGen.clear();
      try {
        recognizer?.free?.();
      } catch {
      }
      recognizer = null;
      for (const [, dv] of detectVads) {
        try {
          dv.free?.();
        } catch {
        }
      }
      detectVads.clear();
      detectVadLastUse.clear();
    },
    warmup: () => {
      void getRecognizer().catch(() => void 0);
      void ensureVadModel().catch(() => void 0);
      if (senseVoice()) void getSenseWorker().catch(() => void 0);
    },
    // 批 E：SenseVoice 预热前置 enterMode——返回 Promise 让 /toggle on=true await，
    // 内部 5s 上限防止慢模型下载 hang 住 enterMode（失败/超时静默降级走 finalize 时 race）。
    warmupSense: async () => {
      if (!senseVoice()) return;
      let warmupTimer = null;
      await Promise.race([
        getSenseWorker().then(() => void 0).catch(() => void 0),
        new Promise((resolve) => {
          warmupTimer = setTimeout(resolve, 5e3);
        })
      ]).finally(() => {
        if (warmupTimer) clearTimeout(warmupTimer);
      });
    },
    // 批 A：仅清缓存键——不 dispose recognizer/senseWorker，避免破坏 I1（in-flight
    // finalize 拿到的旧 recognizer 引用被 free 会丢句）。让现有 fingerprint-gated 路径
    // （getSenseWorker:396-401）下次自然触发重建。
    markStale: () => {
      senseWorkerLangKey = "";
    },
    modelStatus: () => {
      const asrFiles = MODEL_FILES.map((n) => ({
        name: n.file,
        exists: (() => {
          try {
            return statSync(join2(repoDir, n.file)).isFile();
          } catch {
            return false;
          }
        })(),
        size: (() => {
          try {
            return statSync(join2(repoDir, n.file)).size;
          } catch {
            return 0;
          }
        })()
      }));
      const vadSize = (() => {
        try {
          return statSync(join2(vadDir, VAD_FILES[0].file)).size;
        } catch {
          return 0;
        }
      })();
      const senseSize = (() => {
        try {
          return statSync(join2(senseDir, SENSE_FILES[0].file)).size;
        } catch {
          return 0;
        }
      })();
      return {
        // ready 语义 = 文件可用（exists），而非进程内是否已实例化——
        // 重启后文件齐全却显示「未下载」会误导用户（体验修复）。
        asr: {
          repo: MODEL_REPO,
          ready: asrFiles.every((f) => f.exists),
          files: asrFiles,
          failLatchMs: Math.max(0, asrFailAt - Date.now())
        },
        vad: {
          repo: VAD_REPO,
          ready: vadSize > 0,
          size: vadSize,
          failLatchMs: Math.max(0, vadFailAt - Date.now())
        },
        sense: {
          repo: SENSE_REPO,
          ready: senseSize > 0,
          size: senseSize,
          failLatchMs: Math.max(0, senseFailAt - Date.now()),
          enabled: senseVoice()
        },
        progress: lastProgress
      };
    },
    retryModel: async (kind) => {
      if (kind === "vad") {
        vadFailAt = 0;
        return !!await ensureVadModel();
      }
      if (kind === "sense") {
        if (!senseVoice()) return false;
        senseFailAt = 0;
        return !!await ensureSenseModel();
      }
      if (modelsReady) return true;
      asrFailAt = 0;
      return await ensureModels();
    }
  };
}
var respondJson = (res, status, payload) => {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(payload));
};
function handleAsrRequest(asr, activeSessionId, req, res) {
  const chunks = [];
  let received = 0;
  let tooLarge = false;
  req.on("data", (c) => {
    if (tooLarge) return;
    received += c.length;
    if (received > MAX_ASR_BYTES) {
      tooLarge = true;
      respondJson(res, 413, { code: "payload_too_large", error: "pcm payload too large" });
      return;
    }
    chunks.push(c);
  });
  req.on("end", () => {
    if (tooLarge) return;
    const url = new URL(req.url ?? "/", "http://localhost");
    const sessionId = url.searchParams.get("sessionId") ?? "";
    const final = url.searchParams.get("final") === "1";
    const reset = url.searchParams.get("reset") === "1";
    const epochParam = url.searchParams.get("epoch");
    const epochN = Number(epochParam);
    const epochOK = epochParam === null || Number.isFinite(epochN) && epochN >= 0 && Number.isInteger(epochN);
    const offsetParam = url.searchParams.get("offset");
    const offsetOK = offsetParam === null || Number.isFinite(Number(offsetParam)) && Number(offsetParam) >= 0 && Number(offsetParam) <= MAX_ASR_BYTES / 4;
    if (!offsetOK) {
      respondJson(res, 400, { code: "bad_request", error: "invalid offset" });
      return;
    }
    if (!epochOK) {
      respondJson(res, 400, { code: "bad_request", error: "invalid epoch" });
      return;
    }
    const epoch = epochParam === null ? 0 : Math.floor(epochN);
    const offset = offsetParam === null ? 0 : Math.floor(Number(offsetParam));
    if (!sessionId || sessionId !== activeSessionId) {
      respondJson(res, 403, { code: "unknown_session", error: "not the active voice session" });
      return;
    }
    if (reset) {
      asr.reset(sessionId);
      respondJson(res, 200, { ok: true });
      return;
    }
    const raw = Buffer.concat(chunks);
    const samples = raw.length === 0 ? final ? new Float32Array(0) : null : pcmToSamples(raw);
    if (!samples) {
      respondJson(res, 400, { code: "bad_request", error: "invalid pcm payload" });
      return;
    }
    if (url.searchParams.get("vadOnly") === "1") {
      void asr.detect(sessionId, samples).then((out) => {
        respondJson(res, 200, { isSpeech: out.isSpeech });
      }).catch((e) => {
        console.warn(`[dsh-voice-mode] asr detect failed: ${String(e)}`);
        respondJson(res, 500, { code: "internal", error: "internal error" });
      });
      return;
    }
    void asr.feed(sessionId, samples, final, offset, epoch, url.searchParams.get("manual") === "1").then((out) => {
      if (out.loading) {
        respondJson(res, 202, { loading: true });
        return;
      }
      const body = { text: out.text };
      if (out.endpoint) body.endpoint = true;
      if (out.isSpeech !== void 0) body.isSpeech = out.isSpeech;
      respondJson(res, 200, body);
    }).catch((e) => {
      console.warn(`[dsh-voice-mode] asr decode failed: ${String(e)}`);
      respondJson(res, 500, { code: "internal", error: "internal error" });
    });
  });
}

// src/segmenter.ts
var SKIP_PREFIX = /^[\s.,，、:：;；!?！？)\]）"'”’〉》】]+$/;
function plainText(text) {
  return String(text).replace(/```[\s\S]*?```/g, " ").replace(/`([^`]*)`/g, "$1").replace(/!\[[^\]]*\]\([^)]*\)/g, " ").replace(/\[([^\]]*)\]\([^)]*\)/g, "$1").replace(/^#{1,6}\s+/gm, "").replace(/\*\*([^*]+)\*\*/g, "$1").replace(/\*([^*]+)\*/g, "$1").replace(/^[-*+]\s+/gm, "").replace(/^\d+\.\s+/gm, "").replace(/<\/?(?:b|i|u|br|p|span|div|strong|em|s|sub|sup|h[1-6]|ul|ol|li|a|img|code|pre|blockquote|hr|table|tr|td|th)\b[^>]*>/gi, " ");
}
function sanitizeForTts(text) {
  return String(text).replace(/[*_#|^=+~`]/g, " ").replace(/\s{2,}/g, " ").replace(/([\u3400-\u9fff])\s+(?=[\u3400-\u9fff])/g, "$1").trim();
}
function splitSentences(chunk) {
  const sentences = [];
  let start = 0;
  const re = /[。！？!?；;…\n]+|\.(?=\s|$)/g;
  let m;
  let lastEnd = 0;
  while ((m = re.exec(chunk)) !== null) {
    const end = m.index + m[0].length;
    if (m[0] === "." && end === chunk.length && /\d/.test(chunk[m.index - 1] ?? "")) continue;
    sentences.push(chunk.slice(start, end));
    start = end;
    lastEnd = end;
  }
  return { sentences, tail: chunk.slice(lastEnd) };
}
function cutHead(text, max) {
  let idx = -1;
  for (let i = Math.min(max, text.length) - 1; i > 0; i--) {
    if (/[，,、\s]/.test(text[i])) {
      idx = i + 1;
      break;
    }
  }
  if (idx <= 0) idx = Math.max(1, Math.floor(max / 2));
  return [text.slice(0, idx), text.slice(idx)];
}
var SentenceSegmenter = class {
  buffer = "";
  maxChars;
  /** 是否处在 ``` 代码块内（跨 delta 有效：流式 delta 只有几个 token，代码块总是被切碎送达，
   *  plainText 的整块正则匹配不到，代码会被逐句念出——「总是朗读」覆盖打字会话后尤其明显）。 */
  inFence = false;
  /** delta 末尾悬挂的 1-2 个反引号（被切开的 ``` 前半；连续 3 个及以上是完整围栏，不悬挂）。 */
  ticks = "";
  /** 围栏内被抑制的内容暂存：流结束时若仍未闭合则回退输出（宁可念代码，也不吞掉整回合正文）。 */
  fenced = "";
  constructor(options = {}) {
    this.maxChars = options.maxSentenceChars ?? 200;
  }
  /** 去掉 ``` 代码块内容（状态跨 delta 保持）；进入代码块处补一个换行，让前面的文字先成句。 */
  dropFences(chunk) {
    let s = this.ticks + chunk;
    this.ticks = "";
    const trail = /`+$/.exec(s);
    const hang = trail ? trail[0].length : 0;
    if (hang > 0 && hang <= 2) {
      this.ticks = s.slice(-hang);
      s = s.slice(0, -hang);
    }
    let out = "";
    for (; ; ) {
      const m = /`{3,}/.exec(s);
      if (!m) {
        if (this.inFence) this.fenced += s;
        else out += s;
        return out;
      }
      if (this.inFence) this.fenced += s.slice(0, m.index);
      else out += s.slice(0, m.index) + "\n";
      this.inFence = !this.inFence;
      if (!this.inFence) this.fenced = "";
      s = s.slice(m.index + m[0].length);
    }
  }
  /**
   * 安全阀：把超过 maxChars 的文字墙切下一段（feed 与 flush 共用同一口径）。
   * 切点取 ≤ maxChars 内**最后一个**「，,、\s」——取第一个会把无标点的英文/代码逐词切成
   * 几十个小段；超长段必须 ≤ maxChars，否则阀形同虚设。没有可用分隔符时取一半。
   */
  cutWall() {
    const [head, rest] = cutHead(this.buffer, this.maxChars);
    this.buffer = rest;
    return sanitizeForTts(head).trim();
  }
  /** 成句本身超过 maxChars（长逗号句）同样过安全阀，逐段输出。 */
  capSentence(s, out) {
    let rest = s;
    while (rest.length > this.maxChars) {
      const [head, tail] = cutHead(rest, this.maxChars);
      rest = tail;
      const t2 = sanitizeForTts(head).trim();
      if (t2 && !SKIP_PREFIX.test(t2)) out.push(t2);
    }
    const t = sanitizeForTts(rest).trim();
    if (t && !SKIP_PREFIX.test(t)) out.push(t);
  }
  /** 喂入一段 raw delta，返回它补全的完整句子。 */
  feed(chunk) {
    const cleaned = plainText(this.dropFences(chunk));
    if (!cleaned) return [];
    this.buffer += cleaned;
    const { sentences, tail } = splitSentences(this.buffer);
    this.buffer = tail;
    const out = [];
    for (const s of sentences) this.capSentence(s, out);
    while (this.buffer.length > this.maxChars) {
      const head = this.cutWall();
      if (head && !SKIP_PREFIX.test(head)) out.push(head);
    }
    return out;
  }
  /**
   * 收尾：flush 剩余缓冲（流结束）。
   * 整回合未见闭合围栏 → 把围栏内被抑制的内容回退输出（v0.7.27 之前未配对围栏零损失，
   * 全靠这一条守住；否则一个孤立的 ``` 会静默吞掉本回合剩余正文）。
   *
   * 过安全阀后的多段仍按「代码内容」朗读：既有取舍是「宁念代码不吞正文」，切分只改朗读粒度
   * （每段一次合成请求）、不改取舍；段序 = 原文字序，段内容 = 原文字符的连续切片（切点沿用
   * feed() 同一口径）。不回退成「整体不念」——那等于回到吞正文的老问题。
   */
  flush() {
    if (this.inFence) this.buffer = plainText(this.buffer + this.fenced);
    const out = [];
    while (this.buffer.length > this.maxChars) {
      const head = this.cutWall();
      if (head && !SKIP_PREFIX.test(head)) out.push(head);
    }
    const t = sanitizeForTts(this.buffer).trim();
    this.buffer = "";
    this.inFence = false;
    this.ticks = "";
    this.fenced = "";
    if (t && !SKIP_PREFIX.test(t)) out.push(t);
    return out;
  }
};

// src/tts-queue.ts
import { MsEdgeTTS, OUTPUT_FORMAT } from "./msedge-tts.cjs";

// src/emotion.ts
var EMOTION_TAG_PATTERN = "<\\s*(break\\s+(?<ms>\\d+)\\s*ms|whisper|\\/whisper|laugh|sigh|emphasis)\\s*>";
var TAG_RE = new RegExp(EMOTION_TAG_PATTERN, "gi");
var STRIP_RE = new RegExp(EMOTION_TAG_PATTERN, "gi");
function stripEmotionTags(raw) {
  return raw.replace(STRIP_RE, " ");
}
function parseEmotionTags(raw) {
  const out = [];
  let buf = "";
  let whisper = false;
  let pendingBreakMs = 0;
  const flush = () => {
    const t = buf.trim();
    buf = "";
    if (t.length === 0) return;
    const seg = { text: t, whisper };
    out.push(seg);
  };
  let lastEnd = 0;
  for (const m of raw.matchAll(TAG_RE)) {
    const idx = m.index ?? 0;
    buf += raw.slice(lastEnd, idx);
    lastEnd = idx + m[0].length;
    const tag = (m[1] ?? "").toLowerCase().trim();
    if (tag.startsWith("break")) {
      const ms = Number(m.groups?.ms ?? 0);
      if (!Number.isFinite(ms) || ms <= 0) continue;
      if (buf.trim().length > 0) {
        flush();
      }
      const last = out[out.length - 1];
      if (last) {
        ;
        last.preBreakMs = (last.preBreakMs ?? 0) + ms;
      } else {
        pendingBreakMs += ms;
      }
    } else if (tag === "whisper") {
      flush();
      whisper = true;
    } else if (tag === "/whisper") {
      flush();
      whisper = false;
    } else {
    }
  }
  buf += raw.slice(lastEnd);
  flush();
  if (pendingBreakMs > 0) {
    const last = out[out.length - 1];
    if (last) {
      ;
      last.preBreakMs = (last.preBreakMs ?? 0) + pendingBreakMs;
    }
    pendingBreakMs = 0;
  }
  if (whisper) {
    for (const s of out) s.whisper = false;
  }
  return out;
}

// src/tts-queue.ts
var MP3_MAGIC = 255;
var TTS_METADATA = { wordBoundaryEnabled: false, sentenceBoundaryEnabled: false };
var EDGE_VOICES_TIMEOUT_MS = 3e4;
function prosodyFromRate(rate) {
  if (rate !== void 0 && rate > 0 && rate !== 1) return { rate };
  return void 0;
}
function isValidMp3(buf) {
  return buf.length > 0 && buf[0] === MP3_MAGIC;
}
function escapeXmlText(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&apos;");
}
function isEmptyAudio(buf) {
  if (buf.length === 0) return true;
  if (buf.length !== 44) return false;
  if (buf.toString("latin1", 0, 4) !== "RIFF") return false;
  if (buf.toString("latin1", 8, 12) !== "WAVE") return false;
  if (buf.toString("latin1", 36, 40) !== "data") return false;
  return buf.readUInt32LE(40) === 0;
}
var EdgeTtsEngine = class {
  voice;
  rate;
  constructor(voice = "zh-CN-XiaoxiaoNeural", rate) {
    this.voice = voice;
    this.rate = rate;
  }
  mime = "audio/mpeg";
  updateVoice(voice, rate) {
    this.voice = voice;
    if (rate !== void 0 && Number.isFinite(rate)) this.rate = rate;
  }
  async synthesize(text, options = {}) {
    const spoken = escapeXmlText(stripEmotionTags(text));
    if (spoken.trim() === "") return Buffer.alloc(0);
    const tts = new MsEdgeTTS();
    try {
      await tts.setMetadata(
        options.voice ?? this.voice,
        OUTPUT_FORMAT.AUDIO_24KHZ_48KBITRATE_MONO_MP3,
        TTS_METADATA
      );
      const { audioStream } = tts.toStream(spoken, prosodyFromRate(options.rate ?? this.rate));
      const chunks = [];
      for await (const chunk of audioStream) chunks.push(chunk);
      const buf = Buffer.concat(chunks);
      if (!isValidMp3(buf)) throw new Error("empty or invalid audio");
      return buf;
    } finally {
      try {
        await tts.close();
      } catch {
      }
    }
  }
  async close() {
  }
};
var edgeVoicesCache = null;
async function listEdgeVoices(force = false) {
  if (edgeVoicesCache && !force) return edgeVoicesCache;
  const tts = new MsEdgeTTS();
  try {
    let voicesTimer = null;
    const voices = await Promise.race([
      tts.getVoices(),
      new Promise((_, reject) => {
        voicesTimer = setTimeout(() => reject(new Error("edge voices request timed out")), EDGE_VOICES_TIMEOUT_MS);
      })
    ]).finally(() => {
      if (voicesTimer) clearTimeout(voicesTimer);
    });
    edgeVoicesCache = voices.map((v) => ({
      ShortName: v.ShortName,
      Locale: v.Locale,
      Gender: v.Gender,
      FriendlyName: v.FriendlyName
    }));
    return edgeVoicesCache;
  } finally {
    try {
      await tts.close();
    } catch {
    }
  }
}
var TtsQueue = class {
  /**
   * 各会话队列：只在**有活**时存在（见 pump finally 的退休处置），不再随历史会话数增长。
   *
   * R15（Issue #15 T-3）：本表原先每会话留一个空 SessionQueue 且永不回收，而所有对外
   * 清理入口都是 cancel()（index.ts:1007/1376/1390/1409/1665）——cancel 按设计必须保留
   * 队列对象，故 turn 退出路径下这张表只增不减（量级 = 历史会话数，非本次回归）。
   * 真正的回收点只能是「队列已排空、泵已收工」的那一刻。
   *
   * 【不变式：seq 单调】回收绝不能让某会话的 sentenceId 归零或回退——client 的
   * rejectSeqUpTo（client.tsx:1359/1385，`frame.sentenceId <= rejectLine` 即丢弃）记的是
   * 「本会话已见过的最大句号」，新句必须**大于**它，否则新回复前 N 句被静默丢弃（静音）。
   * 这正是 index.ts:1376/1409 与 setEngine B1 都写明「用 cancel 而非 prune 保 seq 连续」
   * 的同一条理由，也是不用 prune() 回收的直接原因。
   * 故 seq 已上移为本类的全局计数器（见下方 seq），会话维度不再持有任何 seq 状态——
   * 这样「回收」与「seq 不归零」不再互相排斥：会话回来时拿到的必然是一个比它历史上
   * 任何一句都大的句号（全局单调在会话上的投影必然严格递增）。
   */
  queues = /* @__PURE__ */ new Map();
  /**
   * 句序号发号器（**全局**，非每会话）。
   *
   * 为什么不是每会话：每会话 seq 必须活到会话彻底消失，可任何「活到会话消失」的形态
   * （墓碑表、有界 LRU 淘汰）都只是把同一张永不回收的表换个名字——LRU 淘汰后该会话再回来
   * 仍会 seq 归零，仍撞 rejectSeqUpTo。全局计数器则让「会话维度 seq 状态」根本不存在，
   * 回收得以无条件成立。
   *
   * 只在成功广播时自增（pump 的 `const sentenceId = this.seq++` 位于成功分支内），
   * 与改动前逐字等价。client 只把 sentenceId 用于**比较**（`> rejectLine` 放行、
   * `!== curSentenceId` 重建拼帧缓冲），从不假设连续或小值——见 client.tsx:1354
   * 「host 的 sentenceId 跨回合单调递增，新回合必然 > 线」。
   */
  seq = 0;
  listeners = /* @__PURE__ */ new Set();
  engine;
  /** TTS 全体不可达通知（每会话去重，成功后复位）。 */
  onError;
  /** 单句合成重试耗尽被跳过通知（每句一次；此前是静默丢句——真机「回复偶尔不朗读」根因之一）。 */
  onSkip;
  /** 某会话队列跑空（无在途、无积压）：「说完就关」据此判定朗读真正结束——客户端只能看到
   *  句间的播放空档，分不清「后面还有句子在合成」与「已全部读完」。
   *
   *  R15（Issue #15 Important-2）：本回调的口径取决于 awaitTurnEnd，见下方 openTurns。 */
  onIdle;
  /**
   * R15（Issue #15 Important-2）：回合闸门——「队列排空」≠「回合结束」。
   *
   * 缺陷（逐行核对 8c0368a）：onIdle 原义「排空一次即发」，无「回合是否真正结束」约束。
   * 而 host 在**每一条** llm/stream 的 finally 都把 turn 打回 listening（index.ts:1875，
   * 于是一条流结束 = 回合结束）。带工具调用的 agent 在第一轮流结束与第二轮开始之间存在
   * 秒级到分钟级的空档：第一段文本排空 → tts-idle → 客户端 ttsDrained=true；播完、playing
   * =false → mic-gesture.onceFinished（sawRun && !running && ttsDrained && !playing）成立
   * → 维持 ONCE_SETTLE_MS=1.2s 后 exitMode('manual') → /toggle off → queue.cancel；第二轮
   * llm/stream 因 activeVoiceSession 已空不再被 tap（index.ts:957）→ **后半段回复永不朗读**，
   * 麦克风在 AI 还在干活时熄灭。非工具场景的等价触发是句间停顿 >1.2s。
   *
   * 修法：onIdle 降级为「候选空闲」，真正的结束信号由 host 在**回合真正结束**时确认——
   * 即宿主 dsh 的 `agent/turn-stopping`（契约原文「the model owes no response: no live
   * tool calls, no fresh steering」，正是工具空档与真回合结束的判别式），在那儿调
   * noteTurnEnd(sessionId)。队列侧：enqueue 视为「新回合在途」标记，pump finally 的排空
   * 回调在该标记存在时**不发**（那只是句间空档）；回合结束标记被清且队列已排空时补发。
   *
   * awaitTurnEnd 缺省 false = 闸门关闭，保持原语义（排空即发），供尚未接回合结束钩子的
   * 宿主零行为变化地兼容；置 true 后本集合才被 enqueue 填充。
   */
  awaitTurnEnd;
  /** awaitTurnEnd 开启时：某会话当前有一轮可能仍在进行（已入队过文本、回合尚未确认结束）。 */
  openTurns = /* @__PURE__ */ new Set();
  /**
   * M7②（Issue #15 R11）：退避重启计时器句柄集。
   *
   * 漏洞：pump 失败退避路径的 `setTimeout(() => void this.pump(...), delay)` 不持句柄，
   * close() 无从取消。close 后计时器仍触发 pump → 若该会话还有 pending，就会调
   * engine.synthesize → 本地引擎 ensureReady() **重新 fork 合成子进程**——插件已卸载、
   * 引擎已 close，却留下一个孤儿子进程（热重载每代泄漏一个，CPU/内存常驻）。
   * 修法：句柄入集，触发即自摘；close() 时全量 clearTimeout（与 asr-host senseP、
   * listEdgeVoices 的 timer 修法同构）。
   */
  backoffTimers = /* @__PURE__ */ new Set();
  /**
   * R13（Issue #15 Important-1）：重试 sleep 计时器句柄集。
   *
   * 与 backoffTimers 分开而非合并：语义不同（一个是「失败退避后重启泵」，一个是
   * 「单句合成重试间的 sleep」），且 sleep 由 retrySleep 统一创建、落定即自摘。
   * 不持句柄的后果：close 落进 400/800ms sleep 窗口时，该 timer 仍会走完——
   * 残留一个 ref'd Timeout 拖住事件循环 ≤800ms（同型反模式家族第 5 处，
   * 前三处：asr-host senseP、listEdgeVoices、warmupSense；第四处 backoffTimers）。
   */
  retryTimers = /* @__PURE__ */ new Set();
  /** R13：可取消的重试 sleep——句柄入集供 close() 取消；落定即自摘。 */
  retrySleep(ms) {
    if (this.closed) return Promise.resolve();
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        this.retryTimers.delete(timer);
        resolve();
      }, ms);
      this.retryTimers.add(timer);
    });
  }
  constructor(options) {
    this.engine = options.engine;
    this.onError = options.onError;
    this.onSkip = options.onSkip;
    this.onIdle = options.onIdle;
    this.awaitTurnEnd = options.awaitTurnEnd === true;
  }
  /**
   * R15（Issue #15 Important-2）：host 确认该会话的一轮作答**真正结束**（不会再有新的
   * llm/stream）时调用，接线点为宿主 dsh 的 `agent/turn-stopping`。
   *
   * 置 awaitTurnEnd 后：关闭该会话的在途标记；若队列此刻已排空则立即补发一次 onIdle
   * ——回合结束时 pump 的 finally 可能早已过去（最后一句播完时就发过「候选空闲」被
   * 闸门拦下），此时没有后续回调可等，必须在此补发。
   * 幂等/重复调用无害：重复触发时消费侧有 `!ttsDrained` 去重护栏（client.tsx:1189）。
   */
  noteTurnEnd(sessionId) {
    if (this.closed) return;
    this.openTurns.delete(sessionId);
    if (!this.awaitTurnEnd) return;
    if (!this.isIdle(sessionId)) return;
    try {
      this.onIdle?.(sessionId);
    } catch {
    }
  }
  /** 该会话当前无在途合成、无积压（未建队列也算空闲）。 */
  isIdle(sessionId) {
    const q = this.queues.get(sessionId);
    return !q || !q.busy && q.pending.length === 0;
  }
  /**
   * R15（Issue #15 T-4）：回合作废——本轮**不会再有任何后续文本**，但与「回合正常结束」
   * （noteTurnEnd）不同的一点是：**不补发 onIdle**。
   *
   * 为何队列侧无法自行判断 abort（宿主侧实测 + 本文件调用点实测）：
   *  · 回合 abort 时宿主是 `catch → 直接重抛`：不派发 `agent/turn-stopping`、不派发
   *    `agent/error`、不经本插件的任何入口——队列侧**收不到任何信号**，无法自愈；
   *  · 唯一在队列侧与 abort「沾边」的入口是 cancel()，但它是五处语义完全不同的出口的
   *    共同出口，对队列可见的输入**只有 sessionId 一个**：/cancel「停止朗读」与用户打断
   *    (index.ts:1665)、进模式 (:1376)、退模式 (:1409)、会话让出 (:1007/:1390)。
   *    在 cancel() 里清 openTurns，「停止朗读」后队列排空就会补发一条 tts-idle → 把
   *    R15 Important-1（回合间空档误发致 mode:once 提前退出、后半段永不朗读）**重新
   *    引进来**。故这里刻意不挂 cancel()，宁可把区分权交回 host。
   *
   * 语义（与 noteTurnEnd 唯一的差别就是不发 onIdle）：
   *  · 只合闸门（清 openTurns）：此后该会话若还有句子读尽，pump finally 会正常补发；
   *    下一回合的 enqueue 也照常重开闸门，语义与 noteTurnEnd 之后完全一致。
   *  · 不主动补发：abort 时队列可能已空，此时该不该让 mode:once 立即收尾取决于 host
   *    的上下文（用户是否马上又要开口），队列无从判断。若 host 判定「abort 前已读尽」，
   *    在其后紧接一次 noteTurnEnd(sessionId) 即可补发（两次调用对 openTurns 幂等）。
   *
   * 接线点：宿主侧在判定回合被 abort / blocked / 空首步时调用（这三类路径都不派发
   * agent/turn-stopping——error 半边已由 ctx.on('agent/error') 覆盖，见 index.ts:1112）。
   */
  abandonTurn(sessionId) {
    this.openTurns.delete(sessionId);
  }
  /** 当前引擎音频 MIME（/preview 的 Content-Type 也用它）。 */
  get mime() {
    return this.engine.mime;
  }
  /**
   * 运行时切换引擎（设置面板「朗读引擎」即时生效 / Kokoro 精度切换 / 清理模型缓存后重建）：
   * 关闭旧引擎、作废所有会话的在途与积压；新句子用新引擎合成。
   *
   * B1（Issue #15 审查 Blocker）：早年实现只 `this.queues.clear()`——不升各队列的
   * epoch，也不保留队列本身，两处都错：
   *  1) 孤儿泵不停播：在途 synthesize 完成后 `item.epoch === q.epoch` 恒真，旧引擎的
   *     帧继续下行，与新引擎的帧双轨广播（prune() 已修同一反模式，setEngine 漏修）；
   *  2) seq 归零撞拒绝线：删队列后重入的新 SessionQueue seq 从 0 起算，而 client 的
   *     sentenceId 拒绝线（rejectSeqUpTo）是「旧会话遗留的最大句号」，新回复前 N 句
   *     全部 ≤ 线被静默丢弃——正是 /toggle 用 cancel 而非 prune 保 seq 连续的同一条
   *     不变式（index.ts:1177 进模式 cancel / :1191 让出旧会话 cancel(previous) / :1210 退出 cancel）。
   * 故此处与 cancel() 同构：逐队列 epoch++ + 清积压，但保留 SessionQueue 对象本身。
   * R15 T-3 后这条顾虑的一半已由构造消除：seq 是**全局**发号器（不再随队列存亡），
   * 故即便队列对象事后被回收，重入的会话拿到的句号仍必然 > 旧线；但 epoch 仍必须
   * 逐队列提升才能压住孤儿泵，那部分保留原样。
   */
  setEngine(engine) {
    const old = this.engine;
    this.engine = engine;
    for (const q of this.queues.values()) {
      q.epoch++;
      q.pending.length = 0;
    }
    void old.close().catch(() => {
    });
  }
  /** 动态更换音色/语速（设置即时生效；正在合成的句子不受影响）。 */
  updateVoice(voice, rate) {
    this.engine.updateVoice(voice, rate);
  }
  /** 当前引擎/模型现状（设置面板状态区轮询）。 */
  status() {
    const s = this.engine.status?.();
    if (s) return s;
    return { engine: "edge", ready: true, loading: false };
  }
  /** 触发当前引擎预热/下载模型并初始化（设置面板「下载」按钮；Edge 为无操作）。 */
  async prepare() {
    if (this.closed) return;
    await this.engine.prepare?.();
  }
  /**
   * 一次性合成（设置卡「试听」用）：委托当前引擎；不干扰朗读队列的在途合成。
   * 失败（含非法音色）抛错。
   */
  async synthesize(text, options = {}) {
    if (this.closed) throw new Error("tts queue closed");
    return this.engine.synthesize(text, options);
  }
  subscribe(listener) {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }
  /**
   * I-1（Issue #15 R12）：关闭闸门。close() 最早期置位，此后：
   *  · enqueue 不再入队/起泵；
   *  · pump 入口直接返回；
   *  · pump while 不再取新句（在途那句已发出的 synthesize 允许自然结束——无法取消，
   *    但其结果不再广播，因为客户端已断开、广播无接收方）；
   *  · finally 的 pending>0 分支不再重排（即时重启与 backoff timer 都不排）。
   * 缺此闸门时 close 后仍会触达 engine.synthesize，而 tts-local 的 close() 已把
   * child/childInit/ready 全置 null → ensureReady 首行守卫失效 → 重走模型校验 + fork
   * 新子进程（热重载每代泄漏一个孤儿子进程）。详见 close() 处注释。
   */
  closed = false;
  /** 为某会话入队一句；若泵空闲则启动。 */
  enqueue(sessionId, text) {
    if (this.closed) return;
    if (this.awaitTurnEnd) this.openTurns.add(sessionId);
    let q = this.queues.get(sessionId);
    if (!q) {
      q = { pending: [], busy: false, epoch: 0, errorNotified: false, backoff: 0, failStreak: 0 };
      this.queues.set(sessionId, q);
    }
    if (q.pending.length >= 500) {
      console.warn("[dsh-voice-mode] TTS queue overflow, dropping oldest sentence");
      q.pending.shift();
    }
    q.pending.push({ text, epoch: q.epoch });
    void this.pump(sessionId, q);
  }
  /**
   * 弃掉某会话的所有积压并作废正在合成的句子（打断）。之后入队的句子
   * 获得新 epoch 正常播放。同时立刻中止在途合成（本地引擎杀子进程释放 CPU）。
   */
  cancel(sessionId) {
    const q = this.queues.get(sessionId);
    if (q) {
      q.epoch++;
      q.pending.length = 0;
    }
    if (![...this.queues].some(([sid, other]) => sid !== sessionId && other.busy)) this.engine.interrupt?.();
  }
  /** 会话退出/被抢占时彻底清理其队列（防止 Map 长期累积）。 */
  prune(sessionId) {
    const q = this.queues.get(sessionId);
    if (q) {
      q.epoch++;
      q.pending.length = 0;
    }
    this.openTurns.delete(sessionId);
    this.queues.delete(sessionId);
  }
  async pump(sessionId, q) {
    if (this.closed || q.busy) return;
    q.busy = true;
    try {
      while (q.pending.length > 0) {
        if (this.closed) break;
        const item = q.pending.shift();
        const MAX_SYNTH_ATTEMPTS = 3;
        let buf = null;
        for (let attempt = 0; attempt < MAX_SYNTH_ATTEMPTS; attempt++) {
          if (item.epoch !== q.epoch) break;
          if (this.closed) break;
          try {
            buf = await this.engine.synthesize(item.text);
            break;
          } catch (e) {
            console.warn(`[dsh-voice-mode] synthesis failed (${attempt + 1}/${MAX_SYNTH_ATTEMPTS}): ${String(e)}`);
            if (attempt < MAX_SYNTH_ATTEMPTS - 1) {
              await this.retrySleep(400 * (attempt + 1));
            }
          }
        }
        if (this.closed) break;
        if (item.epoch !== q.epoch) continue;
        if (buf === null) {
          q.failStreak++;
          this.onSkip?.(sessionId, item.text, q.failStreak);
          continue;
        }
        if (isEmptyAudio(buf)) continue;
        q.errorNotified = false;
        q.failStreak = 0;
        q.backoff = 0;
        const sentenceId = this.seq++;
        const mime = this.engine.mime;
        const dataFrame = {
          sessionId,
          sentenceId,
          chunkId: 0,
          final: false,
          audio: buf.toString("base64"),
          mime
        };
        for (const fn of this.listeners) {
          try {
            fn(dataFrame);
          } catch {
          }
        }
        const finalFrame = {
          sessionId,
          sentenceId,
          chunkId: 1,
          final: true,
          text: item.text,
          audio: "",
          mime
        };
        for (const fn of this.listeners) {
          try {
            fn(finalFrame);
          } catch {
          }
        }
      }
    } catch (e) {
      console.warn(`[dsh-voice-mode] TTS unavailable: sid=${sessionId} err=${String(e)}`);
      if (!q.errorNotified) {
        q.errorNotified = true;
        this.onError?.(sessionId);
      }
    } finally {
      q.busy = false;
      if (this.queues.get(sessionId) !== q) return;
      if (this.closed) return;
      if (q.pending.length === 0) {
        if (!this.openTurns.has(sessionId)) {
          try {
            this.onIdle?.(sessionId);
          } catch {
          }
        }
        this.queues.delete(sessionId);
      } else {
        const delay = q.errorNotified ? q.backoff : 0;
        q.backoff = Math.min(8e3, delay + 1e3);
        if (delay > 0) {
          const timer = setTimeout(() => {
            this.backoffTimers.delete(timer);
            void this.pump(sessionId, q);
          }, delay);
          this.backoffTimers.add(timer);
        } else void this.pump(sessionId, q);
      }
    }
  }
  async close() {
    this.closed = true;
    this.openTurns.clear();
    for (const timer of this.backoffTimers) clearTimeout(timer);
    this.backoffTimers.clear();
    for (const timer of this.retryTimers) clearTimeout(timer);
    this.retryTimers.clear();
    await this.engine.close();
  }
};

// src/tts-local.ts
import { fork } from "node:child_process";

// src/tts-runtime.ts
import { spawnSync } from "node:child_process";
var probeNode = (command, env) => {
  try {
    const r = spawnSync(
      command,
      ["-p", "JSON.stringify([process.versions.node, !!process.versions.electron, process.execPath])"],
      { env, encoding: "utf8", timeout: 5e3, windowsHide: true }
    );
    if (r.status !== 0) return null;
    const [version, electron, execPath] = JSON.parse(r.stdout.trim());
    return !electron && Number.parseInt(version, 10) >= 18 && typeof execPath === "string" ? execPath : null;
  } catch {
    return null;
  }
};
var NATIVE_RUNTIME_UNAVAILABLE = 'Local Kokoro cannot run inside the Electron-based desktop host (native add-on blocked: "External buffers are not allowed"). Install Node.js >= 18 on PATH (or set DSHVM_NODE to its path), or switch the read-aloud engine to Edge or VITS.';
var NativeRuntimeUnavailableError = class extends Error {
  code = "kokoro_needs_node";
  constructor() {
    super(NATIVE_RUNTIME_UNAVAILABLE);
    this.name = "NativeRuntimeUnavailableError";
  }
};
var cached;
function resolveNativeRuntime(opts = {}) {
  const env = opts.env ?? process.env;
  const useCache = opts.useCache ?? opts.env === void 0;
  if (useCache && cached !== void 0) return cached;
  const electron = opts.electron ?? !!process.versions.electron;
  let result;
  if (!electron) {
    result = { env };
  } else {
    const childEnv = { ...env };
    delete childEnv.ELECTRON_RUN_AS_NODE;
    const probe = opts.probe ?? probeNode;
    const candidates = [env.DSHVM_NODE, "node"].filter((c) => typeof c === "string" && c.length > 0);
    let execPath = null;
    for (const c of candidates) {
      execPath = probe(c, childEnv);
      if (execPath) break;
    }
    result = execPath ? { execPath, env: childEnv } : null;
  }
  if (useCache) cached = result;
  return result;
}

// src/tts-local.ts
import { fileURLToPath as fileURLToPath2 } from "node:url";
import { join as join3 } from "node:path";
import { statSync as statSync2 } from "node:fs";

// src/voice-catalog.ts
var VITS_SPEAKERS = [
  { name: "suyingxue", sid: 0, zh: "\u7D20\u6620\u96EA", en: "Su Yingxue", gender: "F" },
  { name: "gunian", sid: 1, zh: "\u987E\u5FF5", en: "Gu Nian", gender: "M" },
  { name: "fushiyu", sid: 2, zh: "\u5085\u65AF\u9047", en: "Fu Siyu", gender: "F" },
  { name: "bingjiao", sid: 3, zh: "\u51B0\u5A07", en: "Bing Jiao", gender: "M" },
  { name: "bazong", sid: 4, zh: "\u9738\u603B", en: "Ba Zong", gender: "M" }
];
var KOKORO_F0 = [196, 208];
var KOKORO_NAMED = {
  0: { name: "sveta", zh: "\u0421\u0432\u0435\u0442\u0430", en: "Sveta" },
  1: { name: "masha", zh: "\u041C\u0430\u0448\u0430", en: "Masha" }
};
var KOKORO_PINNED = [1, 0];
var KOKORO_VOICES = [
  ...KOKORO_PINNED.map((sid) => ({ name: KOKORO_NAMED[sid]?.name ?? String(sid), sid })),
  ...KOKORO_F0.map((_, sid) => ({ name: KOKORO_NAMED[sid]?.name ?? String(sid), sid })).filter(
    (v) => !KOKORO_PINNED.includes(v.sid)
  )
];

// src/tts-local.ts
var TTS_MODEL_REPO = "csukuangfj/sherpa-onnx-vits-zh-ll";
var KOKORO_MODEL_DIR_INT8 = "zaakirio/kokoro-ru";
var kokoroModelDir = (_m) => KOKORO_MODEL_DIR_INT8;
var TTS_MODEL_FILES = [
  { file: "model.onnx", sha256: "6c349bdd73dc928234dd7bc86929748bba32cd5264d32d915bf7b7aa0595965b" },
  { file: "lexicon.txt", sha256: "b3a82f16b286c424953dea3686039e7ab465fa8e15d87ef8abd0ec69175beb21" },
  { file: "tokens.txt", sha256: "34b035b9aeb070df6188b022f29c00e0e142c7ade9f25611ced65db5e9cc8402" },
  { file: "G_multisperaker_latest.json", sha256: "f31e4bf23827c3528fdf090fd7b6fb8e63333709b80670d40fa864f1fa9fadf3" },
  { file: "date.fst", sha256: "eb8aa079ae3cb81d8f4404992f39d61a0cb990947512b5b8d1e54d1f6980e718" },
  { file: "phone.fst", sha256: "1ac2b6fa56b1442320c4de7db08353bab8963a2b57f365eebcdd3a2d3562f8d7" },
  { file: "number.fst", sha256: "743f402181fcfebf76cc2f0546b71fa26476e626fbe4e460fb7b4c3a7a8bd5bd" }
];
function voiceToSid(voice) {
  const v = String(voice ?? "").trim().toLowerCase();
  if (/^\d+$/.test(v)) {
    const n = Number(v);
    if (n >= 0 && n < VITS_SPEAKERS.length) return n;
  }
  const hit = VITS_SPEAKERS.find((s) => s.name.toLowerCase() === v);
  return hit ? hit.sid : 0;
}
function kokoroVoiceToSid(voice) {
  const v = String(voice ?? "").trim().toLowerCase();
  if (/^\d+$/.test(v)) {
    const n = Number(v);
    if (n >= 0 && n <= KOKORO_VOICES.length - 1) return n;
  }
  const hit = KOKORO_VOICES.find((s) => s.name.toLowerCase() === v);
  return hit ? hit.sid : 1;
}
function floatToPcm16(samples) {
  const buf = Buffer.alloc(samples.length * 2);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    buf.writeInt16LE(s < 0 ? s * 32768 : s * 32767, i * 2);
  }
  return buf;
}
function pcmToWav(pcm, sampleRate) {
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + pcm.length, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36);
  header.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([header, pcm]);
}
var VITS_SPEC = {
  files: TTS_MODEL_FILES,
  workerPaths: (dir) => ({
    model: join3(dir, "model.onnx"),
    lexicon: join3(dir, "lexicon.txt"),
    tokens: join3(dir, "tokens.txt"),
    date: join3(dir, "date.fst"),
    phone: join3(dir, "phone.fst"),
    number: join3(dir, "number.fst")
  }),
  defaultVoice: "suyingxue",
  toSid: voiceToSid
};
var kokoroWorkerPaths = (dir) => ({
  model: join3(dir, "model.onnx"),
  voices: join3(dir, "voices.bin"),
  tokens: join3(dir, "tokens.txt")
});
function kokoroSpec(_model) {
  return {
    files: [],
    workerPaths: (dir) => kokoroWorkerPaths(dir),
    defaultVoice: "masha",
    toSid: kokoroVoiceToSid
  };
}
var CALL_TIMEOUT_MS = 12e4;
function createSherpaLocalEngine(options) {
  const { cacheDir, modelHost, allowCustomHost, broadcast } = options;
  let downloadProgress = null;
  const trackedBroadcast = (event, payload) => {
    if (event === "asr-progress" && payload && typeof payload === "object") {
      const pr = payload;
      if (typeof pr.file === "string" && typeof pr.percent === "number") {
        downloadProgress = { file: pr.file, percent: Math.min(100, Math.max(0, pr.percent)) };
      }
    }
    broadcast(event, payload);
  };
  const kokoroModel = options.model ?? "int8";
  const spec = options.kind === "kokoro" ? kokoroSpec(kokoroModel) : VITS_SPEC;
  const repoName = options.kind === "kokoro" ? kokoroModelDir(kokoroModel) : TTS_MODEL_REPO;
  const repoDir = join3(cacheDir, repoName);
  const workerPath = fileURLToPath2(
    new URL(options.kind === "kokoro" ? "./ru-tts-worker.cjs" : "./tts-vits-worker.cjs", import.meta.url)
  );
  let child = null;
  let childInit = false;
  let closed = false;
  const dlAbort = new AbortController();
  let starting = false;
  const respawnChild = async () => {
    if (child) {
      child.kill();
      child = null;
    }
    childInit = false;
    if (!starting) ready = null;
  };
  let voice = spec.defaultVoice;
  let speed = 1;
  let ready = null;
  let engineLoading = false;
  let engineError;
  let engineErrorCode;
  let nextId = 1;
  const pending = /* @__PURE__ */ new Map();
  const call = (msg, timeoutMs = CALL_TIMEOUT_MS) => new Promise((resolve, reject) => {
    if (!child) {
      reject(new Error("tts child not running"));
      return;
    }
    const id = nextId++;
    const timer = setTimeout(() => {
      if (!pending.delete(id)) return;
      reject(new Error(`tts child call timeout: ${String(msg.type)}`));
      if (msg.type !== "close") void respawnChild();
    }, timeoutMs);
    pending.set(id, {
      owner: child,
      resolve: (r) => (clearTimeout(timer), resolve(r)),
      reject: (e) => (clearTimeout(timer), reject(e))
    });
    child.send({ id, ...msg });
  });
  const rejectAll = (e, owner) => {
    for (const [id, p] of pending) {
      if (owner && p.owner !== owner) continue;
      pending.delete(id);
      p.reject(e);
    }
  };
  const ensureReady = () => {
    if (ready && child) return ready;
    if (!ready) {
      ready = (async () => {
        starting = true;
        engineLoading = true;
        engineError = void 0;
        engineErrorCode = void 0;
        downloadProgress = null;
        try {
          if (closed) return;
          if (!child || !childInit) {
            for (const f of spec.files) {
              const ok = await ensureModelFile({
                repo: repoName,
                repoDir,
                spec: f,
                primaryHost: modelHost(),
                allowCustomHost,
                broadcast: trackedBroadcast,
                signal: dlAbort.signal
              });
              if (closed) return;
              if (!ok) throw new Error("local TTS model download/verify failed: " + f.file);
            }
            if (options.kind === "kokoro" && spec.files.length > 0) {
              const treeOk = await ensureModelTree({
                repo: repoName,
                repoDir,
                subdir: "espeak-ng-data",
                primaryHost: modelHost(),
                allowCustomHost,
                broadcast: trackedBroadcast,
                signal: dlAbort.signal,
                // I-3：本地完整则 0 网络；无清单的旧缓存以哨兵 + 文件数兜底。
                // 取上游整树文件数（HF API 实测 int8/fp32 均 355）：低于它一律联网补全，半成品树不会被误判完整。
                sentinel: "espeak-ng-data/phontab",
                minFiles: 355
              });
              if (closed) return;
              if (!treeOk) throw new Error("local TTS model download failed: espeak-ng-data");
            }
            if (!child) {
              const runtime = options.kind === "kokoro" ? resolveNativeRuntime() : { env: process.env };
              if (!runtime) throw new NativeRuntimeUnavailableError();
              child = fork(workerPath, [], {
                stdio: ["ignore", "ignore", "pipe", "ipc"],
                ...runtime.execPath ? { execPath: runtime.execPath, env: runtime.env } : {}
              });
              const me = child;
              let stderrTail = "";
              child.stderr?.on("data", (chunk) => {
                stderrTail += String(chunk);
                const lines = stderrTail.split("\n");
                stderrTail = lines.pop() ?? "";
                for (const line of lines) {
                  const s = line.trim();
                  if (!s) continue;
                  if (/Skip unknown phonemes/.test(s)) continue;
                  console.error(`[tts-worker:${options.kind}] ${s}`);
                }
              });
              child.on("message", (m) => {
                const p = pending.get(m.id);
                if (!p) return;
                pending.delete(m.id);
                if (m.ok) p.resolve(m);
                else p.reject(new Error(m.error ?? "tts child error"));
              });
              child.on("error", (e) => {
                rejectAll(e instanceof Error ? e : new Error(String(e)), me);
                if (child !== me) return;
                child = null;
                childInit = false;
                if (!starting) ready = null;
              });
              child.on("exit", (code) => {
                rejectAll(new Error(`tts child exited with code ${code}`), me);
                if (child !== me) return;
                child = null;
                childInit = false;
                if (!starting) ready = null;
              });
            }
            if (!childInit) {
              const init = await call({ type: "init", kind: options.kind, paths: spec.workerPaths(repoDir) });
              if (closed) return;
              if (!init.ok) throw new Error(init.error ?? "tts child init failed");
              childInit = true;
            }
          }
          broadcast("tts-ready", { engine: options.kind, worker: true });
        } catch (e) {
          engineError = e instanceof Error ? e.message : String(e);
          engineErrorCode = e instanceof NativeRuntimeUnavailableError ? e.code : void 0;
          throw e;
        } finally {
          engineLoading = false;
          starting = false;
          downloadProgress = null;
        }
      })().then(
        (v) => {
          return v;
        },
        (e) => {
          ready = null;
          throw e;
        }
      );
    }
    return ready;
  };
  return {
    mime: "audio/wav",
    updateVoice(nextVoice, nextRate) {
      voice = nextVoice || voice;
      if (typeof nextRate === "number" && Number.isFinite(nextRate)) {
        speed = Math.min(2, Math.max(0.5, nextRate));
      }
    },
    status() {
      const files = spec.files.map((f) => {
        const p = join3(repoDir, f.file);
        let exists = false;
        let size = 0;
        try {
          const st = statSync2(p);
          exists = st.isFile();
          size = st.size;
        } catch {
        }
        return { name: f.file, exists, size };
      });
      return {
        engine: options.kind,
        ready: childInit === true,
        loading: engineLoading,
        error: engineError,
        errorCode: engineErrorCode,
        progress: downloadProgress ?? void 0,
        local: {
          repo: repoName,
          ready: files.every((f) => f.exists),
          loading: engineLoading,
          error: engineError,
          files
        }
      };
    },
    // 设置面板「下载」按钮：无文本也触发模型下载 + 子进程初始化（与首次合成路径一致）。
    prepare: () => ensureReady(),
    async synthesize(text, opts = {}) {
      if (closed) throw new Error("local TTS engine closed");
      await ensureReady();
      if (closed) throw new Error("local TTS engine closed");
      const sid = spec.toSid(opts.voice ?? voice);
      const spd = typeof opts.rate === "number" && Number.isFinite(opts.rate) ? Math.min(2, Math.max(0.5, opts.rate)) : speed;
      const segments = parseEmotionTags(text);
      if (segments.length === 0) {
        return Buffer.alloc(0);
      }
      const chunks = [];
      let resolvedSampleRate = 0;
      let needsRespawn = true;
      for (let i = 0; i < segments.length; i++) {
        const seg = segments[i];
        const preBreak = seg.preBreakMs ?? 0;
        let res = await call({ type: "synth", text: seg.text, sid, speed: spd });
        if (!res.ok && /Aborted/.test(res.error ?? "") && needsRespawn) {
          await respawnChild();
          res = await call({ type: "synth", text: seg.text, sid, speed: spd });
          needsRespawn = false;
        }
        if (!res.ok) throw new Error(res.error ?? "local TTS synthesis failed");
        if (typeof res.samples !== "string" || res.samples.length === 0) continue;
        const bytes = Buffer.from(res.samples, "base64");
        const samples = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
        if (samples.length === 0) continue;
        const sr = res.sampleRate || 16e3;
        if (!resolvedSampleRate) resolvedSampleRate = sr;
        if (seg.whisper) {
          for (let j = 0; j < samples.length; j++) samples[j] *= 0.5;
        }
        chunks.push(samples);
        if (preBreak > 0 && resolvedSampleRate) {
          chunks.push(new Float32Array(Math.round(resolvedSampleRate * preBreak / 1e3)));
        }
      }
      if (chunks.length === 0) {
        return Buffer.alloc(0);
      }
      const total = chunks.reduce((acc, c) => acc + c.length, 0);
      const merged = new Float32Array(total);
      let off = 0;
      for (const c of chunks) {
        merged.set(c, off);
        off += c.length;
      }
      return pcmToWav(floatToPcm16(merged), resolvedSampleRate || 16e3);
    },
    async close() {
      closed = true;
      dlAbort.abort();
      const c = child;
      if (c) {
        try {
          await call({ type: "close" }, 100);
        } catch {
        }
        try {
          c.kill();
        } catch {
        }
      }
      child = null;
      childInit = false;
      ready = null;
    },
    interrupt() {
      void respawnChild();
    }
  };
}
function createSherpaVitsEngine(options) {
  return createSherpaLocalEngine({ ...options, kind: "vits" });
}
function createSherpaKokoroEngine(options) {
  return createSherpaLocalEngine({ ...options, kind: "kokoro" });
}

// src/security.ts
var LOOPBACK_ADDRESSES = /* @__PURE__ */ new Set(["127.0.0.1", "::1", "::ffff:127.0.0.1"]);
function isLoopbackRequest(req) {
  const addr = req.socket.remoteAddress ?? "";
  return LOOPBACK_ADDRESSES.has(addr);
}
function sameOriginRequest(req) {
  const origin = req.headers.origin;
  if (!origin) return true;
  try {
    const originHost = new URL(origin).host;
    const xfh = req.headers["x-forwarded-host"];
    const host = (typeof xfh === "string" && xfh ? xfh.split(",")[0].trim() : "") || req.headers.host;
    if (!host) return false;
    return originHost === host;
  } catch {
    return false;
  }
}
var RateLimiter = class {
  buckets = /* @__PURE__ */ new Map();
  maxKeys;
  constructor(maxKeys = 1e4) {
    this.maxKeys = maxKeys;
  }
  /** 命中一次；返回是否允许。maxHits 次 / windowMs 毫秒。 */
  hit(key, maxHits, windowMs) {
    const now = Date.now();
    const cutoff = now - windowMs;
    let bucket = this.buckets.get(key);
    if (!bucket) {
      if (this.buckets.size >= this.maxKeys) return false;
      bucket = [];
      this.buckets.set(key, bucket);
    }
    while (bucket.length > 0 && bucket[0] <= cutoff) bucket.shift();
    if (bucket.length >= maxHits) return false;
    bucket.push(now);
    return true;
  }
  /** 定期清理（由调用方在低频路径触发即可）。 */
  prune(now = Date.now(), windowMs) {
    const cutoff = now - windowMs;
    for (const [key, bucket] of this.buckets) {
      while (bucket.length > 0 && bucket[0] <= cutoff) bucket.shift();
      if (bucket.length === 0) this.buckets.delete(key);
    }
  }
};

// src/strings.ts
var zh = {
  stateVoiceMode: "\u8BED\u97F3\u6A21\u5F0F",
  ttsNoticeFail: "\u6717\u8BFB\u8FDE\u63A5\u5931\u8D25\uFF1A\u6B63\u5728\u91CD\u8BD5\u2026",
  ttsSkipNotice: "\u6709\u4E00\u53E5\u6717\u8BFB\u5931\u8D25\uFF0C\u5DF2\u8DF3\u8FC7\uFF08\u53EF\u91CD\u53D1\u8FD9\u6761\u6D88\u606F\uFF09",
  ttsUnavailableNotice: "\u6717\u8BFB\u8FDE\u7EED\u5931\u8D25\uFF1A\u4E91\u7AEF\u6717\u8BFB\u6682\u4E0D\u53EF\u7528\u3002\u53EF\u7A0D\u540E\u518D\u8BD5\uFF0C\u6216\u5728\u8BBE\u7F6E\u91CC\u628A\u6717\u8BFB\u5F15\u64CE\u5207\u6362\u4E3A\u672C\u5730 VITS / Kokoro",
  enterFail: "\u8FDB\u5165\u8BED\u97F3\u6A21\u5F0F\u5931\u8D25",
  disabled: "\u8BED\u97F3\u6A21\u5F0F\u5DF2\u7981\u7528\uFF08\u63D2\u4EF6 enabled=false\uFF09",
  sendFailKept: "\u53D1\u9001\u5931\u8D25\uFF0C\u5DF2\u4FDD\u7559\u5728\u8349\u7A3F",
  micDenied: "\u9EA6\u514B\u98CE\u88AB\u62D2\u7EDD\uFF1A\u8BF7\u5728\u6D4F\u89C8\u5668\u5730\u5740\u680F\u5141\u8BB8\u9EA6\u514B\u98CE\u6743\u9650",
  micBusy: "\u9EA6\u514B\u98CE\u88AB\u5176\u4ED6\u7A0B\u5E8F\u5360\u7528\uFF1A\u8BF7\u5173\u95ED\u5360\u7528\u7A0B\u5E8F\u540E\u91CD\u8BD5",
  micUnavailable: "\u9EA6\u514B\u98CE\u4E0D\u53EF\u7528",
  recognizing: "\u8BC6\u522B\u4E2D\u2026",
  // UX-I1 联动态：纠错在途 / 纠错放弃回退 / 附和词忽略 / 语音按钮态文案。
  correctFixing: "\u7EA0\u9519\u4E2D\u2026",
  correctAbandoned: "\u5DF2\u4FDD\u7559\u539F\u6587\u5728\u8349\u7A3F",
  backchannelDropped: "\u5DF2\u5FFD\u7565\u9644\u548C\u8BCD\uFF0C\u4E0D\u53D1\u9001",
  buttonLoading: "\u52A0\u8F7D\u4E2D\u2026",
  buttonListening: "\u8046\u542C\u4E2D",
  holdToTalk: "\u6309\u4F4F\u8BF4\u8BDD",
  releaseToSend: "\u677E\u5F00\u53D1\u9001",
  entering: "\u8FDB\u5165\u4E2D\u2026",
  voiceBtn: "\u8BED\u97F3",
  ariaActive: "\u8BED\u97F3\u6A21\u5F0F\u8FDB\u884C\u4E2D",
  ariaEnter: "\u8FDB\u5165\u8BED\u97F3\u5BF9\u8BDD\u6A21\u5F0F",
  titleHold: "\u8BED\u97F3\u6A21\u5F0F\u8FDB\u884C\u4E2D \xB7 \u6309\u4F4F\u8BF4\u8BDD\u3001\u677E\u624B\u53D1\u9001\uFF1B\u77ED\u6309\u9000\u51FA\uFF1BEsc/\u5931\u53BB\u7126\u70B9\u653E\u5F03\uFF1BCtrl+Shift+V \u9000\u51FA",
  titleToggle: "\u8BED\u97F3\u6A21\u5F0F\u8FDB\u884C\u4E2D \xB7 \u70B9\u51FB\u9000\u51FA\uFF08Ctrl+Shift+V\uFF09\xB7 \u8BF4\u8BDD\u65F6\u6309\u4F4F Ctrl \u7ACB\u5373\u53D1\u9001",
  titleEnter: "\u8FDB\u5165\u8BED\u97F3\u5BF9\u8BDD\u6A21\u5F0F\uFF08Ctrl+Shift+V\uFF09",
  loadingModel: "\u6B63\u5728\u52A0\u8F7D\u6A21\u578B\u2026",
  listening: "\u8046\u542C\u4E2D\u2026",
  thinking: "\u601D\u8003\u4E2D\u2026",
  barHold: "\u8BED\u97F3\u6A21\u5F0F \xB7 \u6309\u4F4F\u8BF4\u8BDD\uFF08\u77ED\u6309\u9000\u51FA\uFF09",
  barListening: "\u8BED\u97F3\u6A21\u5F0F \xB7 \u8046\u542C\u4E2D\u2026",
  wakeWord: "\u5524\u9192\u8BCD",
  sayWake: "\u8BF4\u300C{wake}\u300D\u5F00\u59CB",
  reading: "\u6717\u8BFB\u4E2D\u2026",
  recognitionFail: "\u8BC6\u522B\u5931\u8D25\uFF0C\u8BF7\u91CD\u8BD5",
  partialFail: "\u5B9E\u65F6\u5B57\u5E55\u6682\u65F6\u4E0D\u53EF\u7528\uFF08\u8BC6\u522B\u901A\u9053\u5F02\u5E38\uFF0C\u91CD\u8BD5\u4E2D\uFF09",
  detectFail: "\u6253\u65AD\u68C0\u6D4B\u6682\u65F6\u4E0D\u53EF\u7528\uFF08\u68C0\u6D4B\u901A\u9053\u5F02\u5E38\uFF0C\u91CD\u8BD5\u4E2D\uFF09",
  sseReconnect: "\u72B6\u6001\u540C\u6B65\u65AD\u8FDE\uFF0C\u91CD\u8FDE\u4E2D\u2026",
  diagCopy: "\u590D\u5236\u8BCA\u65AD",
  diagCopied: "\u8BCA\u65AD\u4FE1\u606F\u5DF2\u590D\u5236",
  diagCopyFailed: "\u590D\u5236\u5931\u8D25\uFF08\u6D4F\u89C8\u5668\u62D2\u7EDD\u4E86\u526A\u8D34\u677F\uFF09",
  sessionExpired: "\u8BED\u97F3\u4F1A\u8BDD\u5DF2\u65AD\u5F00\uFF0C\u6B63\u5728\u91CD\u8FDE\u2026",
  sessionExpiredFail: "\u8BED\u97F3\u4F1A\u8BDD\u91CD\u8FDE\u5931\u8D25\uFF0C\u8BF7\u91CD\u65B0\u5F00\u542F\u8BED\u97F3\u6A21\u5F0F",
  modelDownloadFail: "\u8BED\u97F3\u6A21\u578B\u4E0B\u8F7D\u5931\u8D25\uFF08{file}\uFF09\uFF1A\u8BF7\u68C0\u67E5\u7F51\u7EDC\u540E\u91CD\u65B0\u8FDB\u5165\u8BED\u97F3\u6A21\u5F0F\u91CD\u8BD5",
  startFail: "\u8BED\u97F3\u6A21\u5F0F\u542F\u52A8\u5931\u8D25\uFF1A{err}",
  holdDots: "\u6309\u4F4F\u8BF4\u8BDD\u2026",
  exit: "\u9000\u51FA",
  configUnavailableNote: "\uFF08\u8BBE\u7F6E\u6587\u6863\u672A\u5C31\u7EEA\uFF0C\u6B63\u5728\u91CD\u8BD5\uFF1B\u9762\u677F\u5C31\u7EEA\u540E\u4F1A\u81EA\u52A8\u51FA\u73B0\uFF09\u3002",
  // settings-form
  previewNameFirst: "\u8BF7\u5148\u586B\u5199\u97F3\u8272\u540D\uFF08ShortName\uFF09",
  previewDisabled: "\u8BED\u97F3\u6A21\u5F0F\u5DF2\u7981\u7528\uFF08\u63D2\u4EF6 enabled=false\uFF09\uFF0C\u65E0\u6CD5\u8BD5\u542C",
  // 批 G 任务 6：本地 TTS 引擎未下载模型时禁用试听按钮 + 提示。
  previewModelMissing: "\u8BF7\u5148\u4E0B\u8F7D\u672C\u5730\u6A21\u578B\uFF08\u89C1\u4E0A\u65B9\u300C\u6717\u8BFB\u5F15\u64CE\u300D\u72B6\u6001\uFF09",
  previewModelLoading: "\u672C\u5730\u6A21\u578B\u6B63\u5728\u4E0B\u8F7D\u4E2D\uFF0C\u8BF7\u7A0D\u5019",
  previewPlayFail: "\u8BD5\u542C\u5931\u8D25\uFF1A\u65E0\u6CD5\u64AD\u653E\u8BE5\u97F3\u8272",
  previewAutoplay: "\u6D4F\u89C8\u5668\u62E6\u622A\u4E86\u81EA\u52A8\u64AD\u653E\uFF0C\u8BF7\u518D\u70B9\u4E00\u6B21\u8BD5\u542C",
  previewCheck: "\u8BD5\u542C\u5931\u8D25\uFF1A\u8BF7\u68C0\u67E5\u7F51\u7EDC\u6216\u97F3\u8272\u540D\uFF08ShortName\uFF09\u662F\u5426\u6B63\u786E",
  previewRateLimited: "\u8BD5\u542C\u592A\u9891\u7E41\u4E86\uFF0C\u7A0D\u5019\u51E0\u79D2\u518D\u70B9\uFF08\u6BCF\u5206\u949F\u9650 20 \u6B21\uFF09",
  previewTimeout: "\u5408\u6210\u8D85\u65F6\uFF1A\u6A21\u578B\u4ECD\u5728\u52A0\u8F7D\uFF0C\u8BF7\u7A0D\u5019\u51E0\u79D2\u518D\u8BD5",
  previewBtnTitle: "\u8BD5\u542C\u5F53\u524D\u97F3\u8272\uFF08\u5F53\u524D\u8BED\u901F\uFF09",
  synthesizing: "\u5408\u6210\u4E2D\u2026",
  preview: "\u8BD5\u542C",
  custom: "\u81EA\u5B9A\u4E49",
  // settings rows
  voicePrev: "\u4E0A\u4E00\u97F3\u8272",
  voiceNext: "\u4E0B\u4E00\u97F3\u8272",
  descVoice: "Edge \u4E91\u7AEF\u97F3\u8272\uFF08\u8FDB\u5165\u65F6\u81EA\u52A8\u52A0\u8F7D\u5FAE\u8F6F\u5168\u90E8\u97F3\u8272\uFF0C\u5E38\u7528\u4E2D\u6587\u97F3\u8272\u7F6E\u9876\uFF0C\u4E0B\u62C9\u6216 \u25C0\u25B6 \u9009\uFF1B\u4E5F\u53EF\u300C\u81EA\u5B9A\u4E49\u300D\u586B ShortName\uFF09",
  descVoiceLocal: "\u672C\u5730\u97F3\u8272\uFF08vits \u4E94\u4E2A\u8BF4\u8BDD\u4EBA\u5168\u90E8\u5217\u51FA\uFF0C\u4E0B\u62C9\u9009\u6216 \u25C0\u25B6 \u5207\u6362\uFF1B\u65E0\u9700\u81EA\u5B9A\u4E49\uFF09",
  descTtsEngine: "\u672C\u5730 VITS\uFF08\u7EAF\u4E2D\u6587\uFF09/ \u672C\u5730 Kokoro\uFF08\u4E2D\u82F1\u5747\u53EF\uFF09/ Edge \u4E91\u7AEF\uFF08\u97F3\u8D28\u6700\u81EA\u7136\uFF0C\u6587\u672C\u4E0A\u5FAE\u8F6F\uFF09",
  engineVits: "\u672C\u5730 VITS",
  engineKokoro: "\u672C\u5730\u4E2D\u82F1",
  engineEdge: "Edge \u4E91\u7AEF",
  descVoiceKokoro: "Kokoro \u4E2D\u82F1\u97F3\u8272\uFF08103 \u4E2A\u5168\u90E8\u5217\u51FA\uFF0C\u4E0B\u62C9\u9009\u6216 \u25C0\u25B6 \u5207\u6362\uFF1B48-51 \u4E2D\u6587\u540D\uFF0C\u5176\u4F59\u6309\u7F16\u53F7+\u5B9E\u6D4B\u6027\u522B\u6807\u6CE8\uFF0C\u4E2D\u82F1\u6DF7\u8BFB\u5747\u53EF\uFF09",
  descRate: "\u6717\u8BFB\u8BED\u901F\u500D\u7387\uFF080.5 \u6162\u901F \uFF5E 2.0 \u5FEB\u901F\uFF0C1.1 \u9ED8\u8BA4\uFF1B\u8BA9\u56DE\u590D\u66F4\u7D27\u51D1\uFF09",
  descInterrupt: "\u53D1\u58F0\u6253\u65AD\u7075\u654F\u5EA6\uFF080 \u9AD8\u95E8\u69DB \u22480.3s \u786E\u8BA4 / 1 \u4E2D\u95E8\u69DB \u22480.2s / 2 \u4F4E\u95E8\u69DB \u22480.1s\uFF0C\u6700\u7075\u654F\uFF09\uFF1B\u503C\u8D8A\u4F4E\u95E8\u69DB\u8D8A\u9AD8\uFF0C\u8D8A\u96BE\u6253\u65AD",
  descBargeIn: "\u6253\u65AD\u65B9\u5F0F\uFF08detect \u81EA\u52A8\u63A2\u6D4B\u672C\u673A\u56DE\u58F0\u6D88\u9664\u72B6\u6001\u3001\u672A\u751F\u6548\u65F6\u5207\u4E3A\u957F\u6309\u6253\u65AD\uFF0C\u9ED8\u8BA4\uFF1Bauto \u5F3A\u5236\u81EA\u52A8\u6253\u65AD\uFF1A\u5F00\u53E3\u5373\u6253\u65AD\uFF0C\u8033\u673A/\u5B89\u9759\u73AF\u5883\u63A8\u8350\uFF1Bmanual \u624B\u52A8\u6253\u65AD\uFF1A\u5916\u653E\u63A8\u8350\u2014\u2014\u56DE\u58F0\u4E0D\u4F1A\u8BEF\u89E6\u53D1\u81EA\u6253\u65AD\uFF0C\u6309\u4F4F\u9EA6\u514B\u98CE/Ctrl \u663E\u5F0F\u6253\u65AD\uFF09\u3002\u6CE8\u610F\uFF1Amanual\uFF08detect \u5728\u65E0\u539F\u751F\u56DE\u58F0\u6D88\u9664\u7684\u673A\u5668\u4E0A\u4E5F\u4F1A\u81EA\u52A8\u843D\u5165\uFF09\u4E0B\u975E\u6309\u538B\u671F\u7684\u8BED\u97F3\u4E0D\u88AB\u8BC6\u522B\u2014\u2014toggle \u6A21\u5F0F\u4EC5\u64AD\u653E\u4E2D\u53EF\u957F\u6309\u8BF4\u8BDD\uFF0C\u5F85\u673A\u65F6\u65E0\u6CD5\u8BED\u97F3\u8F93\u5165\uFF0C\u8BF7\u6539\u7528 hold \u6A21\u5F0F\u6216 auto/detect",
  bargeInDetect: "\u81EA\u52A8\u63A2\u6D4B",
  bargeInAuto: "\u81EA\u52A8",
  bargeInManual: "\u624B\u52A8",
  descEchoGate: "\u56DE\u58F0\u95E8\u63A7\u9608\u503C\uFF08dB\uFF0C\u9ED8\u8BA4 6\uFF09\uFF1A\u81EA\u52A8\u6253\u65AD\u8981\u6C42\u6B8B\u5DEE\u9AD8\u4E8E\u56DE\u58F0\u5730\u677F\u6B64\u503C\u3002\u5F53\u524D ASR \u6A21\u578B\u9ED8\u8BA4\u539F\u751F AEC \u751F\u6548\u65F6\u6B64\u95E8\u63A7\u95F2\u7F6E\uFF1BSafari / \u8033\u673A\u7B49\u65E0\u539F\u751F AEC \u73AF\u5883\u4F1A\u515C\u5E95\u751F\u6548\u3002\u5916\u653E\u4ECD\u8BEF\u6253\u65AD\u8C03\u5927\uFF088~10\uFF09\uFF0C\u592A\u96BE\u6253\u65AD\u8C03\u5C0F\uFF083~4\uFF09",
  descShortcut: "\u8FDB\u5165/\u9000\u51FA\u8BED\u97F3\u6A21\u5F0F\u7684\u5FEB\u6377\u952E\uFF08\u5F62\u5982 Ctrl+Shift+V\uFF1B\u7559\u7A7A\u7981\u7528\u5FEB\u6377\u952E\uFF0C\u53EA\u7528\u9EA6\u514B\u98CE\u6309\u94AE\uFF1B\u907F\u514D\u6D4F\u89C8\u5668\u4FDD\u7559\u7EC4\u5408\u5982 Ctrl+W/N/T\uFF09",
  vadDetected: "VAD \u68C0\u6D4B\u5230\u8BED\u97F3",
  aecOff: "\u539F\u751F\u56DE\u58F0\u6D88\u9664\u672A\u751F\u6548",
  aecOffHint: "\u6D4F\u89C8\u5668\u539F\u751F\u56DE\u58F0\u6D88\u9664\u672A\u751F\u6548\uFF08\u5916\u653E\u53EF\u80FD\u81EA\u6253\u65AD\uFF09\uFF0C\u5EFA\u8BAE\u7528\u8033\u673A\u6216\u5207\u6362\u300C\u624B\u52A8\u6253\u65AD\u300D",
  // 批 7O（ADR-0006）：detect 模式第一级探测降级提示（状态条）。
  bargeInDetectFallback: "\u672C\u673A\u56DE\u58F0\u6D88\u9664\u672A\u751F\u6548\uFF0C\u5DF2\u5207\u4E3A\u957F\u6309\u6253\u65AD",
  // R1 收尾（fix-client 直连 tk() 调用）：连续 3 次回声拒断的解释出口 + 快捷键非法停用提示。
  echoRejectHint: "\u56DE\u58F0\u8F83\u5F3A\uFF0C\u6253\u65AD\u53EF\u80FD\u4E0D\u7075\u654F\u2014\u2014\u8FD9\u662F\u56DE\u58F0\u9632\u62A4\u5728\u751F\u6548\uFF0C\u53EF\u6234\u8033\u673A\u6216\u6539\u7528\u300C\u624B\u52A8\u6253\u65AD\u300D",
  shortcutInvalid: "\u5FEB\u6377\u952E\u65E0\u6548\uFF0C\u5DF2\u505C\u7528\uFF08\u53EF\u5728\u8BBE\u7F6E\u91CC\u6539\u4E3A\u5982 Ctrl+Shift+V\uFF1B\u7559\u7A7A\u8868\u793A\u53EA\u7528\u9EA6\u514B\u98CE\u6309\u94AE\uFF09",
  descComposerShortcut: "\u8F93\u5165\u6846\u6709\u7126\u70B9\u65F6\u7684\u5207\u6362\u5FEB\u6377\u952E\uFF08\u9ED8\u8BA4 Alt/Option+Enter\uFF1B\u7559\u7A7A\u5219\u53EA\u5728\u8F93\u5165\u6846\u5916\u751F\u6548\uFF09",
  composerShortcutLabel: "\u8FDB\u5165 / \u9000\u51FA\u8BED\u97F3\u6A21\u5F0F\uFF08\u8F93\u5165\u6846\u5185\uFF09",
  shortcutRecord: "\u5F55\u5236",
  shortcutRecording: "\u6309\u4E0B\u7EC4\u5408\u952E\u2026",
  shortcutCancel: "\u53D6\u6D88",
  shortcutNeedModifier: "\u9700\u8981\u81F3\u5C11\u4E00\u4E2A\u4FEE\u9970\u952E",
  shortcutUnsupportedKey: "\u53EA\u652F\u6301\u5B57\u6BCD\u3001\u6570\u5B57\u3001Enter\u3001Space\u3001Tab",
  shortcutPlaceholder: "\u70B9\u5F55\u5236\u540E\u6309\u7EC4\u5408\u952E",
  shortcutReset: "\u6062\u590D\u9ED8\u8BA4",
  shortcutNote: "\u8F93\u5165\u6846\u5185\u53EF\u7528 Alt/Option+Enter \u5207\u6362\uFF1B\u652F\u6301 Cmd / Ctrl / Alt / Shift \u7EC4\u5408\uFF1B\u7559\u7A7A\u8868\u793A\u53EA\u7528\u9EA6\u514B\u98CE\u6309\u94AE\u3002\u5F55\u5236\u65F6 Backspace \u6E05\u7A7A\u3001Esc \u53D6\u6D88\uFF1B\u907F\u514D\u6D4F\u89C8\u5668\u5360\u7528\u7684\u7EC4\u5408\uFF08\u5982 Cmd+W / T / N / L\uFF09\u3002\u4FEE\u6539\u540E\u7ACB\u5373\u751F\u6548\uFF0C\u65E0\u9700\u5237\u65B0\u3002",
  secShortcut: "\u5FEB\u6377\u952E",
  elapsedHint: "\u5F53\u524D\u8BED\u97F3\u4F1A\u8BDD\u5DF2\u7528\u65F6\u957F",
  botLevelsHint: "AI \u6717\u8BFB\u97F3\u91CF\uFF08\u84DD\u8272\u6761 = \u6717\u8BFB\uFF0C\u7EFF\u8272\u6761 = \u9EA6\u514B\u98CE\uFF09",
  interruptConfirm: "\u6253\u65AD\u786E\u8BA4",
  sev0: "0 \u9AD8\u95E8\u69DB",
  sev1: "1 \u4E2D",
  sev2: "2 \u4F4E",
  // 数值单位（设置卡片「当前生效值」用：中文汉字单位、英文国际缩写，避免「1500 ms」直出）
  unitMs: "\u6BEB\u79D2",
  unitMin: "\u5206\u949F",
  unitDb: "\u5206\u8D1D",
  descSilence: "\u8BF4\u5B8C\u6574\u4E00\u53E5\u7684\u9759\u97F3\u505C\u987F\u6BEB\u79D2\u6570\uFF08\u9ED8\u8BA4 1500 \u6BEB\u79D2\uFF0C\u7ED9\u601D\u8003\u505C\u987F\u7559\u7A7A\u95F4\uFF1B\u81F3\u5C11 250ms \u8BED\u97F3\u624D\u5224\u53E5\uFF0C\u9632\u77ED\u4FC3\u566A\u58F0\u8BEF\u89E6\u53D1\uFF09",
  descIdle: "\u65E0\u6D3B\u52A8\u81EA\u52A8\u9000\u51FA\u8BED\u97F3\u6A21\u5F0F\u7684\u5206\u949F\u6570\uFF08\u9ED8\u8BA4 5\uFF1B\u9000\u51FA\u524D 30 \u79D2\u4F1A\u6709\u5012\u6570\u9884\u8B66\uFF09",
  descModelHost: "ASR \u6A21\u578B\u4E0B\u8F7D\u6E90\uFF08\u5B98\u65B9\u6E90 / \u56FD\u5185\u955C\u50CF\uFF0C\u6216\u9009\u300C\u81EA\u5B9A\u4E49\u300D\u586B\u4EFB\u610F\u955C\u50CF\uFF09",
  descAutoSend: "\u9759\u97F3\u5230\u70B9\u81EA\u52A8\u53D1\u9001\uFF08\u8FDE\u7EED\u591A\u6BB5\u62FC\u6210\u4E00\u6761\uFF1B\u5173=\u53EA\u8FDB\u8349\u7A3F\uFF1B\u8BF4\u8BDD\u8FC7\u7A0B\u4E2D\u6309\u4F4F Ctrl \u53EF\u7ACB\u5373\u53D1\u9001\uFF0Chold \u6A21\u5F0F\u677E\u624B\u5373\u53D1\u9001\uFF09",
  // 批 7N 重做 5/5：明确「开启后下次进入语音会话即生效」——消除「点完为何不立即恢复」的预期差。
  descAutoResume: "\u5207\u6362\u56DE\u4E0A\u6B21\u8BED\u97F3\u4F1A\u8BDD\u65F6\u81EA\u52A8\u6062\u590D\u8BED\u97F3\u6A21\u5F0F\uFF08\u9ED8\u8BA4\u5173\uFF1B\u5F00\u542F\u540E\u4E0B\u6B21\u8FDB\u5165\u8BED\u97F3\u4F1A\u8BDD\u5373\u751F\u6548\u2014\u2014\u81EA\u52A8\u8FDB\u5165\u8BED\u97F3\u6A21\u5F0F + \u6062\u590D\u4E0A\u6B21\u4F1A\u8BDD\uFF1B\u5173\u95ED\u5219\u9700\u624B\u52A8\u6309 Ctrl+Shift+V \u91CD\u65B0\u8FDB\u5165\uFF09",
  descSpokenFormat: "\u8BED\u97F3\u4F1A\u8BDD\u6CE8\u5165\u53E3\u8BED\u5316\u63D0\u793A\u8BCD\uFF08\u56DE\u590D\u53E3\u8BED\u5316\u3001\u77ED\u53E5\u3001\u4E0D\u7528 Markdown \u6392\u7248\u7B26\u53F7\uFF0C\u6717\u8BFB\u66F4\u987A\u66F4\u5FEB\uFF1B\u9ED8\u8BA4\u5F00\uFF0C\u6539\u52A8\u5373\u65F6\u751F\u6548\uFF09",
  descSenseVoice: "\u8BC6\u522B\u6A21\u5F0F\uFF1A\u5F00 = \u51C6\u786E\u6A21\u5F0F\uFF08SenseVoice \u5B9A\u7A3F\uFF0C\u5E26\u6807\u70B9+\u6570\u5B57\u5F52\u4E00\u5316\uFF0C\u957F\u53E5\u5728 CPU \u4E0A\u53EF\u80FD\u7B49\u6570\u79D2\uFF09\uFF1B\u5173 = \u5FEB\u901F\u6A21\u5F0F\uFF08\u6D41\u5F0F\u7ED3\u679C\u76F4\u63A5\u5B9A\u7A3F\uFF0C\u4E0D\u7B49 228MB \u91CD\u8BD1\u6A21\u578B\uFF0C\u51C6\u786E\u5EA6\u7565\u964D\uFF09",
  recognizingSec: "\u8BC6\u522B\u4E2D\u2026 {sec}s",
  descToolBeep: '\u5DE5\u5177\u8C03\u7528\u63D0\u793A\u97F3\uFF08\u9ED8\u8BA4\u5173\uFF09\uFF1AAI \u601D\u8003/\u8C03\u7528\u5DE5\u5177\u65F6"\u6EF4"\u4E00\u58F0\uFF1B\u5ACC\u5435\u5C31\u4FDD\u6301\u5173\u95ED',
  descSenseITN: "SenseVoice \u6570\u5B57/\u65E5\u671F/\u683C\u5F0F\u89C4\u8303\u5316\uFF08\u9ED8\u8BA4\u5F00\uFF1B\u5173\u95ED\u540E\u8F93\u51FA\u66F4\u63A5\u8FD1\u53E3\u8BED\u539F\u6587\uFF09",
  descCaptionFontSize: "\u6D6E\u5C42\u5B57\u5E55\u5B57\u53F7\u6863\u4F4D\uFF08\u5C0F 12 / \u6807\u51C6 14 / \u5927 18 / \u7279\u5927 24 px\uFF1B\u9ED8\u8BA4\u5C0F\u4E0E\u73B0\u72B6\u5B57\u8282\u7B49\u4EF7\uFF09",
  descCaptionMaxWidth: "\u6D6E\u5C42\u5B57\u5E55\u5BBD\u5EA6\u6863\u4F4D\uFF0850vw / 70vw / 90vw\uFF1B\u9ED8\u8BA4 70vw\uFF1B24px \u591A\u884C\u65F6\u53D7 30vh \u9AD8\u5EA6\u7EA6\u675F\u4E0D\u76D6\u8F93\u5165\u6846\uFF09",
  captionSizeS: "\u5C0F",
  captionSizeM: "\u6807\u51C6",
  captionSizeL: "\u5927",
  captionSizeXL: "\u7279\u5927",
  captionWidth50: "\u7A84",
  captionWidth70: "\u4E2D",
  captionWidth90: "\u5BBD",
  // 跳过按钮：只跳当前 TTS 句（非整段朗读）——UX 审查歧义修正。
  skipReading: "\u8DF3\u8FC7\u672C\u53E5",
  descBackchannelYield: "\u6717\u8BFB\u671F\u7528\u6237\u8BF4\u300C\u55EF/\u5BF9\u300D\u7B49\u77ED\u5E94\u7B54\u65F6\u81EA\u52A8\u8BA9\u4F4D\uFF08\u8DF3\u8FC7\u5F53\u524D TTS \u53E5 + \u77ED\u6682\u4E22\u5E27 1.5s\uFF1B\u771F\u8981\u8BF4\u5219\u539F hardBreak \u63A5\u7BA1\uFF1B\u5173 = \u4E0D\u8BA9\u4F4D\uFF0C\u884C\u4E3A\u7B49\u540C\u6539\u9020\u524D\uFF09\u3002\u4EC5\u300C\u6309\u4F4F\u8BF4\u8BDD\u300D\u6A21\u5F0F\u64AD\u653E\u671F\u53EF\u89E6\u53D1\uFF1A\u300C\u6301\u7EED\u8046\u542C\u300D\u6A21\u5F0F\u64AD\u653E\u671F\u4E0D\u8BC6\u522B\u8BED\u97F3\uFF0C\u5F00\u53E3\u8BF4\u8BDD\u6309\u6253\u65AD\u5904\u7406",
  // Issue #15 批 1 / ADR-0012：打断语音抢救。
  descBargeInSalvage: "AI \u6717\u8BFB\u65F6\u4F60\u5F00\u53E3\u6253\u65AD\uFF1A\u81EA\u52A8\u4FDD\u7559\u4ECE\u5F00\u53E3\u5230\u6253\u65AD\u751F\u6548\u4E4B\u95F4\u4F60\u8BF4\u7684\u8BDD\uFF0C\u5E76\u5165\u8BC6\u522B\u7EE7\u7EED\u53D1\u9001\uFF08\u4FEE\u300C\u6253\u65AD\u5403\u53E5\u300D\uFF1B\u5173 = \u4E22\u5F03\u6253\u65AD\u524D\u8BED\u97F3\u2014\u2014\u4EC5\u5916\u653E\u4E14\u65E0\u56DE\u58F0\u6D88\u9664\u7684\u73AF\u5883\u5EFA\u8BAE\u5173\uFF09",
  descLlmCorrect: "\u53D1\u9001\u524D\u7528\u5927\u6A21\u578B\u5BF9\u6574\u6761\u8BED\u97F3\u6587\u672C\u505A\u4E00\u6B21\u7EA0\u9519\uFF08\u53EA\u6539\u9519\u522B\u5B57/\u540C\u97F3\u5B57/\u6807\u70B9\uFF0C\u4FDD\u6301\u539F\u610F\uFF1B\u9700\u5BBF\u4E3B\u5DF2\u914D\u7F6E\u9ED8\u8BA4\u6A21\u578B\uFF0C\u672A\u914D\u7F6E\u5219\u81EA\u52A8\u7528\u539F\u6587\uFF09\u3002\u5F00 = \u6BCF\u6761\u6D88\u606F\u591A\u4E00\u6B21\u7EA0\u9519\u7B49\u5F85\uFF1A\u901A\u5E38 1 \u79D2\u5185\uFF0C\u6700\u957F\u7EA6 10 \u79D2\uFF1B\u6BCF\u5206\u949F\u6700\u591A 10 \u6B21\uFF0C\u8BED\u97F3\u6A21\u5F0F\u5173\u95ED\u6216\u89E6\u53D1\u9650\u6D41\u65F6\u6309\u539F\u6587\u53D1\u9001\u3002\u7EA0\u9519\u5931\u8D25\u6216\u8F93\u51FA\u5F02\u5E38\u540C\u6837\u81EA\u52A8\u7528\u539F\u6587",
  descYieldMs: "\u8BA9\u4F4D\u8BED\u4E49\u89E6\u53D1\u540E\u7B49\u5F85\u65F6\u957F\uFF08\u6BEB\u79D2\uFF0C500-3000\uFF0C\u9ED8\u8BA4 1500\uFF1B\u7A97\u53E3\u5185\u7528\u6237\u771F\u8981\u8BF4\u5219\u539F hardBreak \u63A5\u7BA1\uFF0C\u7A97\u53E3\u5230\u70B9\u81EA\u52A8\u6062\u590D\u64AD\u653E\uFF09\u3002\u4EC5\u300C\u6309\u4F4F\u8BF4\u8BDD\u300D\u6A21\u5F0F\u64AD\u653E\u671F\u53EF\u89E6\u53D1",
  descMode: "\u4EA4\u4E92\u6A21\u5F0F\uFF08toggle \u6301\u7EED\u8046\u542C+\u9759\u97F3\u65AD\u53E5 / hold \u6309\u4F4F\u8BF4\u8BDD / once \u8BF4\u5B8C\u4E00\u6BB5\u3001\u7B49\u56DE\u590D\u6717\u8BFB\u5B8C\u518D\u81EA\u52A8\u9000\u51FA\uFF09",
  modeToggle: "\u6301\u7EED\u8046\u542C",
  modeHold: "\u6309\u4F4F\u8BF4\u8BDD",
  modeOnce: "\u8BF4\u5B8C\u5C31\u5173",
  stopReading: "\u505C\u6B62\u6717\u8BFB",
  readAloudLabel: "\u6717\u8BFB\u56DE\u590D",
  readAloudVoice: "\u8BED\u97F3\u6A21\u5F0F",
  readAloudAlways: "\u603B\u662F",
  readAloudOff: "\u5173\u95ED",
  descReadAloud: "\u8BED\u97F3\u6A21\u5F0F=\u53EA\u5728\u8BED\u97F3\u6A21\u5F0F\u91CC\u6717\u8BFB\uFF08\u9ED8\u8BA4\uFF09\uFF1B\u603B\u662F=\u5F53\u524D\u4F1A\u8BDD\u7684\u6BCF\u6761\u56DE\u590D\u90FD\u6717\u8BFB\uFF08\u542B\u6253\u5B57\u53D1\u9001\u7684\uFF09\uFF1B\u5173\u95ED=\u4ECE\u4E0D\u6717\u8BFB",
  showCaptionsLabel: "\u5B57\u5E55\u6D6E\u5C42",
  descShowCaptions: "\u6717\u8BFB\u65F6\u5728\u53F3\u4E0B\u89D2\u663E\u793A\u5F53\u524D\u53E5\u5B50\u7684\u5B57\u5E55\u6D6E\u5C42",
  showStatusBarLabel: "\u72B6\u6001\u6761",
  descShowStatusBar: "\u5728\u8F93\u5165\u6846\u4E0A\u65B9\u663E\u793A\u8BED\u97F3\u72B6\u6001\u6761\uFF08\u6CE2\u5F62\u3001\u8BC6\u522B\u6587\u5B57\u3001\u8BA1\u65F6\u3001\u63D0\u793A\u3001\u9000\u51FA\uFF09\uFF1B\u5173\u95ED\u540E\u72B6\u6001\u53EA\u5728\u9EA6\u514B\u98CE\u6309\u94AE\u4E0A\u663E\u793A",
  descWakeWord: "\u5524\u9192\u8BCD\uFF08\u9ED8\u8BA4\u5173\uFF1B\u5982\u300C\u4F60\u597D\u5C0FD\u300D\uFF09\uFF1A\u8BF4\u51FA\u8BCD\u5934\u624D\u8FDB\u5165\u8BC6\u522B\uFF0C\u907F\u514D\u8BEF\u89E6\u3002\u53EF\u4E0E\u547D\u4EE4\u8FDE\u8BF4\uFF08\u300C\u4F60\u597D\u5C0FD\uFF0C\u5E2E\u6211\u67E5\u5929\u6C14\u300D\u2014\u2014\u8BCD\u5934\u4F1A\u81EA\u52A8\u5265\u6389\u3001\u4E0D\u8FDB\u6D88\u606F\uFF09\uFF1B\u6BCF\u53E5\u65AD\u53E5\u6216\u6253\u65AD\u540E\u9700\u91CD\u8BF4\uFF1B\u4EC5 toggle \u6A21\u5F0F\u751F\u6548\uFF08hold / \u624B\u52A8\u6253\u65AD\u4E0B\u4E0D\u751F\u6548\uFF1Bdetect \u81EA\u52A8\u63A2\u6D4B\u672A\u751F\u6548\u3001\u56DE\u843D manual \u65F6\u540C\u6837\u4E0D\u751F\u6548\u2014\u2014Safari \u7B49\u65E0\u539F\u751F\u56DE\u58F0\u6D88\u9664\u7684\u673A\u5668\u4F1A\u9ED8\u8BA4\u843D\u5230\u8FD9\u4E00\u6863\uFF09\uFF1B\u6717\u8BFB\u671F\u8BF4\u5524\u9192\u8BCD\u4E0D\u89E6\u53D1\uFF08\u6253\u65AD\u4ECD\u6309 VAD \u5F00\u53E3\u5373\u6253\u65AD\uFF09\u3002\u5339\u914D\u5E26\u5BB9\u9519\uFF08\u540C\u97F3\u5B57 / \u524D\u5BFC\u8BED\u6C14\u8BCD\uFF09\uFF0C\u5EFA\u8BAE 3-4 \u5B57\uFF1B\u975E\u4E13\u7528 KWS \u5F15\u64CE\uFF0C\u5608\u6742\u73AF\u5883\u53EF\u80FD\u5EF6\u8FDF\u6216\u8BEF\u6FC0\u6D3B",
  wakePlaceholder: "\u5982\uFF1A\u4F60\u597D\u5C0FD",
  settingsCardDesc: "\u6717\u8BFB\u5F15\u64CE / \u97F3\u8272 / \u8BED\u901F / \u6253\u65AD\u7075\u654F\u5EA6 / \u6253\u65AD\u65B9\u5F0F / \u56DE\u58F0\u95E8\u63A7 / \u9759\u97F3\u505C\u987F / \u7A7A\u95F2\u8D85\u65F6 / \u6A21\u578B\u955C\u50CF / \u81EA\u52A8\u53D1\u9001 / \u6717\u8BFB\u56DE\u590D / \u72B6\u6001\u6761 / \u5B57\u5E55\u6D6E\u5C42 / \u81EA\u52A8\u6062\u590D / \u4EA4\u4E92\u6A21\u5F0F / \u5524\u9192\u8BCD / \u5DE5\u5177\u63D0\u793A\u97F3 / \u9006\u6587\u672C\u5F52\u4E00\u5316 / \u5B57\u5E55\u5B57\u53F7 / \u5B57\u5E55\u5BBD\u5EA6 / \u77ED\u5E94\u7B54\u8BA9\u4F4D / \u8BA9\u4F4D\u7A97\u53E3 / \u6253\u65AD\u8BED\u97F3\u62A2\u6551 / LLM \u7EA0\u9519",
  settingsEffectiveNote: "\u751F\u6548\u65F6\u673A\u5206\u56DB\u6863\uFF1A\u2460 \u5373\u65F6\u2014\u2014\u6717\u8BFB\u5F15\u64CE / \u97F3\u8272 / \u8BED\u901F / Kokoro \u6A21\u578B\u7CBE\u5EA6\uFF08\u5DF2\u7F13\u5B58\u5219\u5373\u65F6\uFF0C\u5426\u5219\u4E0B\u6B21\u4E0B\u8F7D\uFF09/ \u53E3\u8BED\u5316\u63D0\u793A\u8BCD\uFF08\u5BF9\u540E\u7EED\u56DE\u590D\u751F\u6548\uFF09/ \u6717\u8BFB\u56DE\u590D / \u72B6\u6001\u6761 / \u5B57\u5E55\u6D6E\u5C42 / \u6A21\u578B\u955C\u50CF\uFF08\u4E0B\u6B21\u4E0B\u8F7D\u5C1D\u8BD5\u5373\u751F\u6548\uFF09\uFF1B\u2461 \u4E0B\u6B21\u8BC6\u522B\u65F6\u2014\u2014\u5B9A\u7A3F\u91CD\u8BD1\uFF08SenseVoice\uFF09/ \u9006\u6587\u672C\u5F52\u4E00\u5316\uFF1B\u2462 \u4E0B\u4E00\u6BB5\u8BDD\u2014\u2014\u9759\u97F3\u5224\u5B9A\u65F6\u957F\uFF08\u65AD\u53E5\u7AEF\u70B9\uFF1B\u53D1\u9001\u7B49\u5F85\u56FA\u5B9A 1.2 \u79D2\u4E0D\u53D7\u5F71\u54CD\uFF09\uFF1B\u2463 \u4E0B\u6B21\u8FDB\u5165\u8BED\u97F3\u6A21\u5F0F\u2014\u2014\u6253\u65AD\u7075\u654F\u5EA6 / \u6253\u65AD\u65B9\u5F0F / \u56DE\u58F0\u95E8\u63A7 / \u5FEB\u6377\u952E / \u81EA\u52A8\u53D1\u9001 / \u81EA\u52A8\u6062\u590D / \u4EA4\u4E92\u6A21\u5F0F / \u5524\u9192\u8BCD / \u5DE5\u5177\u63D0\u793A\u97F3 / \u5B57\u5E55\u5B57\u53F7 / \u5B57\u5E55\u5BBD\u5EA6 / \u77ED\u5E94\u7B54\u8BA9\u4F4D / \u8BA9\u4F4D\u7A97\u53E3 / \u6253\u65AD\u8BED\u97F3\u62A2\u6551 / LLM \u7EA0\u9519 / \u7A7A\u95F2\u65F6\u957F\u3002\u4E0B\u6B21\u8FDB\u5165\u8BED\u97F3\u6A21\u5F0F\u624D\u751F\u6548\u7684\u9879\u4F1A\u5217\u5728\u5361\u7247\u9876\u90E8\uFF08\u542B\u5F53\u524D\u751F\u6548\u503C\uFF09\u3002",
  effectivePending: "\u6709 {count} \u9879\u4FEE\u6539\u5C06\u5728\u4E0B\u6B21\u8FDB\u5165\u8BED\u97F3\u6A21\u5F0F\u65F6\u751F\u6548\uFF1A{items}",
  valueOn: "\u5F00",
  valueOff: "\u5173",
  valueQuoted: "\u300C{value}\u300D",
  effectiveItem: "{label}\uFF08\u5F53\u524D\u751F\u6548 {value}\uFF09",
  effectiveItemSep: "\u3001",
  configUnavailable: "\u914D\u7F6E\u6682\u4E0D\u53EF\u7528",
  // telemetry（P1-5 开发模式延迟埋点状态条：各段耗时标签）
  telUtteranceEnd: "\u8BF4\u5B8C",
  telEndpoint: "\u7AEF\u70B9",
  // 「提交」= 定稿上传发起；「定稿就绪」= 定稿文本可用。两者之间是宿主定稿耗时
  // （SenseVoice 整段重译），故不得都叫「定稿」——2026-11-03 补 final-ready 时改名。
  telSubmitted: "\u63D0\u4EA4",
  telFinalReady: "\u5B9A\u7A3F\u5C31\u7EEA",
  telFirstTokenLabel: "\u9996Token",
  telFirstSentence: "\u9996\u53E5",
  telFirstChunk: "\u9996chunk",
  telFirstPlayed: "\u9996\u97F3",
  // 模型管理（设置面板实时状态/重试）
  modelsTitle: "\u8BED\u97F3\u6A21\u578B",
  modelsDisabled: "\u5DF2\u5173\u95ED\uFF08\u8BBE\u7F6E\u4E2D\u5F00\u542F\uFF09",
  modelStreamingAsr: "\u6D41\u5F0F\u8BC6\u522B",
  modelVad: "\u7AEF\u70B9 VAD",
  modelSense: "\u5B9A\u7A3F\u91CD\u8BD1",
  modelsReady: "\u5C31\u7EEA",
  modelsDownloading: "{file} {percent}%",
  modelsFail: "\u4E0B\u8F7D\u5931\u8D25\uFF08{sec} \u79D2\u540E\u81EA\u52A8\u91CD\u8BD5\uFF09",
  modelsMissing: "\u672A\u4E0B\u8F7D",
  modelsRetry: "\u91CD\u8BD5\u4E0B\u8F7D",
  modelsRetrying: "\u91CD\u8BD5\u4E2D\u2026",
  modelsRetryHint: "\u955C\u50CF\u5207\u6362\u6216\u4E0B\u8F7D\u5931\u8D25\u540E\u70B9\u51FB\u7ACB\u5373\u91CD\u8BD5",
  modelsHint: "\u4E0B\u8F7D/\u8FDB\u5EA6\u5B9E\u65F6\u8DDF\u8FDB\uFF1B\u5931\u8D25\u81EA\u52A8\u9000\u907F 60s \u91CD\u8BD5\u3002\u955C\u50CF\u6E90\u5207\u6362\u540E\u70B9\u300C\u91CD\u8BD5\u4E0B\u8F7D\u300D\u7ACB\u5373\u751F\u6548\uFF1B\u4E5F\u53EF\u7528 npm run prefetch \u9884\u4E0B\u8F7D\u3002",
  // 朗读引擎状态（设置面板内联展示）
  engineLoading: "\u52A0\u8F7D\u4E2D\u2026",
  engineReady: "\u5C31\u7EEA",
  engineError: "\u52A0\u8F7D\u5931\u8D25",
  errKokoroNeedsNode: "\u672C\u5730 Kokoro \u65E0\u6CD5\u5728\u5B98\u65B9\u684C\u9762\u7AEF\uFF08Electron\uFF09\u5185\u76F4\u63A5\u8FD0\u884C\uFF1A\u8BF7\u5728\u7CFB\u7EDF PATH \u4E0A\u5B89\u88C5 Node.js \u226518\uFF08\u6216\u8BBE\u73AF\u5883\u53D8\u91CF DSHVM_NODE \u6307\u5411\u5B83\uFF09\uFF0C\u6216\u628A\u6717\u8BFB\u5F15\u64CE\u5207\u6362\u4E3A Edge / \u672C\u5730 VITS\u3002",
  ttsModelsMissing: "\u672C\u5730\u6A21\u578B\u672A\u5C31\u7EEA",
  ttsDownload: "\u4E0B\u8F7D",
  ttsDelete: "\u5220\u9664",
  ttsDownloading: "\u4E0B\u8F7D\u4E2D\u2026",
  ttsDeleting: "\u5220\u9664\u4E2D\u2026",
  ttsDownloadHint: "\u4E0B\u8F7D\u8BE5\u5F15\u64CE\u7684\u672C\u5730\u6A21\u578B\uFF1B\u4E0B\u8F7D\u5B8C\u6210\u540E\u7ACB\u5373\u5C31\u7EEA\uFF0C\u53EF\u8BD5\u542C/\u6717\u8BFB",
  ttsDeleteHint: "\u5220\u9664\u672C\u5730\u6A21\u578B\uFF08\u91CA\u653E\u7A7A\u95F4\uFF1B\u4E0B\u6B21\u4F7F\u7528\u4F1A\u81EA\u52A8\u91CD\u65B0\u4E0B\u8F7D\uFF09",
  dataFlowHint: "\u6570\u636E\u6D41\u5411\uFF1AASR \u8BC6\u522B\uFF08zipformer2+SenseVoice\uFF09\u59CB\u7EC8\u672C\u5730\uFF1BTTS \u6717\u8BFB\u53D6\u51B3\u4E8E\u5F15\u64CE\uFF08edge=\u5FAE\u8F6F\u4E91\u7AEF\uFF0Cvits/kokoro=\u672C\u5730\uFF09",
  kokoroModelInt8: "int8\uFF08\u9ED8\u8BA4\uFF09",
  kokoroModelFp32: "fp32\uFF08\u97F3\u8D28\u66F4\u597D\uFF09",
  descKokoroModel: "Kokoro \u6A21\u578B\u7CBE\u5EA6\uFF1Aint8 \u4F53\u79EF\u5C0F/\u52A0\u8F7D\u5FEB\uFF08\u7EAF CPU \u670D\u52A1\u5668\u6216\u4F4E\u5E26\u5BBD\u63A8\u8350\uFF0C\u9ED8\u8BA4\uFF09\uFF1Bfp32 \u97F3\u8D28\u66F4\u597D\u4F46\u7EA6 311MB\u3001\u66F4\u6162\uFF08\u6709\u72EC\u7ACB\u663E\u5361\u6216\u5927\u5185\u5B58\u673A\u5668\u63A8\u8350\uFF09\u3002\u4E24\u6863\u5171\u7528\u540C\u4E00\u5957 103 \u97F3\u8272\uFF0C\u5207\u6362\u5373\u65F6\u751F\u6548\u3002",
  // 设置分组标题
  secRead: "\u6717\u8BFB\u4E0E\u97F3\u8272",
  secInterrupt: "\u6253\u65AD\u4E0E\u9759\u97F3",
  secInteraction: "\u4EA4\u4E92\u4E0E\u8F93\u5165",
  secRecognition: "\u8BC6\u522B\u4E0E\u53E3\u8BED",
  secModel: "\u6A21\u578B\u4E0E\u955C\u50CF",
  telTotal: "\u5408\u8BA1",
  // 字段标题镜像键（*Label 后缀）：设置行标题经 settings-form 的 tr(`${name}Label`) 取值。
  // 批 C 曾与 settings-form 的 FIELD_LABELS 中文双轨并行；整合批已删 FIELD_LABELS，
  // *Label 键现在是行标题唯一真源（R1 M-2：孪生 base 键已删，防双轨漂移复发）。
  senseITNLabel: "\u9006\u6587\u672C\u5F52\u4E00\u5316",
  captionFontSizeLabel: "\u5B57\u5E55\u5B57\u53F7",
  captionMaxWidthLabel: "\u5B57\u5E55\u5BBD\u5EA6",
  backchannelYieldLabel: "\u77ED\u5E94\u7B54\u8BA9\u4F4D",
  // Issue #15 批 1 / ADR-0012：打断语音抢救（行标题键）。
  bargeInSalvageLabel: "\u6253\u65AD\u8BED\u97F3\u62A2\u6551",
  llmCorrectLabel: "LLM \u7EA0\u9519",
  // 让位窗口（行标题键）。
  yieldMsLabel: "\u8BA9\u4F4D\u7A97\u53E3",
  // 批 G 任务 2：空闲预警 + 退出提示文案。
  idleWarn30s: "30 \u79D2\u540E\u81EA\u52A8\u9000\u51FA\uFF08\u8BBE\u7F6E\u91CC\u53EF\u8C03\u7A7A\u95F2\u65F6\u957F\uFF09",
  idleTimeoutQuit: "\u7A7A\u95F2\u8D85\u65F6\u5DF2\u81EA\u52A8\u9000\u51FA\uFF08\u8BBE\u7F6E\u91CC\u53EF\u8C03\u65F6\u957F\uFF09",
  // 批 G 任务 1：Number 校验红框 + clamp 提示。
  numberInvalid: "\u6570\u503C\u975E\u6CD5",
  numberClamped: "\u5DF2\u81EA\u52A8\u8C03\u6574\u4E3A {value}",
  // 批 H 任务 4：autoResume 关 + 切回上次语音会话时的引导提示（5s 后自动清）。
  // 批 7N 重做 2/5：明确「自动恢复」=「自动进入语音模式 + 恢复上次会话」。
  autoResumeHint: "\u300C\u81EA\u52A8\u6062\u590D\u300D= \u5F00\u542F\u540E\u5207\u6362\u56DE\u4E0A\u6B21\u8BED\u97F3\u4F1A\u8BDD\u4F1A\u81EA\u52A8\u8FDB\u5165\u8BED\u97F3\u6A21\u5F0F\u5E76\u6062\u590D\u4E0A\u6B21\u4F1A\u8BDD\uFF1B\u5173\u95ED\u5219\u9700\u624B\u52A8\u6309 Ctrl+Shift+V",
  // —— 国际化补全（字段标题/音色描述词/错误码文案/试听分类）——
  ttsEngineLabel: "\u6717\u8BFB\u5F15\u64CE",
  kokoroModelLabel: "Kokoro \u6A21\u578B\u7CBE\u5EA6",
  voiceLabel: "\u97F3\u8272",
  rateLabel: "\u8BED\u901F",
  interruptLevelLabel: "\u6253\u65AD\u7075\u654F\u5EA6",
  bargeInModeLabel: "\u6253\u65AD\u65B9\u5F0F",
  echoGateDbLabel: "\u56DE\u58F0\u95E8\u63A7",
  modeLabel: "\u4EA4\u4E92\u6A21\u5F0F",
  shortcutLabel: "\u8FDB\u5165 / \u9000\u51FA\u8BED\u97F3\u6A21\u5F0F\uFF08\u5E94\u7528\u5185\uFF09",
  wakeWordLabel: "\u5524\u9192\u8BCD",
  toolBeepLabel: "\u5DE5\u5177\u63D0\u793A\u97F3",
  autoSendLabel: "\u81EA\u52A8\u53D1\u9001",
  autoResumeLabel: "\u81EA\u52A8\u6062\u590D",
  senseVoiceLabel: "\u5B9A\u7A3F\u91CD\u8BD1",
  spokenFormatLabel: "\u53E3\u8BED\u5316\u63D0\u793A\u8BCD",
  silenceMsLabel: "\u9759\u97F3\u505C\u987F",
  idleTimeoutMinutesLabel: "\u7A7A\u95F2\u8D85\u65F6",
  modelHostLabel: "\u6A21\u578B\u955C\u50CF",
  genderFemale: "\u5973",
  genderMale: "\u7537",
  genderNeutral: "\u4E2D\u6027",
  accentMandarin: "\u7B80\u4F53\u4E2D\u6587",
  accentNortheast: "\u4E1C\u5317\u8BDD",
  accentShaanxi: "\u9655\u897F\u8BDD",
  accentCantonese: "\u7CA4\u8BED",
  accentTaiwan: "\u53F0\u6E7E\u8154",
  accentEnglish: "English",
  styleDeep: "\u6DF1\u6C89",
  styleRich: "\u6D51\u539A",
  styleClear: "\u6E05\u4EAE",
  styleMagnetic: "\u78C1\u6027",
  voiceKokoroPopularMale: "\u5E38\u7528\u7537\u58F0",
  voiceKokoroChineseFemale: "\u4E2D\u6587\u5973",
  voiceKokoroMale: "\u7537\u58F0",
  voiceKokoroFemale: "\u5973\u58F0",
  voiceKokoroGeneric: "\u97F3\u8272",
  hostOfficial: "\u5B98\u65B9\u6E90",
  hostMirror: "\u56FD\u5185\u955C\u50CF",
  engineBadgeCloud: "\u8BC6\u522B\u672C\u5730 \xB7 \u6717\u8BFB\u4E91\u7AEF",
  engineBadgeLocal: "\u8BC6\u522B\u672C\u5730 \xB7 \u6717\u8BFB\u672C\u5730",
  errRateLimited: "\u8BF7\u6C42\u8FC7\u4E8E\u9891\u7E41\uFF0C\u8BF7\u7A0D\u540E\u518D\u8BD5",
  errUnknownSession: "\u8BED\u97F3\u4F1A\u8BDD\u5DF2\u5931\u6548\uFF0C\u8BF7\u91CD\u65B0\u5F00\u542F\u8BED\u97F3\u6A21\u5F0F",
  errForbidden: "\u8BF7\u6C42\u88AB\u62D2\u7EDD\uFF08\u4EC5\u9650\u672C\u673A\u8BBF\u95EE\uFF0C\u6216\u6765\u6E90\u4E0D\u7B26\uFF09",
  errBadRequest: "\u8BF7\u6C42\u65E0\u6548",
  errMethodNotAllowed: "\u8BF7\u6C42\u65B9\u6CD5\u4E0D\u88AB\u652F\u6301",
  errEngineNotActive: "\u8BE5\u6717\u8BFB\u5F15\u64CE\u5F53\u524D\u672A\u542F\u7528",
  errModelDownload: "\u6A21\u578B\u4E0B\u8F7D\u5931\u8D25\uFF1A\u8BF7\u68C0\u67E5\u7F51\u7EDC",
  errTooManyStreams: "\u8BED\u97F3\u8FDE\u63A5\u8FC7\u591A\uFF0C\u8BF7\u5173\u95ED\u5176\u5B83\u8BED\u97F3\u9875\u9762\u540E\u91CD\u8BD5",
  errTooLarge: "\u8BF7\u6C42\u5185\u5BB9\u8FC7\u5927",
  errInternal: "\u670D\u52A1\u51FA\u9519\uFF0C\u8BF7\u7A0D\u540E\u91CD\u8BD5",
  previewNetwork: "\u8BD5\u542C\u5931\u8D25\uFF1A\u7F51\u7EDC\u4E0D\u53EF\u8FBE\uFF08Edge \u4E91\u7AEF\u9700\u8BBF\u95EE\u5FAE\u8F6F\u8BED\u97F3\u670D\u52A1\uFF09\uFF0C\u8BF7\u68C0\u67E5\u7F51\u7EDC\u6216\u4EE3\u7406",
  previewEngine: "\u8BD5\u542C\u5931\u8D25\uFF1A\u5F15\u64CE\u672A\u5C31\u7EEA\uFF08\u672C\u5730\u6A21\u578B\u4E0B\u8F7D\u4E2D\u3001\u521D\u59CB\u5316\u5931\u8D25\u6216\u5B50\u8FDB\u7A0B\u5F02\u5E38\uFF09\uFF0C\u8BF7\u7A0D\u540E\u518D\u8BD5\u6216\u5728\u8BBE\u7F6E\u9762\u677F\u67E5\u770B TTS \u72B6\u6001",
  previewText: "\u8BD5\u542C\u5931\u8D25\uFF1A\u5408\u6210\u5F15\u64CE\u4EA7\u51FA\u7A7A\u97F3\u9891\uFF08\u97F3\u8272\u4E0E\u8BED\u79CD\u53EF\u80FD\u4E0D\u5339\u914D\uFF09\uFF0C\u8BF7\u66F4\u6362\u97F3\u8272\u6216\u68C0\u67E5\u8BED\u8A00\u8BBE\u7F6E",
  voiceHintChineseVoice: "\u63D0\u793A\uFF1A\u5F53\u524D\u662F\u4E2D\u6587\u97F3\u8272\uFF1B\u82E5\u4E3B\u8981\u7528\u82F1\u6587\u5BF9\u8BDD\uFF0C\u5EFA\u8BAE\u5728\u5217\u8868\u4E2D\u9009\u62E9\u82F1\u6587\u97F3\u8272\uFF08en-\u2026\uFF09\u3002",
  voiceHintEnglishVoice: "\u63D0\u793A\uFF1A\u5F53\u524D\u662F\u82F1\u6587\u97F3\u8272\uFF0C\u6717\u8BFB\u4E2D\u6587\u56DE\u590D\u6548\u679C\u4E0D\u4F73\uFF1B\u4E2D\u6587\u5BF9\u8BDD\u5EFA\u8BAE\u9009\u62E9\u4E2D\u6587\u97F3\u8272\uFF08zh-\u2026\uFF09\u3002"
};
var en = {
  stateVoiceMode: "Voice Mode",
  ttsNoticeFail: "Read-aloud connection failed: retrying\u2026",
  ttsSkipNotice: "One sentence failed to read and was skipped (resend the message to retry)",
  ttsUnavailableNotice: "Read-aloud keeps failing: cloud TTS is unavailable. Try again later, or switch the read-aloud engine to local VITS / Kokoro in settings",
  enterFail: "Failed to enter voice mode",
  disabled: "Voice mode disabled (plugin enabled=false)",
  sendFailKept: "Send failed; text kept in draft",
  micDenied: "Microphone denied: allow mic access for this site",
  micBusy: "Microphone is in use by another program \u2014 close it and retry",
  micUnavailable: "Microphone unavailable",
  recognizing: "Recognizing\u2026",
  // UX-I1 联动态：纠错在途 / 纠错放弃回退 / 附和词忽略 / 语音按钮态文案。
  correctFixing: "Correcting\u2026",
  correctAbandoned: "Original text kept in draft",
  backchannelDropped: "Backchannel ignored, not sent",
  buttonLoading: "Loading\u2026",
  buttonListening: "Listening",
  holdToTalk: "Hold to talk",
  releaseToSend: "Release to send",
  entering: "Entering\u2026",
  voiceBtn: "Voice",
  ariaActive: "Voice mode active",
  ariaEnter: "Enter voice mode",
  titleHold: "Voice mode \xB7 hold to talk, release to send; tap to exit; Esc/blur cancels; Ctrl+Shift+V exits",
  titleToggle: "Voice mode \xB7 click to exit (Ctrl+Shift+V) \xB7 hold Ctrl while speaking to send now",
  titleEnter: "Enter voice mode (Ctrl+Shift+V)",
  loadingModel: "Loading model\u2026",
  listening: "Listening\u2026",
  thinking: "Thinking\u2026",
  barHold: "Voice mode \xB7 hold to talk (tap to exit)",
  barListening: "Voice mode \xB7 listening\u2026",
  wakeWord: "Wake word",
  sayWake: 'Say "{wake}" to start',
  reading: "Reading aloud\u2026",
  recognitionFail: "Recognition failed, try again",
  partialFail: "Live captions temporarily unavailable (recognition channel error, retrying)",
  detectFail: "Break-in detection temporarily unavailable (detect channel error, retrying)",
  sseReconnect: "State sync disconnected, reconnecting\u2026",
  diagCopy: "Copy diagnostics",
  diagCopied: "Diagnostics copied",
  diagCopyFailed: "Copy failed (clipboard rejected by browser)",
  sessionExpired: "Voice session expired, reconnecting\u2026",
  sessionExpiredFail: "Voice session reconnect failed; please re-enter voice mode",
  modelDownloadFail: "Model download failed ({file}): check network and re-enter voice mode",
  startFail: "Voice mode failed to start: {err}",
  holdDots: "Hold to talk\u2026",
  exit: "Exit",
  configUnavailableNote: " (settings document not ready; retrying \u2014 the panel will appear when it is).",
  previewNameFirst: "Enter a voice ShortName first",
  previewDisabled: "Voice mode disabled (plugin enabled=false); preview unavailable",
  // 批 G 任务 6：本地 TTS 引擎未下载模型时禁用试听按钮 + 提示。
  previewModelMissing: "Download the local model first (see the engine status above)",
  previewModelLoading: "Local model is downloading \u2014 please wait",
  previewPlayFail: "Preview failed: cannot play this voice",
  previewAutoplay: "Autoplay blocked \u2014 click preview again",
  previewCheck: "Preview failed: check network or ShortName",
  previewRateLimited: "Preview too frequent \u2014 wait a few seconds (20/min limit)",
  previewTimeout: "Synthesis timed out: model still loading, retry in a few seconds",
  previewBtnTitle: "Preview voice (current rate)",
  synthesizing: "Synthesizing\u2026",
  preview: "Preview",
  custom: "Custom",
  voicePrev: "Previous voice",
  voiceNext: "Next voice",
  descVoice: "Edge cloud voices (auto-loads all Microsoft voices; common Chinese voices pinned on top; \u25C0\u25B6 or dropdown; custom ShortName allowed)",
  descVoiceLocal: "Local voice (vits, all 5 speakers listed; dropdown or \u25C0\u25B6; no custom needed)",
  descTtsEngine: "Local VITS (Chinese only) / Local Kokoro (Chinese + English) / Edge cloud (most natural, text sent to Microsoft)",
  engineVits: "Local VITS",
  engineKokoro: "Local Kokoro",
  engineEdge: "Edge cloud",
  descVoiceKokoro: "Kokoro zh-en voices (103; \u25C0\u25B6 to cycle; 48-51 named Chinese, others numbered with measured gender; mixed zh-en supported)",
  descRate: "Speech rate (0.5 slow \u2013 2.0 fast, 1.1 default; more compact replies)",
  descInterrupt: "Interrupt sensitivity (0 high \u22480.3 s confirm / 1 medium \u22480.2 s / 2 low \u22480.1 s, most responsive); lower = higher barrier = harder to interrupt",
  descBargeIn: "Barge-in mode (detect: auto-probe native echo cancellation, fall back to hold-to-talk when inactive \u2014 default; auto: force interrupt by speaking \u2014 headphones/quiet; manual: for loudspeaker, no echo-triggered self-interrupt \u2014 hold mic/Ctrl to interrupt). Note: in manual mode (including detect falling back to manual on machines without native echo cancellation) audio outside a press is not recognized \u2014 in toggle mode you can long-press to speak only while the agent is playing; when idle there is no voice input, so use hold mode or switch to auto/detect",
  bargeInDetect: "Auto-detect",
  bargeInAuto: "Auto",
  bargeInManual: "Manual",
  descEchoGate: "Echo gate threshold (dB, default 6): auto barge-in requires the residual to exceed the echo floor by this value. With the native browser AEC active, this gate is idle; it kicks in as a fallback in Safari or environments without native AEC (e.g. some headphones). Raise (8-10) if speaker echo still interrupts, lower (3-4) if hard to interrupt",
  descShortcut: "Shortcut to enter/exit voice mode (e.g. Ctrl+Shift+V; empty disables it, mic button only; avoid browser-reserved combos like Ctrl+W/N/T)",
  vadDetected: "VAD speech",
  aecOff: "Native AEC off",
  aecOffHint: "Native echo cancellation is not active (speaker echo may self-interrupt); use headphones or Manual barge-in",
  // 批 7O（ADR-0006）：detect 模式第一级探测降级提示（英文）。
  bargeInDetectFallback: "Native echo cancellation is off \u2014 switched to hold-to-talk barge-in",
  // R1 收尾（fix-client 直连 tk() 调用）：连续 3 次回声拒断的解释出口 + 快捷键非法停用提示。
  echoRejectHint: "Strong echo detected \u2014 barge-in is limited by echo protection. Use headphones or switch to manual barge-in",
  shortcutInvalid: "Invalid shortcut, disabled. Fix it in settings (e.g. Ctrl+Shift+V); leave empty to use the mic button only",
  descComposerShortcut: "Shortcut to toggle voice mode while the composer has focus (default Alt/Option+Enter)",
  composerShortcutLabel: "Enter / exit voice mode (in composer)",
  shortcutRecord: "Record",
  shortcutRecording: "Press a key combo\u2026",
  shortcutCancel: "Cancel",
  shortcutNeedModifier: "At least one modifier is required",
  shortcutUnsupportedKey: "Only letters, digits, Enter, Space and Tab are supported",
  shortcutPlaceholder: "Click Record, then press a combo",
  shortcutReset: "Reset to default",
  shortcutNote: "Alt/Option+Enter toggles inside the composer; Cmd / Ctrl / Alt / Shift combos are supported; leaving it empty means mic button only. While recording, Backspace clears and Esc cancels; avoid browser-reserved combos (Cmd+W / T / N / L). Changes apply immediately.",
  secShortcut: "Shortcut",
  elapsedHint: "Current voice session duration",
  botLevelsHint: "AI playback level (blue bar = TTS playback, green bar = microphone)",
  interruptConfirm: "interrupt confirm",
  sev0: "0 high",
  sev1: "1 medium",
  sev2: "2 low",
  // 数值单位（设置卡片「当前生效值」用：中文汉字单位、英文国际缩写，避免「1500 ms」直出）
  unitMs: "ms",
  unitMin: "min",
  unitDb: "dB",
  descSilence: "Silence pause before a sentence is committed (default 1500 ms; at least 250 ms of speech required, guards against noise triggers)",
  descIdle: "Auto-exit voice mode after idle minutes (default 5; a 30-second countdown warning appears before exit)",
  descModelHost: "ASR model download source (official source / mirror, or any custom URL)",
  descAutoSend: "Auto-send once quiet (consecutive segments join into one message; off = draft only; holding Ctrl while speaking sends immediately; in hold mode, releasing sends)",
  // 批 7N 重做 5/5：clarify that toggling takes effect on next session entry (not the current one).
  descAutoResume: "Auto-resume voice mode when switching back to the last voice session (default off; takes effect on next session entry \u2014 auto-enters voice mode and restores the last session; when disabled, you must press Ctrl+Shift+V to re-enter)",
  descSpokenFormat: "Inject spoken-format prompt into voice replies (colloquial, short sentences, no Markdown; default on, live)",
  descSenseVoice: "Recognition mode: on = accurate mode (SenseVoice finalization with punctuation + ITN; long utterances may take seconds on CPU); off = fast mode (streaming result finalized directly, skips the 228 MB re-transcription model, slightly less accurate)",
  recognizingSec: "Recognizing\u2026 {sec}s",
  descToolBeep: "Tool-call beep (default off): beep when the agent is thinking/calling tools; keep off if it annoys you",
  descSenseITN: "SenseVoice number/date/format normalization (default on; turn off to keep raw spoken form)",
  descCaptionFontSize: "Caption font size (Small 12 / Standard 14 / Large 18 / X-Large 24 px; default Small matches current behavior)",
  descCaptionMaxWidth: "Caption width (50vw / 70vw / 90vw; default 70vw; capped at 30vh height so multi-line 24px never overlaps the input box)",
  captionSizeS: "S",
  captionSizeM: "M",
  captionSizeL: "L",
  captionSizeXL: "XL",
  captionWidth50: "Narrow",
  captionWidth70: "Medium",
  captionWidth90: "Wide",
  skipReading: "Skip this sentence",
  descBackchannelYield: 'When the user says a short answer like "mm-hmm/right" while the agent is reading aloud, yield automatically (skip the current TTS sentence + drop frames for 1.5s; if the user really wants to speak, the existing hardBreak takes over; off = no yielding, behavior matches before this change). Only triggers in hold-to-talk mode: in continuous-listening mode speech is not recognized during playback (speaking is treated as an interrupt)',
  // Issue #15 batch 1 / ADR-0012: barge-in speech salvage.
  descBargeInSalvage: "When you speak to interrupt while the agent is reading aloud, keep what you said from speech onset to break confirmation and merge it into recognition (fixes swallowed break-in speech; off = discard pre-break audio; recommended off only on speakers without echo cancellation)",
  descLlmCorrect: "Before sending, run one LLM pass over the whole transcribed text (typos/homophones/punctuation only, meaning preserved; requires a default model configured in the host \u2014 otherwise the original text is sent as-is). On = one extra correction round trip per message: usually under 1 second, at most ~10 seconds; at most 10 times per minute \u2014 when voice mode is off or the limit is hit, the original text is sent. Failed or abnormal output also falls back to the original text",
  descYieldMs: "How long (ms) to drop frames after a yield trigger (500-3000, default 1500; within the window the existing hardBreak takes over if the user really wants to speak; auto-resume after the window expires). Hold-to-talk mode only, during playback",
  descMode: "Interaction mode (toggle: continuous listen + auto-send / hold: press to talk / once: one utterance, then exit after the reply is read aloud)",
  modeToggle: "Continuous listening",
  modeHold: "Hold to talk",
  modeOnce: "One-shot",
  stopReading: "Stop reading",
  readAloudLabel: "Read replies",
  readAloudVoice: "Voice mode",
  readAloudAlways: "Always",
  readAloudOff: "Off",
  descReadAloud: "Voice mode = read replies only in voice mode (default); Always = read every reply in the current session, typed ones included; Off = never read aloud",
  showCaptionsLabel: "Caption overlay",
  descShowCaptions: "Show a floating caption of the current sentence while reading aloud",
  showStatusBarLabel: "Status bar",
  descShowStatusBar: "Show the voice status bar above the composer (waveform, transcript, timer, notices, exit); when off, state shows only on the mic button",
  descWakeWord: 'Wake word (default off; e.g. "Hey D"): recognition starts only after you say it, to avoid accidental triggers. You may say it together with your command ("Hey D, check the weather" \u2014 the wake word is stripped and never sent); it must be repeated after each utterance split or barge-in; toggle mode only (inactive in hold / manual barge-in \u2014 and equally inactive when the detect probe fails and falls back to manual, the default on machines without native echo cancellation such as Safari); saying it while the agent is reading does not trigger (barge-in stays VAD-based). Fault-tolerant matching (homophones / leading fillers); 3-4 characters recommended; not a dedicated KWS engine \u2014 noisy environments may delay or falsely trigger',
  wakePlaceholder: "e.g. Hey D",
  settingsCardDesc: "Engine / voice / rate / interrupt / barge-in / echo gate / silence / idle / model host / auto-send / read replies / status bar / caption overlay / auto-resume / mode / wake word / tool beep / ITN / caption font / caption width / yielding / yield window / barge-in salvage / LLM correction",
  settingsEffectiveNote: "When settings apply, in four tiers: \u2460 immediately \u2014 engine / voice / rate / Kokoro model precision (immediate when cached, else next download) / spoken format (applies to later replies) / read replies / status bar / caption overlay / model mirror (next download attempt); \u2461 next recognition \u2014 finalization re-transcription (SenseVoice) / ITN; \u2462 next utterance \u2014 silence threshold (endpoint; the send wait stays a fixed 1.2s); \u2463 next time you enter voice mode \u2014 interrupt sensitivity / barge-in mode / echo gate / shortcut / auto-send / auto-resume / interaction mode / wake word / tool beep / caption font / caption width / backchannel yielding / yield window / barge-in salvage / LLM correction / idle timeout. Items that take effect next time you enter voice mode are listed at the top of this card with their current effective values.",
  effectivePending: "{count} change(s) will apply next time you enter voice mode: {items}",
  valueOn: "On",
  valueOff: "Off",
  valueQuoted: '"{value}"',
  effectiveItem: "{label} (currently effective: {value})",
  effectiveItemSep: ", ",
  configUnavailable: "Configuration unavailable",
  telUtteranceEnd: "end",
  telEndpoint: "endpoint",
  telSubmitted: "submit",
  telFinalReady: "final ready",
  telFirstTokenLabel: "1st token",
  telFirstSentence: "1st sentence",
  telFirstChunk: "1st chunk",
  telFirstPlayed: "1st audio",
  modelsTitle: "Voice models",
  modelsDisabled: "off (enable in settings)",
  modelStreamingAsr: "Streaming ASR",
  modelVad: "Endpoint VAD",
  modelSense: "Finalize",
  modelsReady: "Ready",
  modelsDownloading: "{file} {percent}%",
  modelsFail: "Download failed (auto-retry in {sec}s)",
  modelsMissing: "not downloaded",
  modelsRetry: "Retry",
  modelsRetrying: "Retrying\u2026",
  modelsRetryHint: "Click to retry now after switching mirror or a failure",
  modelsHint: "Live download state; failures auto-backoff 60s. After switching the mirror, click Retry to take effect immediately; npm run prefetch pre-downloads.",
  engineLoading: "loading\u2026",
  engineReady: "ready",
  engineError: "failed",
  errKokoroNeedsNode: "Local Kokoro cannot run directly inside the official desktop app (Electron): install Node.js >= 18 on PATH (or set DSHVM_NODE to its path), or switch the read-aloud engine to Edge / local VITS.",
  ttsModelsMissing: "local models missing",
  ttsDownload: "Download",
  ttsDelete: "Delete",
  ttsDownloading: "Downloading\u2026",
  ttsDeleting: "Deleting\u2026",
  ttsDownloadHint: "Download this engine's local model; becomes ready immediately after",
  dataFlowHint: "Data flow: ASR recognition (zipformer2+SenseVoice) is always local; TTS playback depends on engine (edge=Microsoft cloud, vits/kokoro=local)",
  ttsDeleteHint: "Delete local models (frees space; auto re-downloads on next use)",
  kokoroModelInt8: "int8 (default)",
  kokoroModelFp32: "fp32 (better quality)",
  descKokoroModel: "Kokoro model precision: int8 is smaller/faster (CPU server or low bandwidth; default); fp32 sounds better but ~311MB and slower (GPU or large memory). Both share the same 103 voices; switches live.",
  secRead: "Reading & voice",
  secInterrupt: "Interrupt & silence",
  secInteraction: "Interaction",
  secRecognition: "Recognition & speech",
  secModel: "Model & mirror",
  telTotal: "total",
  // 字段标题（Row 标题）：settings-form 经 tr(`${name}Label`) 取值，zh/en 均为正式文案。
  senseITNLabel: "Inverse text normalization",
  captionFontSizeLabel: "Caption size",
  captionMaxWidthLabel: "Caption width",
  backchannelYieldLabel: "Short-reply yielding",
  // Issue #15 batch 1 / ADR-0012: barge-in speech salvage (row title key).
  bargeInSalvageLabel: "Barge-in speech salvage",
  llmCorrectLabel: "LLM correction",
  // Yield window (row title key).
  yieldMsLabel: "Yield window",
  // 批 G 任务 2：空闲预警 + 退出提示文案。
  idleWarn30s: "Auto-exit in 30 seconds (adjustable in settings)",
  idleTimeoutQuit: "Idle timeout \u2014 voice mode auto-exited (adjustable in settings)",
  // 批 G 任务 1：Number 校验红框 + clamp 提示。
  numberInvalid: "Invalid number",
  numberClamped: "Auto-clamped to {value}",
  // 批 H 任务 4：autoResume 关 + 切回上次语音会话时的引导提示（5s 后自动清）。
  // 批 7N 重做 2/5：明确「自动恢复」=「自动进入语音模式 + 恢复上次会话」。
  autoResumeHint: '"Auto-resume" = when enabled, switching back to your last voice session auto-enters voice mode and restores the session; when disabled, press Ctrl+Shift+V to re-enter',
  // —— i18n completion (field labels / voice descriptors / error-code copy / preview categories) ——
  ttsEngineLabel: "Read-aloud engine",
  kokoroModelLabel: "Kokoro model precision",
  voiceLabel: "Voice",
  rateLabel: "Speed",
  interruptLevelLabel: "Interrupt sensitivity",
  bargeInModeLabel: "Interrupt mode",
  echoGateDbLabel: "Echo gate",
  modeLabel: "Interaction mode",
  shortcutLabel: "Enter / exit voice mode (app-wide)",
  wakeWordLabel: "Wake word",
  toolBeepLabel: "Tool-call beep",
  autoSendLabel: "Auto-send",
  autoResumeLabel: "Auto-resume",
  senseVoiceLabel: "Re-transcribe on finalize",
  spokenFormatLabel: "Spoken-style prompt",
  silenceMsLabel: "Silence pause",
  idleTimeoutMinutesLabel: "Idle timeout",
  modelHostLabel: "Model mirror",
  genderFemale: "Female",
  genderMale: "Male",
  genderNeutral: "Neutral",
  accentMandarin: "Mandarin",
  accentNortheast: "Northeastern",
  accentShaanxi: "Shaanxi",
  accentCantonese: "Cantonese",
  accentTaiwan: "Taiwanese Mandarin",
  accentEnglish: "English",
  styleDeep: "Deep",
  styleRich: "Rich",
  styleClear: "Clear",
  styleMagnetic: "Magnetic",
  voiceKokoroPopularMale: "Popular male",
  voiceKokoroChineseFemale: "Chinese female",
  voiceKokoroMale: "Male",
  voiceKokoroFemale: "Female",
  voiceKokoroGeneric: "Voice",
  hostOfficial: "Official",
  hostMirror: "Mirror (CN)",
  engineBadgeCloud: "ASR local \xB7 TTS cloud",
  engineBadgeLocal: "ASR local \xB7 TTS local",
  errRateLimited: "Too many requests \u2014 try again shortly",
  errUnknownSession: "Voice session expired \u2014 re-enter voice mode",
  errForbidden: "Request denied (local access only, or origin mismatch)",
  errBadRequest: "Invalid request",
  errMethodNotAllowed: "Request method not allowed",
  errEngineNotActive: "That read-aloud engine is not active",
  errModelDownload: "Model download failed \u2014 check your network",
  errTooManyStreams: "Too many voice connections \u2014 close other voice tabs and retry",
  errTooLarge: "Request too large",
  errInternal: "Server error \u2014 please retry shortly",
  previewNetwork: "Preview failed: network unreachable (Edge cloud needs the Microsoft speech service) \u2014 check your network or proxy",
  previewEngine: "Preview failed: engine not ready (local model downloading, init failed, or worker crashed) \u2014 retry later or check TTS status in settings",
  previewText: "Preview failed: the engine produced empty audio (voice and language may not match) \u2014 pick another voice",
  voiceHintChineseVoice: "Note: this is a Chinese voice. For English replies, pick an English voice (en-\u2026) from the list.",
  voiceHintEnglishVoice: "Note: this is an English voice and reads Chinese replies poorly. For Chinese conversations, pick a Chinese voice (zh-\u2026)."
};
var RU = {
  // --- чтение и озвучка ---
  secRead: "\u041E\u0437\u0432\u0443\u0447\u043A\u0430",
  ttsEngineLabel: "\u0414\u0432\u0438\u0436\u043E\u043A \u043E\u0437\u0432\u0443\u0447\u043A\u0438",
  descTtsEngine: "\u041B\u043E\u043A\u0430\u043B\u044C\u043D\u044B\u0439 Kokoro (\u0440\u0443\u0441\u0441\u043A\u0438\u0439, \u0433\u043E\u043B\u043E\u0441\u0430 \u041C\u0430\u0448\u0430 \u0438 \u0421\u0432\u0435\u0442\u0430) / \u043B\u043E\u043A\u0430\u043B\u044C\u043D\u044B\u0439 VITS (\u043A\u0438\u0442\u0430\u0439\u0441\u043A\u0438\u0439) / Edge \u2014 \u043E\u0431\u043B\u0430\u043A\u043E Microsoft, \u0442\u0435\u043A\u0441\u0442 \u0443\u0445\u043E\u0434\u0438\u0442 \u043D\u0430\u0440\u0443\u0436\u0443",
  voiceLabel: "\u0413\u043E\u043B\u043E\u0441",
  descVoiceKokoro: "\u0420\u0443\u0441\u0441\u043A\u0438\u0435 \u0433\u043E\u043B\u043E\u0441\u0430: \u041C\u0430\u0448\u0430 (\u043F\u043E \u0443\u043C\u043E\u043B\u0447\u0430\u043D\u0438\u044E) \u0438 \u0421\u0432\u0435\u0442\u0430. \u041F\u0435\u0440\u0435\u043A\u043B\u044E\u0447\u0435\u043D\u0438\u0435 \u0441\u0442\u0440\u0435\u043B\u043A\u0430\u043C\u0438 \u0438\u043B\u0438 \u0438\u0437 \u0441\u043F\u0438\u0441\u043A\u0430",
  descVoiceLocal: "\u0413\u043E\u043B\u043E\u0441\u0430 \u043B\u043E\u043A\u0430\u043B\u044C\u043D\u043E\u0433\u043E \u0434\u0432\u0438\u0436\u043A\u0430 (\u0432 \u043A\u0438\u0442\u0430\u0439\u0441\u043A\u043E\u0439 \u043C\u043E\u0434\u0435\u043B\u0438 VITS \u2014 \u043F\u044F\u0442\u044C \u0434\u0438\u043A\u0442\u043E\u0440\u043E\u0432)",
  rateLabel: "\u0421\u043A\u043E\u0440\u043E\u0441\u0442\u044C",
  descRate: "\u041C\u043D\u043E\u0436\u0438\u0442\u0435\u043B\u044C \u0441\u043A\u043E\u0440\u043E\u0441\u0442\u0438 \u0440\u0435\u0447\u0438: 0.5 \u043C\u0435\u0434\u043B\u0435\u043D\u043D\u043E, 2.0 \u0431\u044B\u0441\u0442\u0440\u043E, \u043F\u043E \u0443\u043C\u043E\u043B\u0447\u0430\u043D\u0438\u044E 1.0",
  readAloudLabel: "\u0427\u0438\u0442\u0430\u0442\u044C \u043E\u0442\u0432\u0435\u0442\u044B",
  descReadAloud: "\u0412 \u0433\u043E\u043B\u043E\u0441\u043E\u0432\u043E\u043C \u0440\u0435\u0436\u0438\u043C\u0435 \u2014 \u0447\u0438\u0442\u0430\u0442\u044C \u0442\u043E\u043B\u044C\u043A\u043E \u0442\u0430\u043C; \u0412\u0441\u0435\u0433\u0434\u0430 \u2014 \u0447\u0438\u0442\u0430\u0442\u044C \u043B\u044E\u0431\u043E\u0439 \u043E\u0442\u0432\u0435\u0442 \u0432 \u0442\u0435\u043A\u0443\u0449\u0435\u0439 \u0441\u0435\u0441\u0441\u0438\u0438; \u0412\u044B\u043A\u043B\u044E\u0447\u0435\u043D\u043E \u2014 \u043D\u0435 \u0447\u0438\u0442\u0430\u0442\u044C \u043D\u0438\u0447\u0435\u0433\u043E (\u043E\u0437\u0432\u0443\u0447\u043A\u0430 \u043F\u043E \u043A\u043E\u043C\u0430\u043D\u0434\u0435 \u0447\u0435\u0440\u0435\u0437 \u0438\u043D\u0441\u0442\u0440\u0443\u043C\u0435\u043D\u0442 say)",
  readAloudVoice: "\u0412 \u0433\u043E\u043B\u043E\u0441\u043E\u0432\u043E\u043C \u0440\u0435\u0436\u0438\u043C\u0435",
  readAloudAlways: "\u0412\u0441\u0435\u0433\u0434\u0430",
  readAloudOff: "\u0412\u044B\u043A\u043B\u044E\u0447\u0435\u043D\u043E",
  showCaptionsLabel: "\u0421\u0443\u0431\u0442\u0438\u0442\u0440\u044B",
  descShowCaptions: "\u041F\u043E\u043A\u0430\u0437\u044B\u0432\u0430\u0442\u044C \u0441\u0443\u0431\u0442\u0438\u0442\u0440\u044B \u043F\u0440\u043E\u0438\u0437\u043D\u043E\u0441\u0438\u043C\u043E\u0433\u043E \u0442\u0435\u043A\u0441\u0442\u0430 \u043F\u043E\u0432\u0435\u0440\u0445 \u0441\u0442\u0440\u0430\u043D\u0438\u0446\u044B",
  showStatusBarLabel: "\u0421\u0442\u0440\u043E\u043A\u0430 \u0441\u043E\u0441\u0442\u043E\u044F\u043D\u0438\u044F",
  descShowStatusBar: "\u041F\u043E\u043A\u0430\u0437\u044B\u0432\u0430\u0442\u044C \u0441\u043E\u0441\u0442\u043E\u044F\u043D\u0438\u0435 \u043C\u0438\u043A\u0440\u043E\u0444\u043E\u043D\u0430 \u0438 \u043E\u0437\u0432\u0443\u0447\u043A\u0438",
  engineKokoro: "\u041B\u043E\u043A\u0430\u043B\u044C\u043D\u044B\u0439 Kokoro",
  engineVits: "\u041B\u043E\u043A\u0430\u043B\u044C\u043D\u044B\u0439 VITS",
  engineEdge: "Edge (\u043E\u0431\u043B\u0430\u043A\u043E)",
  preview: "\u041F\u0440\u043E\u0441\u043B\u0443\u0448\u0430\u0442\u044C",
  valueOn: "\u0432\u043A\u043B",
  valueOff: "\u0432\u044B\u043A\u043B"
};
for (const [key, value] of Object.entries(RU)) {
  if (value === void 0) continue;
  zh[key] = value;
  en[key] = value;
}

// src/llm-correct.ts
function correctionSystem(lang) {
  return lang === "en" ? "You are a speech-recognition corrector. The user text comes from ASR and may contain homophone errors, typos, or missing/wrong punctuation. Make only minimal fixes: correct typos/homophones and fix punctuation, preserving the original meaning and colloquial style. Do not rewrite sentences, do not add content, do not answer questions, and do not alter numbers or proper nouns. Output only the corrected text, with no explanation. If the input has no obvious errors, return it unchanged." : "\u4F60\u662F\u4E2D\u6587\u8BED\u97F3\u8BC6\u522B\u7EA0\u9519\u5668\u3002\u7528\u6237\u7684\u8BDD\u6765\u81EA\u8BED\u97F3\u8BC6\u522B\uFF08ASR\uFF09\uFF0C\u53EF\u80FD\u6709\u540C\u97F3\u5B57\u3001\u9519\u522B\u5B57\u3001\u6807\u70B9\u7F3A\u5931\u6216\u9519\u8BEF\u3002\u8BF7\u53EA\u505A\u6700\u5C0F\u4FEE\u6B63\uFF1A\u6539\u9519\u522B\u5B57/\u540C\u97F3\u5B57\u3001\u8865\u6807\u70B9\uFF0C\u4FDD\u6301\u539F\u610F\u4E0E\u53E3\u8BED\u98CE\u683C\uFF1B\u4E0D\u8981\u6539\u5199\u53E5\u5B50\u3001\u4E0D\u8981\u8865\u5145\u5185\u5BB9\u3001\u4E0D\u8981\u56DE\u7B54\u95EE\u9898\u3001\u4E0D\u8981\u6539\u53D8\u6570\u5B57\u548C\u4E13\u4E1A\u672F\u8BED\u3002\u53EA\u8F93\u51FA\u4FEE\u6B63\u540E\u7684\u6587\u672C\uFF0C\u4E0D\u8981\u4EFB\u4F55\u89E3\u91CA\u3002\u5982\u679C\u539F\u6587\u6CA1\u6709\u660E\u663E\u9519\u8BEF\uFF0C\u539F\u6837\u8FD4\u56DE\u3002";
}
var CORRECTION_PREAMBLE_RE = /^(?:以下是|这是|已|文本)?(?:修正|纠正|更正|纠错)|^(?:here(?:'s| is) (?:the |an |a )?|i have |text after )?(?:corrected|correction)\b/i;
var CORRECTION_MIN_SIMILARITY = 0.45;
var SIMILARITY_MAX_CHARS = 1500;
function normalizedSimilarity(a, b) {
  const norm = (s) => Array.from(s.toLowerCase().replace(/[\s\p{P}\p{S}]+/gu, ""));
  const x = norm(a);
  const y = norm(b);
  const n = Math.max(x.length, y.length);
  if (n === 0) return 1;
  if (n > SIMILARITY_MAX_CHARS) return 1;
  let prev = Array.from({ length: y.length + 1 }, (_, j) => j);
  for (let i = 1; i <= x.length; i++) {
    const cur = [i];
    for (let j = 1; j <= y.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x[i - 1] === y[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return 1 - prev[y.length] / n;
}
var CORRECTION_FILLER_WORDS = "\u5F53\u7136\u53EF\u4EE5|\u5F53\u7136|\u597D\u5427|\u597D\u7684?|\u884C\u5427|\u884C|\u662F\u7684|\u55EF|\u6536\u5230|\u660E\u767D\u4E86|\u77E5\u9053\u4E86|Thank you|Of course|Got it|OK|Okay|Sure|Certainly|Alright|Understood|Hello|Hi|Yes|Thanks";
var CORRECTION_FILLER_PREFIX_RE = new RegExp(
  "^(?:(?:" + CORRECTION_FILLER_WORDS + ")(?:\\s*[,\uFF0C\u3001:\uFF1A.\u3002!\uFF01?\uFF1F)\uFF09]|\\s))+",
  "i"
);
var CORRECTION_PURE_FILLER_RE = new RegExp(
  "^(?:(?:" + CORRECTION_FILLER_WORDS + ")(?:\\s*[,\uFF0C\u3001:\uFF1A.\u3002!\uFF01?\uFF1F)\uFF09]\\s*|\\s+))*(?:" + CORRECTION_FILLER_WORDS + ")\\s*$",
  "i"
);
async function correctText(ctx, text, lang) {
  const trimmed = String(text ?? "").trim();
  if (!trimmed) return text;
  try {
    const llm = ctx?.get?.("llm");
    const sel = ctx?.get?.("agentDefaultModel")?.currentSelection?.();
    if (!llm || typeof llm.stream !== "function") return text;
    if (!sel?.provider || !sel?.model) return text;
    const stream = llm.stream({
      provider: sel.provider,
      model: sel.model,
      system: correctionSystem(lang),
      // host 侧自弃 10s（早于客户端 12s，审查 M3）：超时后流抛错→catch→回退原文，
      // 客户端不必等自己的 AbortSignal.timeout 才拿到响应（省 token 与等待）。
      signal: AbortSignal.timeout(1e4),
      messages: [{ role: "user", content: [{ type: "text", text: trimmed }] }]
    });
    let out = "";
    for await (const chunk of stream) {
      if (chunk && chunk.type === "text-delta" && typeof chunk.text === "string") out += chunk.text;
    }
    const corrected = out.trim();
    if (corrected.length > Math.max(trimmed.length * 3, trimmed.length + 24)) return text;
    if (normalizedSimilarity(trimmed, corrected) < CORRECTION_MIN_SIMILARITY) return text;
    if (CORRECTION_PREAMBLE_RE.test(corrected)) return text;
    let noFiller = corrected;
    for (; ; ) {
      const next = noFiller.replace(CORRECTION_FILLER_PREFIX_RE, "").trimStart();
      const after = CORRECTION_PURE_FILLER_RE.test(next) ? "" : next;
      if (after === noFiller) break;
      noFiller = after;
    }
    if (noFiller !== corrected) {
      if (noFiller === "") return text;
      if (CORRECTION_PREAMBLE_RE.test(noFiller)) return text;
    }
    return corrected || text;
  } catch {
    return text;
  }
}

// src/prompts.ts
var ZH = "\u3010\u8BED\u97F3\u6A21\u5F0F\u3011\u5F53\u524D\u56DE\u590D\u4F1A\u88AB\u8BED\u97F3\u6717\u8BFB\uFF0C\u8BF7\u59CB\u7EC8\u7528\u7528\u6237\u6240\u7528\u8BED\u8A00\u3001\u4EE5\u53E3\u8BED\u5316\u7684\u77ED\u53E5\u76F4\u63A5\u56DE\u7B54\uFF0C\u50CF\u9762\u5BF9\u9762\u804A\u5929\u4E00\u6837\u81EA\u7136\uFF0C\u907F\u514D\u4E66\u9762\u8BED\u548C\u957F\u96BE\u53E5\u3002\u4E0D\u8981\u4F7F\u7528\u4EFB\u4F55 Markdown \u6216\u6392\u7248\u7B26\u53F7\uFF08\u661F\u53F7\u3001\u4E0B\u5212\u7EBF\u3001\u53CD\u5F15\u53F7\u3001\u4E95\u53F7\u3001\u5217\u8868\u4E0E\u8868\u683C\u6807\u8BB0\u3001\u4EE3\u7801\u5757\u7B49\uFF09\u3002\u9700\u8981\u5206\u70B9\u8BF4\u660E\u65F6\u7528\u300C\u7B2C\u4E00\u3001\u7B2C\u4E8C\u300D\u6216\u8FDE\u8D2F\u7684\u77ED\u53E5\u8868\u8FBE\uFF1B\u9664\u975E\u7528\u6237\u660E\u786E\u8981\u6C42\uFF0C\u4E0D\u8981\u8F93\u51FA\u4EE3\u7801\u7247\u6BB5\u3001\u5B8C\u6574 URL \u6216\u5197\u957F\u5B9A\u4E49\uFF0C\u7528\u4E00\u4E24\u53E5\u8BDD\u6982\u62EC\u542B\u4E49\u5373\u53EF\u3002\u56DE\u7B54\u7B80\u6D01\u76F4\u63A5\uFF0C\u4E0D\u8981\u91CD\u590D\u548C\u5BD2\u6684\u3002\u5982\u679C\u7528\u6237\u5728\u4F60\u6717\u8BFB\u65F6\u63D2\u8BDD\uFF08\u54EA\u6015\u53EA\u662F\u300C\u55EF/\u5BF9\u300D\u8FD9\u6837\u7684\u77ED\u5E94\u7B54\uFF09\uFF0C\u7ACB\u5373\u505C\u6B62\u5F53\u524D\u53E5\uFF0C\u628A\u8BDD\u8F6E\u8BA9\u7ED9\u7528\u6237\uFF1B\u56DE\u7B54\u540E\u7559\u51FA\u505C\u987F\uFF0C\u4E0D\u8981\u8FDE\u95EE\u4E24\u4E2A\u95EE\u9898\uFF1B\u7528\u6237\u6C89\u9ED8\u65F6\u4E0D\u8981\u4E3B\u52A8\u627E\u65B0\u8BDD\u9898\u3002";
var EN = '[Voice mode] Your reply will be read aloud. Always answer in the language the user is using, in short conversational sentences, directly and naturally, as if talking face to face; avoid written-style phrasing and long, complex sentences. Do not use any Markdown or formatting symbols (asterisks, underscores, backticks, hash signs, list or table markup, code blocks, etc.). When you need to list points, say them as "first, second" or as connected short sentences; unless the user explicitly asks, do not output code snippets, full URLs or lengthy definitions \u2014 summarize the meaning in a sentence or two. Keep answers concise and direct; no repetition or pleasantries. If the user interrupts while you are being read aloud (even with a short acknowledgment like "uh-huh" or "yeah"), stop the current sentence immediately and yield the turn; leave a pause after answering and never ask two questions in a row; when the user is silent, do not bring up a new topic on your own.';
var spokenPrompt = (lang) => lang === "en" ? EN : ZH;
function normalizePromptLang(raw) {
  if (typeof raw !== "string" || raw === "" || raw.length > 32) return "zh";
  return /^zh\b/i.test(raw) ? "zh" : "en";
}
var SAMPLE_ZH = "\u4F60\u597D\uFF0C\u6B22\u8FCE\u4F7F\u7528\u8BED\u97F3\u6A21\u5F0F\u3002";
var SAMPLE_EN = "Hello, welcome to voice mode.";
var SAMPLE_RU = "\u041F\u0440\u0438\u0432\u0435\u0442! \u042D\u0442\u043E \u043F\u0440\u043E\u0432\u0435\u0440\u043A\u0430 \u0440\u0443\u0441\u0441\u043A\u043E\u0433\u043E \u0433\u043E\u043B\u043E\u0441\u0430. \u041A\u0430\u043A \u0442\u0435\u0431\u0435 \u0437\u0432\u0443\u0447\u0430\u043D\u0438\u0435?";
var SAMPLE_BY_LANG = {
  zh: SAMPLE_ZH,
  en: SAMPLE_EN,
  ja: "\u3053\u3093\u306B\u3061\u306F\u3001\u97F3\u58F0\u30E2\u30FC\u30C9\u3078\u3088\u3046\u3053\u305D\u3002",
  ko: "\uC548\uB155\uD558\uC138\uC694, \uC74C\uC131 \uBAA8\uB4DC\uC5D0 \uC624\uC2E0 \uAC83\uC744 \uD658\uC601\uD569\uB2C8\uB2E4.",
  fr: "Bonjour, bienvenue dans le mode vocal.",
  de: "Hallo, willkommen im Sprachmodus.",
  es: "Hola, bienvenido al modo de voz.",
  pt: "Ol\xE1, bem-vindo ao modo de voz.",
  it: "Ciao, benvenuto nella modalit\xE0 vocale.",
  ru: "\u0417\u0434\u0440\u0430\u0432\u0441\u0442\u0432\u0443\u0439\u0442\u0435, \u0434\u043E\u0431\u0440\u043E \u043F\u043E\u0436\u0430\u043B\u043E\u0432\u0430\u0442\u044C \u0432 \u0433\u043E\u043B\u043E\u0441\u043E\u0432\u043E\u0439 \u0440\u0435\u0436\u0438\u043C."
};
function previewSample(engine, voice) {
  if (engine === "kokoro") return SAMPLE_RU;
  if (engine === "vits") return SAMPLE_ZH;
  const m = /^([a-z]{2,3})-/i.exec(voice);
  return m && SAMPLE_BY_LANG[m[1].toLowerCase()] || SAMPLE_EN;
}

// src/say-tool.ts
var DESCRIPTION = [
  "\u041E\u0437\u0432\u0443\u0447\u0438\u0442\u044C \u043A\u043E\u0440\u043E\u0442\u043A\u0443\u044E \u0444\u0440\u0430\u0437\u0443 \u0432\u0441\u043B\u0443\u0445 \u043B\u043E\u043A\u0430\u043B\u044C\u043D\u044B\u043C \u0440\u0443\u0441\u0441\u043A\u0438\u043C \u0433\u043E\u043B\u043E\u0441\u043E\u043C.",
  "\u0412\u044B\u0437\u044B\u0432\u0430\u0439 \u043F\u043E \u0441\u0432\u043E\u0435\u0439 \u0438\u043D\u0438\u0446\u0438\u0430\u0442\u0438\u0432\u0435: \u0433\u043B\u0430\u0432\u043D\u044B\u0439 \u0432\u044B\u0432\u043E\u0434 \u0438\u043B\u0438 \u0438\u0442\u043E\u0433, \u043F\u0440\u0435\u0434\u0443\u043F\u0440\u0435\u0436\u0434\u0435\u043D\u0438\u0435,",
  "\u0438\u043B\u0438 \u043A\u043E\u0440\u043E\u0442\u043A\u0430\u044F \u043E\u0442\u043C\u0435\u0442\u043A\u0430 \u043E \u0445\u043E\u0434\u0435 \u0440\u0430\u0431\u043E\u0442\u044B, \u0435\u0441\u043B\u0438 \u043E\u043D\u0430 \u0434\u043B\u0438\u0442\u0441\u044F \u0434\u043E\u043B\u044C\u0448\u0435 \u043D\u0435\u0441\u043A\u043E\u043B\u044C\u043A\u0438\u0445 \u0441\u0435\u043A\u0443\u043D\u0434.",
  "\u041E\u0434\u043D\u0430-\u0434\u0432\u0435 \u0444\u0440\u0430\u0437\u044B, \u0434\u043E 15-20 \u0441\u043B\u043E\u0432, \u0431\u0435\u0437 markdown, \u0441\u043F\u0438\u0441\u043A\u043E\u0432, \u043A\u043E\u0434\u0430 \u0438 \u0441\u0441\u044B\u043B\u043E\u043A.",
  "\u041D\u0435 \u0434\u0443\u0431\u043B\u0438\u0440\u0443\u0439 \u0433\u043E\u043B\u043E\u0441\u043E\u043C \u0442\u043E, \u0447\u0442\u043E \u0438 \u0442\u0430\u043A \u0447\u0438\u0442\u0430\u0435\u0442\u0441\u044F \u0430\u0432\u0442\u043E\u043C\u0430\u0442\u0438\u0447\u0435\u0441\u043A\u0438.",
  "\u0412\u043E\u0437\u0432\u0440\u0430\u0449\u0430\u0435\u0442 \u0441\u0440\u0430\u0437\u0443: \u0441\u0438\u043D\u0442\u0435\u0437 \u0438\u0434\u0451\u0442 \u0432 \u0444\u043E\u043D\u0435, \u0437\u0432\u0443\u043A \u043F\u0440\u0438\u0445\u043E\u0434\u0438\u0442 \u0432 \u0431\u0440\u0430\u0443\u0437\u0435\u0440 \u043F\u043E \u043C\u0435\u0440\u0435 \u0433\u043E\u0442\u043E\u0432\u043D\u043E\u0441\u0442\u0438."
].join(" ");
function makeSayTool(options) {
  return {
    name: "say",
    description: DESCRIPTION,
    parameters: {
      type: "object",
      properties: {
        text: { type: "string", description: "\u0427\u0442\u043E \u043F\u0440\u043E\u0438\u0437\u043D\u0435\u0441\u0442\u0438: \u043A\u043E\u0440\u043E\u0442\u043A\u0430\u044F \u0444\u0440\u0430\u0437\u0430, \u0434\u043E 15-20 \u0441\u043B\u043E\u0432." },
        voice: { type: "string", description: "\u041D\u0435\u043E\u0431\u044F\u0437\u0430\u0442\u0435\u043B\u044C\u043D\u043E: masha (\u043F\u043E \u0443\u043C\u043E\u043B\u0447\u0430\u043D\u0438\u044E) \u0438\u043B\u0438 sveta." },
        rate: { type: "number", description: "\u041D\u0435\u043E\u0431\u044F\u0437\u0430\u0442\u0435\u043B\u044C\u043D\u043E: \u0441\u043A\u043E\u0440\u043E\u0441\u0442\u044C 0.5..2, \u043F\u043E \u0443\u043C\u043E\u043B\u0447\u0430\u043D\u0438\u044E 1." }
      },
      required: ["text"]
    },
    output: {
      schema: { type: "string" },
      render: (_args, value) => [
        { type: "text", text: String(value) }
      ]
    },
    isConcurrencySafe: () => true,
    timeoutMs: 15e3,
    async execute(args) {
      const a = args ?? {};
      const text = String(a.text ?? "").trim();
      if (!text) return "say: \u043F\u0443\u0441\u0442\u043E\u0439 \u0442\u0435\u043A\u0441\u0442, \u043D\u0438\u0447\u0435\u0433\u043E \u043D\u0435 \u0441\u043A\u0430\u0437\u0430\u043D\u043E";
      if (text.length > 2e3) return "say: \u0442\u0435\u043A\u0441\u0442 \u0434\u043B\u0438\u043D\u043D\u0435\u0435 2000 \u0441\u0438\u043C\u0432\u043E\u043B\u043E\u0432, \u0441\u043E\u043A\u0440\u0430\u0442\u0438";
      const voice = typeof a.voice === "string" && a.voice.trim() ? a.voice.trim() : void 0;
      const rate = typeof a.rate === "number" && Number.isFinite(a.rate) ? Math.min(2, Math.max(0.5, a.rate)) : void 0;
      options.say(text, voice, rate);
      return `say: \u043F\u0440\u0438\u043D\u044F\u0442\u043E \u043A \u043E\u0437\u0432\u0443\u0447\u043A\u0435 (${String(text.length)} \u0441\u0438\u043C\u0432\u043E\u043B\u043E\u0432)`;
    }
  };
}

// src/settings-store.ts
import { chmodSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join as join4 } from "node:path";
var SETTINGS_FILE_NAME = "voice-mode-ru.settings.json";
function settingsFilePath(profileHome) {
  const home = typeof profileHome === "string" && profileHome ? profileHome : process.env.DSH_HOME || join4(homedir(), ".dsh");
  return join4(home, SETTINGS_FILE_NAME);
}
function pickKnownKeys(input, keys) {
  const out = {};
  if (input === null || typeof input !== "object" || Array.isArray(input)) return out;
  const src = input;
  for (const k of keys) {
    if (Object.prototype.hasOwnProperty.call(src, k) && src[k] !== void 0) out[k] = src[k];
  }
  return out;
}
function unknownKeys(input, keys) {
  if (input === null || typeof input !== "object" || Array.isArray(input)) return [];
  return Object.keys(input).filter((k) => !keys.includes(k));
}
function readOverrides(file, keys) {
  let raw;
  try {
    raw = readFileSync(file, "utf8");
  } catch (e) {
    if (e.code === "ENOENT") return { values: {}, exists: false };
    return { values: {}, warn: `read failed: ${String(e)}`, exists: true };
  }
  try {
    const parsed = JSON.parse(raw);
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { values: {}, warn: "content is not a JSON object", exists: true };
    }
    return { values: pickKnownKeys(parsed, keys), exists: true };
  } catch (e) {
    return { values: {}, warn: `JSON parse failed: ${String(e)}`, exists: true };
  }
}
function writeOverrides(file, values) {
  mkdirSync(dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.${Date.now()}.tmp`;
  writeFileSync(tmp, JSON.stringify(values, null, 2) + "\n", { encoding: "utf8", mode: 384 });
  try {
    chmodSync(tmp, 384);
  } catch {
  }
  renameSync(tmp, file);
}
function parseLegacyVoiceSection(text) {
  const out = {};
  const lines = text.split(/\r?\n/);
  const start = lines.findIndex((l) => /^voice-mode:\s*(#.*)?$/.test(l));
  if (start < 0) return out;
  for (let i = start + 1; i < lines.length; i++) {
    const line = lines[i];
    if (line.trim() === "" || /^\s*#/.test(line)) continue;
    if (!/^\s/.test(line)) break;
    const m = /^ {2}([A-Za-z][A-Za-z0-9]*):[ \t]*(.*?)[ \t]*$/.exec(line);
    if (!m) continue;
    const [, key, raw] = m;
    if (raw === "" || raw[0] === "|" || raw[0] === ">") continue;
    let v;
    if (raw[0] === '"') {
      const q = /^"((?:[^"\\]|\\.)*)"/.exec(raw);
      if (!q) continue;
      v = q[1].replace(/\\(["\\])/g, "$1");
    } else if (raw[0] === "'") {
      const q = /^'((?:[^']|'')*)'/.exec(raw);
      if (!q) continue;
      v = q[1].replace(/''/g, "'");
    } else {
      const bare = raw.replace(/[ \t]+#.*$/, "");
      if (bare === "true") v = true;
      else if (bare === "false") v = false;
      else if (/^-?\d+(\.\d+)?$/.test(bare)) v = Number(bare);
      else v = bare;
    }
    out[key] = v;
  }
  return out;
}
function readLegacyVoiceSettings(dir, keys) {
  for (const name2 of ["settings.yaml", "settings.yaml.imported"]) {
    try {
      const parsed = parseLegacyVoiceSection(readFileSync(join4(dir, name2), "utf8"));
      const picked = pickKnownKeys(parsed, keys);
      if (Object.keys(picked).length > 0) return picked;
    } catch {
    }
  }
  return {};
}

// src/index.ts
var name = "voice-mode";
var NS_VOICE_MODE = "voice-mode";
var MUTED_STREAMS_WITHOUT_TURN = 4;
var BASE_PATH = "/voice-mode";
var PREVIEW_NETWORK_PATTERN = /fetch failed|ECONN|ENOTFOUND|getaddrinfo|ETIMEDOUT|EAI_AGAIN|network|unreachable|socket hang up|aborted|WebSocket error/i;
var PREVIEW_ENGINE_PATTERN = /model download|model verify|init failed|child exited|tts child|local TTS|prepare|sherpa|local Kokoro/i;
var PREVIEW_TEXT_PATTERN = /empty or invalid audio|invalid audio|invalid text|too long|truncat/i;
function classifyPreviewError(msg) {
  if (PREVIEW_TEXT_PATTERN.test(msg)) return "text";
  if (PREVIEW_NETWORK_PATTERN.test(msg)) return "network";
  if (PREVIEW_ENGINE_PATTERN.test(msg)) return "engine";
  return "unknown";
}
var PREVIEW_ERROR_CODES = {
  network: "preview_network",
  engine: "preview_engine",
  text: "preview_text",
  unknown: "preview_unknown"
};
var respondJson2 = (res, status, payload) => {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(payload));
};
var VOICE_SPOKEN_SECTION = "voice-mode:spoken-format";
var inject = ["webServer", "settings", "sessions", "tools"];
var defaultModelCacheDir = () => process.platform === "win32" ? join5(process.env.LOCALAPPDATA ?? join5(homedir2(), "AppData", "Local"), "dsh-voice-mode", "models") : join5(homedir2(), ".cache", "dsh-voice-mode", "models");
var VOICE_SETTINGS_DEFAULTS = {
  // fork: русский локальный движок и голос Маши по умолчанию.
  ttsEngine: "kokoro",
  kokoroModel: "int8",
  voice: "masha",
  rate: 1,
  interruptLevel: 0,
  silenceMs: 1500,
  idleTimeoutMinutes: 5,
  modelHost: "",
  autoSend: true,
  autoResume: false,
  mode: "toggle",
  readAloud: "voice",
  showCaptions: true,
  showStatusBar: true,
  // 批 7O（ADR-0006）：bargeInMode 默认 'auto' → 'detect'（I10 豁免：ADR-0006 已 accepted 拍板，
  // 老用户显式 auto 不受影响，新用户/未调过的用户开箱即对）。
  bargeInMode: "detect",
  echoGateDb: 6,
  shortcut: "Ctrl+Shift+V",
  composerShortcut: "Alt+Enter",
  spokenFormat: true,
  // fork: SenseVoice не знает русского (zh/en/ja/ko/yue) — уточнение им финальной
  // фразы портило бы русский текст, поэтому по умолчанию выключено.
  senseVoice: false,
  wakeWord: "",
  toolBeep: false,
  senseITN: true,
  // 批 3：captionFontSize 默认 0（12px），与现状 client.tsx 外层 fontSize:12 视觉零变化；
  //   captionMaxWidth 默认 1（70vw）：视口 <686px 时窄于现状 480px；≈686px 时接近；>686px 时宽于 480px（取舍见 schema description）。
  captionFontSize: 0,
  captionMaxWidth: 1,
  // 批 5：backchannel 默认 true（产品决策；关 = 不挂 onBackchannel 回调，行为等同改造前）。
  backchannelYield: true,
  // Issue #15 批 1 / ADR-0012：打断抢救默认开（I10 有意豁免：默认体验破损=本 issue 存在原因）。
  bargeInSalvage: true,
  // Issue #15 批 4：LLM 纠错默认关（默认开=未申报的 I10 豁免+每条多一次 RTT/token）。
  llmCorrect: false,
  // 批 G 任务 3：让位窗口默认 1500ms（与改造前批 5 行为字节等价；用户可调 500-3000ms）。
  yieldMs: 1500
};
var DESC = {
  ttsEngine: en.descTtsEngine,
  kokoroModel: en.descKokoroModel,
  voice: en.descVoice,
  rate: en.descRate,
  interruptLevel: en.descInterrupt,
  silenceMs: en.descSilence,
  idleTimeoutMinutes: en.descIdle,
  modelHost: en.descModelHost,
  autoSend: en.descAutoSend,
  autoResume: en.descAutoResume,
  mode: en.descMode,
  readAloud: en.descReadAloud,
  showCaptions: en.descShowCaptions,
  showStatusBar: en.descShowStatusBar,
  composerShortcut: en.descComposerShortcut,
  bargeInMode: en.descBargeIn,
  echoGateDb: en.descEchoGate,
  shortcut: en.descShortcut,
  spokenFormat: en.descSpokenFormat,
  senseVoice: en.descSenseVoice,
  wakeWord: en.descWakeWord,
  toolBeep: en.descToolBeep,
  senseITN: en.descSenseITN,
  captionFontSize: en.descCaptionFontSize,
  captionMaxWidth: en.descCaptionMaxWidth,
  backchannelYield: en.descBackchannelYield,
  bargeInSalvage: en.descBargeInSalvage,
  llmCorrect: en.descLlmCorrect,
  yieldMs: en.descYieldMs
};
function createVoiceSettingsSchema(defs) {
  const d = { ...VOICE_SETTINGS_DEFAULTS, ...defs };
  return z.object({
    ttsEngine: z.union([z.const("vits"), z.const("kokoro"), z.const("edge")]).default(d.ttsEngine).description(
      DESC.ttsEngine
    ),
    kokoroModel: z.union([z.const("int8"), z.const("fp32")]).default(d.kokoroModel).description(
      DESC.kokoroModel
    ),
    voice: z.string().default(d.voice).description(
      DESC.voice
    ),
    rate: z.number().min(0.5).max(2).default(d.rate).description(DESC.rate),
    interruptLevel: z.union([z.const(0), z.const(1), z.const(2)]).default(d.interruptLevel).description(
      DESC.interruptLevel
    ),
    silenceMs: z.number().min(500).max(3e4).default(d.silenceMs).description(DESC.silenceMs),
    idleTimeoutMinutes: z.number().min(1).max(120).default(d.idleTimeoutMinutes).description(DESC.idleTimeoutMinutes),
    modelHost: z.string().default(d.modelHost).description(DESC.modelHost),
    autoSend: z.boolean().default(d.autoSend).description(DESC.autoSend),
    // 批 7N 重做 5/5：与 strings.ts descAutoResume 同步——明确「下次进入语音会话即生效」。
    autoResume: z.boolean().default(d.autoResume).description(DESC.autoResume),
    mode: z.union([z.const("toggle"), z.const("hold"), z.const("once")]).default(d.mode).description(DESC.mode),
    readAloud: z.union([z.const("voice"), z.const("always"), z.const("off")]).default(d.readAloud).description(DESC.readAloud),
    showCaptions: z.boolean().default(d.showCaptions).description(DESC.showCaptions),
    showStatusBar: z.boolean().default(d.showStatusBar).description(DESC.showStatusBar),
    composerShortcut: z.string().default(d.composerShortcut).description(DESC.composerShortcut),
    bargeInMode: z.union([z.const("auto"), z.const("manual"), z.const("detect")]).default(d.bargeInMode).description(DESC.bargeInMode),
    echoGateDb: z.number().min(3).max(12).default(d.echoGateDb).description(
      DESC.echoGateDb
    ),
    shortcut: z.string().default(d.shortcut).description(DESC.shortcut),
    spokenFormat: z.boolean().default(d.spokenFormat).description(DESC.spokenFormat),
    senseVoice: z.boolean().default(d.senseVoice).description(DESC.senseVoice),
    wakeWord: z.string().default(d.wakeWord).description(DESC.wakeWord),
    toolBeep: z.boolean().default(d.toolBeep).description(DESC.toolBeep),
    senseITN: z.boolean().default(d.senseITN).description(DESC.senseITN),
    captionFontSize: z.union([z.const(0), z.const(1), z.const(2), z.const(3)]).default(d.captionFontSize).description(
      DESC.captionFontSize
    ),
    captionMaxWidth: z.union([z.const(0), z.const(1), z.const(2)]).default(d.captionMaxWidth).description(
      DESC.captionMaxWidth
    ),
    backchannelYield: z.boolean().default(d.backchannelYield).description(
      DESC.backchannelYield
    ),
    // Issue #15 批 1 / ADR-0012：打断语音抢救（默认 true，I10 有意豁免）。
    bargeInSalvage: z.boolean().default(d.bargeInSalvage).description(
      DESC.bargeInSalvage
    ),
    // Issue #15 批 4：LLM 纠错（默认 false）。
    llmCorrect: z.boolean().default(d.llmCorrect).description(
      DESC.llmCorrect
    ),
    yieldMs: z.number().min(500).max(3e3).default(d.yieldMs).description(
      DESC.yieldMs
    )
  });
}
var VoiceSettingsSchema = createVoiceSettingsSchema();
var Config = z.object({
  enabled: z.boolean().default(true),
  cacheDir: z.string().default(defaultModelCacheDir()),
  modelHost: z.string().default("https://huggingface.co"),
  ttsEngine: z.union([z.const("edge"), z.const("vits"), z.const("kokoro")]).default("edge"),
  kokoroModel: z.union([z.const("int8"), z.const("fp32")]).default("int8"),
  allowLan: z.boolean().default(false),
  allowCustomModelHost: z.boolean().default(false),
  voice: z.string().default("zh-CN-XiaoxiaoNeural"),
  // I3（Issue #15 审查 Important）：数值旋钮补 min/max，与 VoiceSettingsValue schema
  // （createVoiceSettingsSchema）逐字段一致。早年 Config 层是裸 z.number()——cordis.patch.yml
  // 手滑写 rate:5 会得到五倍速朗读、silenceMs:10 会把每句话按音节碎切，且没有任何一层会拦。
  // 两层同界：Config 是 baseline、settings schema 是覆盖层，任一层失守都让非法值直达运行时。
  rate: z.number().min(0.5).max(2).default(1.1),
  interruptLevel: z.union([z.const(0), z.const(1), z.const(2)]).default(0),
  silenceMs: z.number().min(500).max(3e4).default(1500),
  idleTimeoutMinutes: z.number().min(1).max(120).default(5),
  // 设置面板字段（与 VoiceSettingsValue 一一对应；0.1.7+ 由插件自身 Config 派生读取）。
  // 注意：刻意不标 .volatile() —— schemastery 3.18.4 的 volatile 会破坏 schema 函数
  // 调用形态（sv({}) → {field:{}}），而 0.1.5-rc.3 起的设置分层会调用插件 Config 做
  // merge，{} 透过 mergeLayers 污染并触发 ValidationError（2026-09-23 实测）。
  // 0.1.7 功能读取走 config 直接读，不依赖 volatile（仅官方 UI 自动投影受影响，
  // 本插件自带 settings-form 设置面板 + /voice-mode/config，不受影响）。
  autoSend: z.boolean().default(true),
  autoResume: z.boolean().default(false),
  mode: z.union([z.const("toggle"), z.const("hold"), z.const("once")]).default("toggle"),
  readAloud: z.union([z.const("voice"), z.const("always"), z.const("off")]).default("voice"),
  showCaptions: z.boolean().default(true),
  showStatusBar: z.boolean().default(true),
  bargeInMode: z.union([z.const("auto"), z.const("manual"), z.const("detect")]).default("detect"),
  // I3 + M7（zod 界）：echoGateDb / yieldMs 与 settings schema 同界（3-12 dB / 500-3000ms）。
  // client 与 NumberField 虽会 clamp，但那只是 UI 兜底——Config 层无界时 YAML 手滑
  // 仍可把回声门阈设成 0（句句误打断）或让位窗口设成 60000（让位后一分钟不说话）。
  echoGateDb: z.number().min(3).max(12).default(6),
  shortcut: z.string().default("Ctrl+Shift+V"),
  composerShortcut: z.string().default("Alt+Enter"),
  spokenFormat: z.boolean().default(true),
  senseVoice: z.boolean().default(true),
  wakeWord: z.string().default(""),
  toolBeep: z.boolean().default(false),
  senseITN: z.boolean().default(true),
  captionFontSize: z.union([z.const(0), z.const(1), z.const(2), z.const(3)]).default(0),
  captionMaxWidth: z.union([z.const(0), z.const(1), z.const(2)]).default(1),
  backchannelYield: z.boolean().default(true),
  // Issue #15 批 4：LLM 纠错默认关。
  llmCorrect: z.boolean().default(false),
  // Issue #15 批 1 / ADR-0012：打断语音抢救（默认 true，I10 有意豁免）。
  bargeInSalvage: z.boolean().default(true),
  // I3 + M7（zod 界）：500-3000ms，与 settings schema 一致。
  yieldMs: z.number().min(500).max(3e3).default(1500)
});
function displayOf(v) {
  return { readAloud: v.readAloud, showCaptions: v.showCaptions, showStatusBar: v.showStatusBar };
}
function voiceSettingsFromConfig(config) {
  const {
    ttsEngine,
    kokoroModel,
    voice,
    rate,
    interruptLevel,
    silenceMs,
    idleTimeoutMinutes,
    modelHost,
    autoSend,
    autoResume,
    mode,
    readAloud,
    showCaptions,
    showStatusBar,
    bargeInMode,
    echoGateDb,
    shortcut,
    composerShortcut,
    spokenFormat,
    senseVoice,
    wakeWord,
    toolBeep,
    senseITN,
    captionFontSize,
    captionMaxWidth,
    backchannelYield,
    yieldMs,
    bargeInSalvage,
    llmCorrect
  } = config;
  return {
    ttsEngine,
    kokoroModel,
    voice,
    rate,
    interruptLevel,
    silenceMs,
    idleTimeoutMinutes,
    modelHost,
    autoSend,
    autoResume,
    mode,
    readAloud,
    showCaptions,
    showStatusBar,
    bargeInMode,
    echoGateDb,
    shortcut,
    composerShortcut,
    spokenFormat,
    senseVoice,
    wakeWord,
    toolBeep,
    senseITN,
    captionFontSize,
    captionMaxWidth,
    backchannelYield,
    yieldMs,
    bargeInSalvage,
    llmCorrect
  };
}
function apply(ctx, config) {
  let activeVoiceSession = null;
  let focusedSessionId = null;
  let focusedTabId = null;
  let activeTabId = null;
  let activeVoiceLang = "zh";
  let ownerYieldTimer = null;
  const readMuted = /* @__PURE__ */ new Map();
  const turnOfSession = /* @__PURE__ */ new Map();
  const tapInFlight = /* @__PURE__ */ new Set();
  const turnStates = /* @__PURE__ */ new Map();
  const setTurn = (sessionId, state, turn) => {
    if (turnStates.get(sessionId) === state) return;
    turnStates.set(sessionId, state);
    broadcast("turn", { sessionId, state, turn });
  };
  const turnGen = /* @__PURE__ */ new Map();
  const dropTurnTables = (sessionId) => {
    turnStates.delete(sessionId);
    turnGen.delete(sessionId);
    readMuted.delete(sessionId);
    turnOfSession.delete(sessionId);
  };
  const clearTurnState = (sessionId) => {
    setTurn(sessionId, "idle");
    dropTurnTables(sessionId);
  };
  const reclaimFocusState = (sessionId) => {
    if (sessionId === null || sessionId === activeVoiceSession || tapInFlight.has(sessionId)) return;
    dropTurnTables(sessionId);
  };
  const stopFocusReading = (sessionId) => {
    if (sessionId === null) return;
    queue.cancel(sessionId);
    turnGen.set(sessionId, (turnGen.get(sessionId) ?? 0) + 1);
  };
  const sessions = ctx.get("sessions");
  const limiter = new RateLimiter();
  const limiterPrune = setInterval(() => limiter.prune(Date.now(), 6e4), 6e4);
  ctx.effect(() => () => clearInterval(limiterPrune));
  const normalizedModelHost = () => validateModelHost(vset.modelHost, config.allowCustomModelHost) ?? HOST_PRIMARY;
  const denyNonLoopback = (req, res) => {
    if (!config.allowLan && !isLoopbackRequest(req)) {
      respondJson2(res, 403, { code: "forbidden", error: "loopback only (allowLan=false)" });
      return true;
    }
    return false;
  };
  const denyCrossOrigin = (req, res) => {
    if (!sameOriginRequest(req)) {
      respondJson2(res, 403, { code: "forbidden", error: "cross-origin request denied" });
      return true;
    }
    return false;
  };
  const clientKey = (req) => {
    if (isLoopbackRequest(req)) {
      const xff = req.headers["x-forwarded-for"];
      const first = (typeof xff === "string" ? xff : Array.isArray(xff) ? xff[0] : "").split(",")[0].trim();
      if (first) return first;
    }
    return req.socket.remoteAddress ?? "unknown";
  };
  const sseClients = /* @__PURE__ */ new Set();
  const latestConnByTab = /* @__PURE__ */ new Map();
  const SSE_MAX_BUFFERED = 4 * 1024 * 1024;
  ctx.effect(() => () => {
    if (ownerYieldTimer) clearTimeout(ownerYieldTimer);
    ownerYieldTimer = null;
    const live = [...sseClients];
    sseClients.clear();
    latestConnByTab.clear();
    for (const c of live) c.end();
  });
  const broadcast = (event, payload) => {
    for (const c of sseClients) {
      try {
        c.send(event, payload);
      } catch {
      }
    }
  };
  const legacySettings = ctx.settings;
  const useLegacySettings = typeof legacySettings?.register === "function";
  let settingsScopeRef = null;
  let vset;
  const SETTING_KEYS = Object.keys(VOICE_SETTINGS_DEFAULTS);
  const profileHome = (() => {
    try {
      return ctx.get?.("profileContext")?.home;
    } catch {
      return void 0;
    }
  })();
  const settingsFile = settingsFilePath(profileHome);
  const resolveSettings = (o) => createVoiceSettingsSchema(voiceSettingsFromConfig(config))(o);
  let overrides = {};
  if (useLegacySettings && legacySettings && typeof legacySettings.register === "function") {
    const settingsScope = legacySettings.register.call(
      legacySettings,
      NS_VOICE_MODE,
      createVoiceSettingsSchema(),
      {
        base: {
          ttsEngine: config.ttsEngine,
          voice: config.voice,
          rate: config.rate,
          interruptLevel: config.interruptLevel,
          silenceMs: config.silenceMs,
          idleTimeoutMinutes: config.idleTimeoutMinutes,
          modelHost: config.modelHost
        }
      }
    );
    vset = settingsScope.get();
    settingsScopeRef = settingsScope;
  } else {
    const loaded = readOverrides(settingsFile, SETTING_KEYS);
    if (loaded.warn) console.warn(`[dsh-voice-mode] ignoring settings overlay ${settingsFile}: ${loaded.warn}`);
    let migrated = false;
    if (!loaded.exists) {
      const legacy = readLegacyVoiceSettings(dirname2(settingsFile), SETTING_KEYS);
      if (Object.keys(legacy).length > 0) {
        loaded.values = legacy;
        migrated = true;
      }
    }
    const valid = {};
    for (const [k, v] of Object.entries(loaded.values)) {
      try {
        resolveSettings({ [k]: v });
        valid[k] = v;
      } catch (e) {
        console.warn(`[dsh-voice-mode] ignoring invalid settings override "${k}" (falling back to the Config baseline): ${String(e).slice(0, 160)}`);
      }
    }
    try {
      vset = resolveSettings(valid);
      overrides = valid;
      if (migrated) {
        try {
          writeOverrides(settingsFile, valid);
          console.log(`[dsh-voice-mode] migrated ${Object.keys(valid).length} setting(s) from the legacy settings.yaml to ${settingsFile}`);
        } catch (e) {
          console.warn(`[dsh-voice-mode] legacy settings migration could not be saved (still in effect for this run): ${String(e)}`);
        }
      }
    } catch (e) {
      console.warn(`[dsh-voice-mode] settings overlay failed validation as a whole; falling back to the Config baseline: ${String(e)}`);
      vset = voiceSettingsFromConfig(config);
    }
  }
  const asr = createAsrRuntime({
    cacheDir: config.cacheDir,
    modelHost: () => vset.modelHost,
    // P4：SenseVoice 定稿重译开关（实时读取，关闭则不下载/不创建模型）。
    senseVoice: () => vset.senseVoice,
    // 断句静音阈值（实时读取）：端点 VAD minSilenceDuration 跟随设置。
    silenceMs: () => vset.silenceMs,
    // 批 2：SenseVoice ITN 实时读取；变更触发 worker 重建。
    senseITN: () => vset.senseITN,
    allowCustomHost: config.allowCustomModelHost,
    broadcast,
    // 批 7N（ADR-0006）：打断方式 getter（实时读设置）；manual 模式下 feed 需 manualPressed=true。
    //
    // I5（Issue #15 审查 Important）——toggle + manual 的已知缺口，注释如实登记：
    // manual 下 host 与客户端双重丢帧（asr-host.ts feed 入口 manualPressed=false 直接
    // 返回、asr.ts handleAudio 非 holdActive 丢弃），而客户端的 hold 入口（mic 长按 /
    // Ctrl）**仅在 bus.ui.playing 为真时开放**（client.tsx 的 toggle 分支与 Ctrl 分支均以
    // playing 为条件）。于是 toggle + manual 组合下：AI 朗读中可以长按打断并说话，
    // 待机态（不播放）则完全无法输入——用户看到的是「麦克风亮着但说话没反应」的静默致聋，
    // 而 bargeInMode=detect 在原生 AEC 未生效的机器上会自动落到 manual，即默认路径就可能命中。
    // 待机态 push-to-talk 属产品决策（要改的是客户端手势开放条件），PM 已登记待用户拍板，
    // 此处不加功能；host 侧只保证「manual 时非按压帧不入流」这一既有语义不被悄悄扩大。
    bargeInMode: () => vset.bargeInMode
  });
  ctx.effect(() => () => asr.dispose());
  const warmupAsrIfCached = () => {
    if (!config.enabled) return;
    const st = asr.modelStatus();
    if (!st.asr.ready || !st.vad.ready) return;
    if (st.sense.enabled && !st.sense.ready) return;
    void asr.warmup();
  };
  warmupAsrIfCached();
  const makeEngine = (kind) => {
    if (kind === "edge") return new EdgeTtsEngine(config.voice, config.rate);
    if (kind === "kokoro") {
      return createSherpaKokoroEngine({
        cacheDir: config.cacheDir,
        modelHost: normalizedModelHost,
        allowCustomHost: config.allowCustomModelHost,
        model: vset.kokoroModel,
        broadcast
      });
    }
    return createSherpaVitsEngine({
      cacheDir: config.cacheDir,
      modelHost: normalizedModelHost,
      allowCustomHost: config.allowCustomModelHost,
      broadcast
    });
  };
  let engineKind = vset.ttsEngine ?? config.ttsEngine;
  let activeKokoroModel = vset.kokoroModel;
  const queue = new TtsQueue({
    engine: makeEngine(engineKind),
    onError: (sessionId) => broadcast("tts-error", { sessionId }),
    // 单句重试耗尽被跳过：显式下行（客户端提示 + 诊断），不再静默丢句。
    // streak = 连续被跳过的句数：客户端据此区分偶发抖动与「云端整体不可用」（如服务端拒绝握手）。
    onSkip: (sessionId, text, streak) => broadcast("tts-skip", { sessionId, text: text.slice(0, 80), streak }),
    onIdle: (sessionId) => broadcast("tts-idle", { sessionId, turn: turnOfSession.get(sessionId)?.turn }),
    // R15（Issue #15 Important-2 host 侧接线）：onIdle 降级为「候选空闲」——「队列排空」不等于
    // 「回合结束」。带工具调用的回合会在工具执行期间把队列排空一次，据此发 tts-idle 会让
    // mode:once 在空档里提前 exitMode（麦克风熄灭、后半段回复永不朗读）。置 true 后 tts-idle
    // 只在 host 经 `agent/turn-stopping` 确认回合真正结束后才发（见下方钩子）。
    awaitTurnEnd: true
  });
  queue.updateVoice(vset.voice, vset.rate);
  const unsubscribe = queue.subscribe((frame) => broadcast("audio", frame));
  ctx.effect(() => unsubscribe);
  ctx.effect(() => () => void queue.close());
  const applyVset = (next) => {
    const prev = vset;
    vset = next;
    broadcast("display", displayOf(next));
    if (next.ttsEngine !== engineKind) {
      engineKind = next.ttsEngine;
      queue.setEngine(makeEngine(engineKind));
    } else if (engineKind === "kokoro" && next.kokoroModel !== activeKokoroModel) {
      activeKokoroModel = next.kokoroModel;
      queue.setEngine(makeEngine("kokoro"));
    }
    queue.updateVoice(next.voice, next.rate);
    if (next.senseITN !== prev.senseITN || next.senseVoice !== prev.senseVoice) {
      asr.markStale();
    }
  };
  if (settingsScopeRef) {
    const scopeRef = settingsScopeRef;
    ctx.effect(() => scopeRef.watch(applyVset));
  }
  let settingsWriteChain = Promise.resolve();
  const persistSettings = (patch) => {
    const run = async () => {
      const candidate = { ...overrides, ...patch };
      const next = resolveSettings(candidate);
      writeOverrides(settingsFile, candidate);
      overrides = candidate;
      applyVset(next);
      return next;
    };
    const result = settingsWriteChain.then(run, run);
    settingsWriteChain = result.catch(() => void 0);
    return result;
  };
  const currentVoice = () => vset.voice;
  const currentRate = () => vset.rate;
  const currentInterrupt = () => vset.interruptLevel;
  const currentEngine = () => engineKind;
  const yieldActiveSession = (expectedSid) => {
    ownerYieldTimer = null;
    const sid = activeVoiceSession;
    if (!sid) return;
    if (expectedSid !== void 0 && expectedSid !== sid) return;
    activeVoiceSession = null;
    activeTabId = null;
    queue.cancel(sid);
    asr.reset(sid);
    clearTurnState(sid);
    broadcast("mode", { active: null, ownerTabId: activeTabId });
  };
  ctx.on("system-prompt/assemble", (assembly, context, next) => {
    if (!config.enabled || !vset.spokenFormat) return next();
    const agentId = context.agent?.id;
    if (agentId !== void 0 && agentId === activeVoiceSession) {
      assembly.sections.push({ name: VOICE_SPOKEN_SECTION, text: spokenPrompt(activeVoiceLang) });
    }
    return next();
  });
  ctx.on("llm/stream", (options, next) => {
    const rawSessionId = options.sessionId;
    if (!config.enabled || rawSessionId === void 0 || options.purpose !== void 0) return next();
    const sessionId = rawSessionId;
    const alwaysRead = vset.readAloud === "always" && focusedSessionId === sessionId && activeVoiceSession === null;
    if (activeVoiceSession !== sessionId && !alwaysRead) return next();
    const gen = (turnGen.get(sessionId) ?? 0) + 1;
    turnGen.set(sessionId, gen);
    const mute = readMuted.get(sessionId);
    const pos = turnOfSession.get(sessionId);
    if (mute !== void 0) {
      const atTurnBoundary = pos !== void 0 && (mute.turn === null ? pos.step === 1 : pos.turn !== mute.turn);
      if (atTurnBoundary) {
        readMuted.delete(sessionId);
      } else if (pos === void 0 || mute.turn === null) {
        mute.streams += 1;
        if (mute.streams > MUTED_STREAMS_WITHOUT_TURN) readMuted.delete(sessionId);
      }
    }
    tapInFlight.add(sessionId);
    return tapActiveStream(
      sessionId,
      next(),
      queue,
      broadcast,
      (state) => {
        const turn = turnOfSession.get(sessionId)?.turn;
        if (state === "listening") {
          tapInFlight.delete(sessionId);
          if (focusedSessionId !== sessionId && activeVoiceSession !== sessionId) {
            reclaimFocusState(sessionId);
          }
        }
        if ((turnGen.get(sessionId) ?? 0) !== gen) return;
        setTurn(sessionId, state, turn);
      },
      vset.readAloud !== "off",
      () => readMuted.has(sessionId) || turnGen.get(sessionId) !== gen
    );
  });
  ctx.on("agent/turn-stopping", ({ agent }) => queue.noteTurnEnd(agent.id));
  ctx.on("agent/error", ({ agent }) => queue.noteTurnEnd(agent.id));
  ctx.on("agent/pre-step", ({ agent, turn, step }, next) => {
    const sid = agent.id;
    if (sid === activeVoiceSession || sid === focusedSessionId) turnOfSession.set(sid, { turn, step });
    return next();
  });
  const base = BASE_PATH;
  ctx.effect(
    () => ctx.webServer.register({
      kind: "prefix",
      path: base,
      handler: (req, res) => {
        if (denyNonLoopback(req, res)) return;
        respondJson2(res, 200, {
          ok: true,
          name: "dsh-voice-mode",
          enabled: config.enabled,
          active: activeVoiceSession
        });
      }
    })
  );
  ctx.effect(
    () => ctx.webServer.register({
      kind: "exact",
      path: `${base}/config`,
      handler: (req, res) => {
        if (denyNonLoopback(req, res)) return;
        respondJson2(res, 200, {
          basePath: base,
          rate: currentRate(),
          voice: currentVoice(),
          senseVoice: vset.senseVoice,
          interruptLevel: currentInterrupt(),
          silenceMs: vset.silenceMs,
          idleTimeoutMinutes: vset.idleTimeoutMinutes,
          modelHost: vset.modelHost,
          autoSend: vset.autoSend,
          autoResume: vset.autoResume,
          mode: vset.mode,
          readAloud: vset.readAloud,
          showCaptions: vset.showCaptions,
          showStatusBar: vset.showStatusBar,
          bargeInMode: vset.bargeInMode,
          echoGateDb: vset.echoGateDb,
          shortcut: vset.shortcut,
          composerShortcut: vset.composerShortcut,
          wakeWord: vset.wakeWord,
          toolBeep: vset.toolBeep,
          captionFontSize: vset.captionFontSize,
          captionMaxWidth: vset.captionMaxWidth,
          backchannelYield: vset.backchannelYield,
          bargeInSalvage: vset.bargeInSalvage,
          llmCorrect: vset.llmCorrect,
          yieldMs: vset.yieldMs,
          senseITN: vset.senseITN,
          cacheDir: config.cacheDir,
          ttsEngine: currentEngine(),
          audioMime: queue.mime,
          allowLan: config.allowLan
        });
      }
    })
  );
  ctx.effect(
    () => ctx.webServer.register({
      kind: "exact",
      path: `${base}/preview`,
      handler: (req, res) => {
        if (denyNonLoopback(req, res)) return;
        if (req.method !== "POST") {
          res.setHeader("allow", "POST");
          respondJson2(res, 405, { code: "method_not_allowed", error: "POST required" });
          return;
        }
        if (denyCrossOrigin(req, res)) return;
        if (!config.enabled) {
          respondJson2(res, 403, { code: "voice_disabled", error: "voice mode disabled" });
          return;
        }
        collectBody(req, res, MAX_JSON_BODY, async (body) => {
          let voice = "";
          let rate;
          try {
            const parsed = JSON.parse(body || "{}");
            voice = String(parsed.voice ?? "").trim();
            if (typeof parsed.rate === "number" && Number.isFinite(parsed.rate)) {
              rate = Math.min(2, Math.max(0.5, parsed.rate));
            }
          } catch {
          }
          if (voice.length > 128) {
            respondJson2(res, 400, { code: "bad_request", error: "voice too long" });
            return;
          }
          if (!voice) {
            respondJson2(res, 400, { code: "bad_request", error: "voice required" });
            return;
          }
          if (!limiter.hit(`preview:${clientKey(req)}`, 20, 6e4)) {
            respondJson2(res, 429, { code: "rate_limited", error: "rate limited" });
            return;
          }
          const sample = previewSample(currentEngine(), voice);
          let buf;
          try {
            buf = await queue.synthesize(sample, { voice, rate });
          } catch (e) {
            const errMsg = e instanceof Error ? e.message : String(e);
            const category = classifyPreviewError(errMsg);
            const engineName = currentEngine();
            const engineStatus = queue.status();
            const sampleLen = sample.length;
            console.warn(
              `[dsh-voice-mode] preview synthesis failed: category=${category} engine=${engineName} engineReady=${engineStatus.ready} sampleLen=${sampleLen} attempt=1 voice=${voice} err=${errMsg}`
            );
            respondJson2(res, 502, { code: PREVIEW_ERROR_CODES[category], error: `preview failed (${category})` });
            return;
          }
          res.writeHead(200, { "content-type": queue.mime, "cache-control": "no-store" });
          res.end(buf);
        });
      }
    })
  );
  let saySeq = 0;
  let sayChain = Promise.resolve();
  const sayText = (text, voice, rate) => {
    const sentences = text.split(/(?<=[.!?…;。])\s+/u).map((s) => s.trim()).filter(Boolean);
    const chunks = sentences.length > 0 ? sentences : [text];
    const run = async () => {
      for (const sentence of chunks) {
        try {
          const buf = await queue.synthesize(sentence, { voice, rate });
          const mime = queue.mime;
          const seq = ++saySeq;
          broadcast("audio", {
            sessionId: "__say__",
            sentenceId: seq,
            chunkId: 0,
            final: false,
            audio: buf.toString("base64"),
            mime,
            say: true
          });
          broadcast("audio", {
            sessionId: "__say__",
            sentenceId: seq,
            chunkId: 1,
            final: true,
            text: sentence,
            audio: "",
            mime,
            say: true
          });
        } catch (e) {
          console.warn(`[dsh-voice-mode] say failed: ${String(e)}`);
        }
      }
    };
    sayChain = sayChain.then(run, run);
  };
  ctx.effect(
    () => ctx.webServer.register({
      kind: "exact",
      path: `${base}/say`,
      handler: (req, res) => {
        if (denyNonLoopback(req, res)) return;
        if (req.method !== "POST") {
          res.setHeader("allow", "POST");
          respondJson2(res, 405, { code: "method_not_allowed", error: "POST required" });
          return;
        }
        if (denyCrossOrigin(req, res)) return;
        if (!config.enabled) {
          respondJson2(res, 403, { code: "voice_disabled", error: "voice mode disabled" });
          return;
        }
        collectBody(req, res, MAX_JSON_BODY, (body) => {
          let text = "";
          let voice = "";
          let rate;
          try {
            const parsed = JSON.parse(body || "{}");
            text = String(parsed.text ?? "").trim();
            voice = String(parsed.voice ?? "").trim();
            if (typeof parsed.rate === "number" && Number.isFinite(parsed.rate)) {
              rate = Math.min(2, Math.max(0.5, parsed.rate));
            }
          } catch {
          }
          if (!text) {
            respondJson2(res, 400, { code: "bad_request", error: "text required" });
            return;
          }
          if (text.length > 2e3) {
            respondJson2(res, 400, { code: "bad_request", error: "text too long" });
            return;
          }
          if (!limiter.hit(`say:${clientKey(req)}`, 30, 6e4)) {
            respondJson2(res, 429, { code: "rate_limited", error: "rate limited" });
            return;
          }
          const sentences = text.split(/(?<=[.!?…;。])\s+/u).map((s) => s.trim()).filter(Boolean);
          const chunks = sentences.length > 0 ? sentences : [text];
          respondJson2(res, 200, { ok: true, accepted: true, sentences: chunks.length });
          sayText(text, voice || void 0, rate);
        });
      }
    })
  );
  let sayToolState = "pending";
  ctx.effect(() => {
    try {
      ctx.tools.register(makeSayTool({ say: sayText }));
      sayToolState = "ok";
    } catch (e) {
      sayToolState = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
      console.warn(`[dsh-voice-mode-ru] say tool not registered: ${sayToolState}`);
    }
    return () => {
    };
  });
  ctx.effect(
    () => ctx.webServer.register({
      kind: "exact",
      path: `${base}/correct`,
      handler: (req, res) => {
        if (denyNonLoopback(req, res)) return;
        if (denyCrossOrigin(req, res)) return;
        collectBody(req, res, MAX_JSON_BODY, (body) => {
          let text = "";
          let lang = "zh";
          try {
            const parsed = JSON.parse(body || "{}");
            text = typeof parsed.text === "string" ? parsed.text : "";
            lang = normalizePromptLang(parsed.lang);
          } catch {
          }
          const trimmed = text.trim();
          if (!trimmed) {
            respondJson2(res, 200, { text });
            return;
          }
          if (!config.enabled) {
            respondJson2(res, 403, { code: "voice_disabled", error: "voice mode disabled", text: trimmed });
            return;
          }
          if (!limiter.hit(`correct:${clientKey(req)}`, 10, 6e4)) {
            respondJson2(res, 429, { code: "rate_limited", error: "rate limited", text: trimmed });
            return;
          }
          void correctText(ctx, trimmed, lang).then((corrected) => {
            respondJson2(res, 200, { text: corrected });
          }).catch(() => respondJson2(res, 200, { text: trimmed }));
        });
      }
    })
  );
  ctx.effect(
    () => ctx.webServer.register({
      kind: "exact",
      path: `${base}/toggle`,
      handler: (req, res) => {
        if (denyNonLoopback(req, res)) return;
        if (denyCrossOrigin(req, res)) return;
        collectBody(req, res, MAX_JSON_BODY, async (body) => {
          let sessionId;
          let on;
          let tabId;
          let langRaw;
          try {
            const parsed = JSON.parse(body || "{}");
            langRaw = parsed.lang;
            sessionId = parsed.sessionId;
            on = parsed.on;
            tabId = typeof parsed.tabId === "string" && parsed.tabId.length <= 64 ? parsed.tabId : void 0;
          } catch {
          }
          if (!sessionId) {
            respondJson2(res, 400, { code: "bad_request", error: "sessionId required" });
            return;
          }
          if (on === void 0) {
            respondJson2(res, 400, { code: "bad_request", error: "on required" });
            return;
          }
          if (typeof on !== "boolean") {
            respondJson2(res, 400, { code: "bad_request", error: "invalid on" });
            return;
          }
          if (on === true && !limiter.hit(`toggle:${sessionId}`, 2, 2e3)) {
            respondJson2(res, 429, { code: "rate_limited", error: "rate limited" });
            return;
          }
          if (on === true) {
            if (!config.enabled) {
              respondJson2(res, 403, { code: "voice_disabled", error: "voice mode disabled" });
              return;
            }
            if (sessions && !sessions.get(sessionId)) {
              respondJson2(res, 403, { code: "unknown_session", error: "unknown session" });
              return;
            }
            await asr.warmupSense();
            asr.reset(sessionId);
            queue.cancel(sessionId);
            const previous = activeVoiceSession;
            activeVoiceSession = sessionId;
            activeVoiceLang = normalizePromptLang(langRaw);
            activeTabId = tabId ?? null;
            if (ownerYieldTimer) {
              clearTimeout(ownerYieldTimer);
              ownerYieldTimer = null;
            }
            if (previous && previous !== sessionId) {
              queue.cancel(previous);
              asr.reset(previous);
              clearTurnState(previous);
            }
            broadcast("mode", { active: activeVoiceSession, ownerTabId: activeTabId });
          } else {
            if (activeVoiceSession === sessionId) {
              activeVoiceSession = null;
              activeTabId = null;
              if (ownerYieldTimer) {
                clearTimeout(ownerYieldTimer);
                ownerYieldTimer = null;
              }
              queue.cancel(sessionId);
              asr.reset(sessionId);
              clearTurnState(sessionId);
              broadcast("mode", { active: null, ownerTabId: null });
            }
          }
          respondJson2(res, 200, { active: activeVoiceSession });
        });
      }
    })
  );
  ctx.effect(
    () => ctx.webServer.register({
      kind: "exact",
      path: `${base}/models/status`,
      handler: (req, res) => {
        if (denyNonLoopback(req, res)) return;
        respondJson2(res, 200, { ...asr.modelStatus(), tts: queue.status(), sayTool: sayToolState });
      }
    })
  );
  ctx.effect(
    () => ctx.webServer.register({
      kind: "exact",
      path: `${base}/models/retry`,
      handler: (req, res) => {
        if (denyNonLoopback(req, res)) return;
        if (req.method !== "POST") {
          res.setHeader("allow", "POST");
          respondJson2(res, 405, { code: "method_not_allowed", error: "POST required" });
          return;
        }
        if (denyCrossOrigin(req, res)) return;
        if (!config.enabled) {
          respondJson2(res, 403, { code: "voice_disabled", error: "voice mode disabled" });
          return;
        }
        collectBody(req, res, MAX_JSON_BODY, (body) => {
          let kind = "asr";
          try {
            const p = JSON.parse(body || "{}");
            if (p.kind === void 0) {
            } else if (p.kind === "vad" || p.kind === "sense" || p.kind === "asr") {
              kind = p.kind;
            } else {
              respondJson2(res, 400, { code: "bad_request", error: "invalid kind" });
              return;
            }
          } catch {
            respondJson2(res, 400, { code: "bad_request", error: "invalid json" });
            return;
          }
          void asr.retryModel(kind).then((done) => {
            respondJson2(res, 200, { ok: done, kind });
          }).catch((e) => {
            console.warn(`[dsh-voice-mode] model retry failed: ${String(e)}`);
            respondJson2(res, 500, { code: "internal", error: "model retry failed" });
          });
        });
      }
    })
  );
  ctx.effect(
    () => ctx.webServer.register({
      kind: "exact",
      path: `${base}/models/clean`,
      handler: (req, res) => {
        if (denyNonLoopback(req, res)) return;
        if (denyCrossOrigin(req, res)) return;
        if (!config.enabled) {
          respondJson2(res, 403, { code: "voice_disabled", error: "voice mode disabled" });
          return;
        }
        collectBody(req, res, MAX_JSON_BODY, (body) => {
          let engine = "vits";
          try {
            const p = JSON.parse(body || "{}");
            if (p.engine === "kokoro" || p.engine === "vits") engine = p.engine;
            else {
              respondJson2(res, 400, { code: "bad_request", error: "invalid engine" });
              return;
            }
          } catch {
            respondJson2(res, 400, { code: "bad_request", error: "invalid json" });
            return;
          }
          const dirs = engine === "kokoro" ? [kokoroModelDir("int8"), kokoroModelDir("fp32")].map((d) => join5(config.cacheDir, d)) : [join5(config.cacheDir, TTS_MODEL_REPO)];
          void Promise.all(dirs.map((dir) => rm(dir, { recursive: true, force: true }))).then(() => {
            if (engineKind === engine) {
              queue.setEngine(makeEngine(engine));
              queue.updateVoice(vset.voice, vset.rate);
            }
            respondJson2(res, 200, { ok: true, engine });
          }).catch((e) => {
            console.warn(`[dsh-voice-mode] models/download failed: ${String(e)}`);
            respondJson2(res, 500, { code: "internal", error: "internal error" });
          });
        });
      }
    })
  );
  ctx.effect(
    () => ctx.webServer.register({
      kind: "exact",
      path: `${base}/models/download`,
      handler: (req, res) => {
        if (denyNonLoopback(req, res)) return;
        if (denyCrossOrigin(req, res)) return;
        if (!config.enabled) {
          respondJson2(res, 403, { code: "voice_disabled", error: "voice mode disabled" });
          return;
        }
        collectBody(req, res, MAX_JSON_BODY, (body) => {
          let engine = "vits";
          try {
            const p = JSON.parse(body || "{}");
            if (p.engine === "kokoro" || p.engine === "vits") engine = p.engine;
            else {
              respondJson2(res, 400, { code: "bad_request", error: "invalid engine" });
              return;
            }
          } catch {
            respondJson2(res, 400, { code: "bad_request", error: "invalid json" });
            return;
          }
          if (engineKind !== engine) {
            respondJson2(res, 400, { code: "engine_not_active", error: "engine not active" });
            return;
          }
          void queue.prepare().then(() => respondJson2(res, 200, { ok: true, engine })).catch((e) => {
            console.warn(`[dsh-voice-mode] model download failed: ${String(e)}`);
            respondJson2(res, 502, { code: "model_download_failed", error: "model download failed" });
          });
        });
      }
    })
  );
  ctx.effect(
    () => ctx.webServer.register({
      kind: "exact",
      path: `${base}/voices`,
      handler: async (req, res) => {
        if (denyNonLoopback(req, res)) return;
        if (denyCrossOrigin(req, res)) return;
        try {
          const voices = await listEdgeVoices();
          respondJson2(res, 200, { voices });
        } catch (e) {
          console.warn(`[dsh-voice-mode] listing Edge voices failed: ${String(e)}`);
          respondJson2(res, 502, { code: "internal", error: "voice list unavailable" });
        }
      }
    })
  );
  ctx.effect(
    () => ctx.webServer.register({
      kind: "exact",
      path: `${base}/asr`,
      handler: (req, res) => {
        if (denyNonLoopback(req, res)) return;
        if (denyCrossOrigin(req, res)) return;
        if (req.method !== "POST") {
          res.setHeader("allow", "POST");
          respondJson2(res, 405, { code: "method_not_allowed", error: "POST required" });
          return;
        }
        let sid = "";
        try {
          const url = new URL(req.url ?? "/", "http://localhost");
          sid = url.searchParams.get("sessionId") ?? "";
        } catch {
        }
        if (!limiter.hit(`asr:${sid || "unknown"}`, 60, 1e3)) {
          respondJson2(res, 429, { code: "rate_limited", error: "rate limited" });
          return;
        }
        if (sid && sid === activeVoiceSession) {
          try {
            const url = new URL(req.url ?? "/", "http://localhost");
            setTurn(sid, url.searchParams.get("final") === "1" ? "finalizing" : "listening");
          } catch {
          }
        }
        handleAsrRequest(asr, activeVoiceSession, req, res);
      }
    })
  );
  ctx.effect(
    () => ctx.webServer.register({
      kind: "exact",
      path: `${base}/cancel`,
      handler: (req, res) => {
        if (denyNonLoopback(req, res)) return;
        if (denyCrossOrigin(req, res)) return;
        collectBody(req, res, MAX_JSON_BODY, (body) => {
          let sessionId;
          let keepAsr = false;
          let isInterrupt = false;
          try {
            const parsed = JSON.parse(body || "{}");
            sessionId = parsed.sessionId;
            keepAsr = parsed.keepAsr === true;
            isInterrupt = parsed.reason === "interrupt";
          } catch {
          }
          if (sessionId && (sessionId === activeVoiceSession || sessionId === focusedSessionId)) {
            if (!limiter.hit(`cancel:${sessionId}`, 2, 1e3)) {
              respondJson2(res, 429, { code: "rate_limited", error: "rate limited" });
              return;
            }
            queue.cancel(sessionId);
            if (isInterrupt) queue.abandonTurn(sessionId);
            readMuted.set(sessionId, { turn: turnOfSession.get(sessionId)?.turn ?? null, streams: 0 });
            if (!keepAsr && sessionId === activeVoiceSession) asr.reset(sessionId);
          }
          respondJson2(res, 200, { ok: true });
        });
      }
    })
  );
  ctx.effect(
    () => ctx.webServer.register({
      kind: "exact",
      path: `${base}/settings`,
      handler: (req, res) => {
        if (denyNonLoopback(req, res)) return;
        if (req.method === "GET") {
          respondJson2(res, 200, { managedBy: settingsScopeRef ? "dsh" : "plugin", value: vset });
          return;
        }
        if (req.method !== "POST") {
          res.setHeader("allow", "GET, POST");
          respondJson2(res, 405, { code: "method_not_allowed", error: "GET or POST required" });
          return;
        }
        if (denyCrossOrigin(req, res)) return;
        if (settingsScopeRef) {
          respondJson2(res, 409, { code: "settings_managed", error: "settings are managed by dsh on this host" });
          return;
        }
        collectBody(req, res, MAX_JSON_BODY, async (body) => {
          let patch;
          try {
            patch = JSON.parse(body || "{}");
          } catch {
            respondJson2(res, 400, { code: "bad_request", error: "invalid JSON" });
            return;
          }
          if (patch === null || typeof patch !== "object" || Array.isArray(patch)) {
            respondJson2(res, 400, { code: "bad_request", error: "body must be a JSON object" });
            return;
          }
          const unknown = unknownKeys(patch, SETTING_KEYS);
          if (unknown.length > 0) {
            respondJson2(res, 400, { code: "bad_request", error: `unknown settings: ${unknown.slice(0, 5).join(", ")}` });
            return;
          }
          try {
            const next = await persistSettings(pickKnownKeys(patch, SETTING_KEYS));
            respondJson2(res, 200, { ok: true, managedBy: "plugin", value: next });
          } catch (e) {
            const isValidation = e instanceof Error && e.name === "ValidationError";
            console.warn(`[dsh-voice-mode] settings update failed: ${String(e)}`);
            respondJson2(res, isValidation ? 400 : 500, {
              code: isValidation ? "bad_request" : "internal",
              error: isValidation ? `invalid value: ${e.message.slice(0, 200)}` : "settings update failed"
            });
          }
        });
      }
    })
  );
  ctx.effect(
    () => ctx.webServer.register({
      kind: "exact",
      path: `${base}/mode`,
      handler: (req, res) => {
        if (denyNonLoopback(req, res)) return;
        if (denyCrossOrigin(req, res)) return;
        collectBody(req, res, MAX_JSON_BODY, (body) => {
          let mode;
          try {
            const parsed = JSON.parse(body || "{}");
            mode = parsed.mode === "toggle" || parsed.mode === "hold" || parsed.mode === "once" ? parsed.mode : void 0;
          } catch {
          }
          if (!mode) {
            respondJson2(res, 400, { code: "bad_request", error: "mode must be toggle, hold or once" });
            return;
          }
          const persistMode = settingsScopeRef ? settingsScopeRef.update({ mode }) : persistSettings({ mode });
          void persistMode.then(() => {
            respondJson2(res, 200, { ok: true, mode });
          }).catch((e) => {
            console.warn(`[dsh-voice-mode] mode update failed: ${String(e)}`);
            respondJson2(res, 500, { code: "internal", error: "mode update failed" });
          });
        });
      }
    })
  );
  ctx.effect(
    () => ctx.webServer.register({
      kind: "exact",
      path: `${base}/focus`,
      handler: (req, res) => {
        if (denyNonLoopback(req, res)) return;
        if (denyCrossOrigin(req, res)) return;
        collectBody(req, res, MAX_JSON_BODY, (body) => {
          let parsed = {};
          try {
            parsed = JSON.parse(body || "{}");
          } catch {
          }
          const { sessionId, tabId } = parsed;
          const prevFocus = focusedSessionId;
          const validTab = typeof tabId === "string" && tabId.length > 0 && tabId.length <= 64;
          const validSid = typeof sessionId === "string" && sessionId.length > 0 && sessionId.length <= 128;
          if (!validTab || !(validSid || sessionId === null)) {
            respondJson2(res, 400, { code: "bad_request", error: "expected {sessionId: string|null, tabId: string}" });
            return;
          }
          if (validSid) {
            if (sessions && !sessions.get(sessionId)) {
              respondJson2(res, 403, { code: "unknown_session", error: "unknown session" });
              return;
            }
            focusedSessionId = sessionId;
            focusedTabId = tabId;
          } else if (tabId === focusedTabId) {
            focusedSessionId = null;
            focusedTabId = null;
          }
          if (prevFocus !== null && prevFocus !== focusedSessionId) stopFocusReading(prevFocus);
          if (prevFocus !== null && prevFocus !== focusedSessionId) reclaimFocusState(prevFocus);
          broadcast("focus", { sessionId: focusedSessionId, tabId: focusedTabId });
          respondJson2(res, 200, { ok: true, sessionId: focusedSessionId, tabId: focusedTabId });
        });
      }
    })
  );
  ctx.effect(
    () => ctx.webServer.register({
      kind: "exact",
      path: `${base}/stream`,
      handler: (req, res) => {
        if (denyNonLoopback(req, res)) return;
        if (denyCrossOrigin(req, res)) return;
        let tabId = null;
        try {
          const u = new URL(req.url ?? "/", "http://localhost");
          tabId = u.searchParams.get("tabId");
        } catch {
        }
        if (tabId !== null && tabId.length > 64) tabId = null;
        if (tabId !== null) {
          for (const old of sseClients) {
            if (old.tabId !== tabId) continue;
            sseClients.delete(old);
            try {
              old.send("replaced", {});
            } catch {
            }
            old.end();
          }
        }
        if (sseClients.size >= 8) {
          respondJson2(res, 429, { code: "too_many_streams", error: "too many streams" });
          return;
        }
        res.writeHead(200, {
          "content-type": "text/event-stream; charset=utf-8",
          "cache-control": "no-cache, no-transform",
          connection: "keep-alive"
        });
        res.write("retry: 3000\n\n");
        const send = (event, payload) => {
          res.write(`event: ${event}
data: ${JSON.stringify(payload)}

`);
          if (res.writableLength > SSE_MAX_BUFFERED) {
            console.warn(`[dsh-voice-mode] SSE client too slow (buffered ${res.writableLength}B); dropping connection`);
            res.destroy();
          }
        };
        const end = () => {
          try {
            res.end();
          } catch {
          }
        };
        const client = { tabId, send, end };
        sseClients.add(client);
        if (tabId !== null) latestConnByTab.set(tabId, client);
        if (tabId !== null && tabId === activeTabId && ownerYieldTimer) {
          clearTimeout(ownerYieldTimer);
          ownerYieldTimer = null;
        }
        send("mode", { active: activeVoiceSession, ownerTabId: activeTabId });
        send("focus", { sessionId: focusedSessionId, tabId: focusedTabId });
        send("display", displayOf(vset));
        const heartbeat = setInterval(() => {
          try {
            res.write(": hb\n");
          } catch {
            cleanup();
          }
        }, 25e3);
        let cleaned = false;
        const cleanup = () => {
          if (cleaned) return;
          cleaned = true;
          clearInterval(heartbeat);
          sseClients.delete(client);
          if (tabId !== null && latestConnByTab.get(tabId) === client) {
            latestConnByTab.delete(tabId);
            if (tabId === focusedTabId) {
              const prevFocus = focusedSessionId;
              focusedSessionId = null;
              focusedTabId = null;
              reclaimFocusState(prevFocus);
              broadcast("focus", { sessionId: null, tabId: null });
            }
            if (tabId === activeTabId) {
              if (ownerYieldTimer) clearTimeout(ownerYieldTimer);
              ownerYieldTimer = setTimeout(() => yieldActiveSession(activeVoiceSession), 8e3);
            }
          }
        };
        req.on("close", cleanup);
        res.on("close", cleanup);
      }
    })
  );
}
var MAX_JSON_BODY = 16 * 1024;
function collectBody(req, res, maxBytes, onBody) {
  const chunks = [];
  let received = 0;
  let tooLarge = false;
  req.on("data", (c) => {
    if (tooLarge) return;
    received += c.length;
    if (received > maxBytes) {
      tooLarge = true;
      respondJson2(res, 413, { code: "payload_too_large", error: "request body too large" });
      req.pause();
      res.once("finish", () => req.destroy());
      return;
    }
    chunks.push(c);
  });
  req.on("end", () => {
    if (tooLarge) return;
    const body = Buffer.concat(chunks).toString("utf8");
    try {
      const r = onBody(body);
      if (r && typeof r.then === "function") r.catch(() => {
      });
    } catch {
    }
  });
  req.on("error", () => {
  });
}
async function* tapActiveStream(sessionId, inner, queue, broadcast, onTurn, speak = true, muted = () => false) {
  const segmenter = new SentenceSegmenter();
  let firstTokenBroadcast = false;
  let firstSentenceBroadcast = false;
  let flushed = false;
  let finishReason = null;
  const flushOnce = () => {
    if (flushed) return;
    flushed = true;
    for (const s of segmenter.flush()) {
      if (speak && !muted()) queue.enqueue(sessionId, s);
    }
  };
  try {
    for await (const chunk of inner) {
      if (chunk.type === "text-delta" && chunk.text) {
        if (!firstTokenBroadcast) {
          firstTokenBroadcast = true;
          broadcast("latency", { sessionId, stage: "first-llm-token" });
          onTurn("agent-speaking");
        }
        for (const s of segmenter.feed(chunk.text)) {
          if (!firstSentenceBroadcast) {
            firstSentenceBroadcast = true;
            broadcast("latency", { sessionId, stage: "first-sentence-text" });
          }
          if (speak && !muted()) queue.enqueue(sessionId, s);
        }
      }
      if (chunk.type === "tool-call-delta" && chunk.name) {
        broadcast("tool", { sessionId, name: chunk.name });
      }
      if (chunk.type === "finish") {
        finishReason = chunk.reason;
      }
      yield chunk;
    }
  } finally {
    const aborted = finishReason !== null && typeof finishReason === "object" && finishReason.kind === "aborted";
    if (!aborted) flushOnce();
    onTurn("listening");
  }
}
export {
  Config,
  VoiceSettingsSchema,
  apply,
  createVoiceSettingsSchema,
  inject,
  name
};
