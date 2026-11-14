"use strict";
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));

// src/ru-tts-worker.ts
var import_sherpa_onnx_node = __toESM(require("sherpa-onnx-node"), 1);

// src/segmenter.ts
function sanitizeForTts(text) {
  return String(text).replace(/[*_#|^=+~`]/g, " ").replace(/\s{2,}/g, " ").replace(/([\u3400-\u9fff])\s+(?=[\u3400-\u9fff])/g, "$1").trim();
}

// src/ru-tts-worker.ts
var { OfflineTts, GenerationConfig } = import_sherpa_onnx_node.default;
var send = (msg) => {
  if (typeof process.send === "function") process.send(msg);
};
var CHAR_MAP = /* @__PURE__ */ new Map([
  ["\u026D", "l"],
  ["\u0275", "o"],
  ["\u0291", "\u0255"],
  ["\u0261", "\u0261"],
  ["g", "\u0261"]
]);
var ZWJ = "\u200D";
var MAX_PHONEMES_PER_SENTENCE = 400;
var PAUSE_MS = /* @__PURE__ */ new Map([
  [",", 150],
  [";", 220],
  [":", 220],
  ["\u2014", 220],
  ["\u2013", 220],
  [".", 350],
  ["!", 350],
  ["?", 350],
  ["\u2026", 450],
  ["\u3002", 350]
]);
var TAIL_PAUSE_MS = 300;
var PAUSE_SCALE = (() => {
  const raw = Number(process.env.DSHVM_RU_PAUSE);
  return Number.isFinite(raw) && raw > 0 ? Math.min(3, raw) : 1;
})();
var tts = null;
var espeakFactory = null;
var sampleRate = 24e3;
function splitByPunctuation(text) {
  const src = String(text ?? "").replace(/\s+/g, " ").trim();
  if (!src) return [];
  const items = [];
  let buf = "";
  for (const ch of src) {
    buf += ch;
    const pause = PAUSE_MS.get(ch);
    if (pause === void 0) continue;
    const piece = buf.trim();
    buf = "";
    if (piece) items.push({ text: piece, pauseMs: Math.round(pause * PAUSE_SCALE) });
  }
  const tail = buf.trim();
  if (tail) items.push({ text: tail, pauseMs: Math.round(TAIL_PAUSE_MS * PAUSE_SCALE) });
  const merged = [];
  for (const item of items) {
    const prev = merged[merged.length - 1];
    if (prev && [...item.text].length < 12) {
      prev.text = `${prev.text} ${item.text}`;
      prev.pauseMs = Math.max(prev.pauseMs, item.pauseMs);
      continue;
    }
    merged.push({ ...item });
  }
  const limit = MAX_PHONEMES_PER_SENTENCE / 1.15;
  const capped = [];
  for (const item of merged) {
    if ([...item.text].length <= limit) {
      capped.push(item);
      continue;
    }
    let current = "";
    for (const word of item.text.split(" ")) {
      const candidate = current ? `${current} ${word}` : word;
      if ([...candidate].length > limit && current) {
        capped.push({ text: current, pauseMs: TAIL_PAUSE_MS });
        current = word;
      } else {
        current = candidate;
      }
    }
    if (current) capped.push({ text: current, pauseMs: item.pauseMs });
  }
  return capped.length > 0 ? capped : [{ text: src, pauseMs: TAIL_PAUSE_MS }];
}
async function phonemize(text) {
  if (!espeakFactory) throw new Error("espeak not initialized");
  const espeak = await espeakFactory({
    arguments: ["-q", "--ipa=3", "-v", "ru", text, "--phonout", "phon.txt"]
  });
  const raw = espeak.FS.readFile("phon.txt", { encoding: "utf8" });
  const ipa = raw.replace(/\([a-z]{2,3}(?:-[a-z0-9]+)?\)/gi, " ");
  let out = "";
  for (const ch of ipa) {
    if (ch === ZWJ) continue;
    const mapped = CHAR_MAP.get(ch);
    out += mapped !== void 0 ? mapped : ch;
  }
  return out.replace(/\s+/g, " ").trim();
}
function toCodepoints(ipa) {
  const cps = [];
  for (const ch of ipa) {
    if (ch === " ") continue;
    cps.push(ch.codePointAt(0) ?? 0);
  }
  return cps;
}
function trimSilence(samples, threshold = 6e-3) {
  let start = 0;
  while (start < samples.length && Math.abs(samples[start]) < threshold) start++;
  let end = samples.length;
  while (end > start && Math.abs(samples[end - 1]) < threshold) end--;
  if (end - start <= 0) return samples;
  return start === 0 && end === samples.length ? samples : samples.subarray(start, end);
}
async function init(paths) {
  if (tts) return;
  if (/[^\x00-\xff]/.test(process.argv[1] ?? "")) process.argv[1] = "dsh-voice-mode-ru-worker";
  for (const key of Object.keys(process.env)) {
    const value = process.env[key];
    if (/[^\x00-\xff]/.test(key) || typeof value === "string" && /[^\x00-\xff]/.test(value)) delete process.env[key];
  }
  const factory = await import("espeak-ng");
  espeakFactory = factory.default;
  tts = new OfflineTts({
    model: {
      kokoro: {
        model: paths.model,
        voices: paths.voices,
        tokens: paths.tokens,
        lexicon: ""
      },
      // 4 线程实测 RTF≈0.89；语音输入同时跑 ASR 时可下调到 2。
      numThreads: Math.max(1, Math.min(8, Number(process.env.DSHVM_RU_THREADS) || 4)),
      debug: 0,
      provider: "cpu"
    },
    ruleFsts: "",
    maxNumSentences: 1
  });
  sampleRate = tts.sampleRate;
}
async function synth(text, sid, speed) {
  if (!tts) throw new Error("tts child not initialized");
  const clean = sanitizeForTts(String(text ?? ""));
  if (!clean.trim()) return new Float32Array(Math.floor(sampleRate * 0.2));
  const parts = [];
  for (const fragment of splitByPunctuation(clean)) {
    const cps = toCodepoints(await phonemize(fragment.text));
    if (cps.length === 0) continue;
    const gc = new GenerationConfig({ sid, speed, silenceScale: 0.2 });
    gc.phonemeCodepoints = [cps];
    const audio = tts.generate({ text: "", generationConfig: gc });
    if (audio.samples.length > 0) {
      parts.push({
        samples: trimSilence(audio.samples),
        silence: Math.floor(sampleRate * (fragment.pauseMs / 1e3))
      });
    }
  }
  if (parts.length === 0) return new Float32Array(Math.floor(sampleRate * 0.2));
  let total = 0;
  for (const part of parts) total += part.samples.length + part.silence;
  const out = new Float32Array(total);
  let offset = 0;
  for (const part of parts) {
    out.set(part.samples, offset);
    offset += part.samples.length + part.silence;
  }
  return out;
}
var reply = (id, payload) => {
  send({ id: typeof id === "number" ? id : 0, ...payload });
};
process.on("message", (msg) => {
  if (!msg || typeof msg !== "object") return;
  const m = msg;
  const id = m.id;
  if (m.type === "init") {
    init(m.paths ?? {}).then(() => reply(id, { ok: true, sampleRate })).catch((e) => reply(id, { ok: false, error: String(e instanceof Error ? e.message : e) }));
    return;
  }
  if (m.type === "synth") {
    const sid = Number(m.sid ?? 0);
    const speed = Number(m.speed ?? 1);
    synth(String(m.text ?? ""), sid, speed).then((samples) => {
      const buf = Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength);
      reply(id, { ok: true, sampleRate, samples: buf.toString("base64") });
    }).catch((e) => reply(id, { ok: false, error: String(e instanceof Error ? e.message : e) }));
    return;
  }
  if (m.type === "close") {
    try {
      tts?.free();
    } catch {
    }
    tts = null;
    reply(id, { ok: true });
    return;
  }
  reply(id, { ok: false, error: `unknown message type: ${String(m.type)}` });
});
process.on("uncaughtException", (e) => {
  console.error("[ru-tts-worker] uncaught:", e instanceof Error ? e.stack ?? e.message : e);
  process.exit(1);
});
