/**
 * fork 新增：俄语 Kokoro 合成子进程（child_process.fork，CJS + IPC）。
 *
 * 与上游 lib/tts-vits-worker.cjs 的差别：上游靠 sherpa-onnx 的 lexicon 前端出音素，
 * 而俄语没有现成 lexicon，且 sherpa 的 Kokoro 前端不含 espeak 音素化。这里改成：
 *
 *   文本 → espeak-ng(WASM, ru) → IPA → sherpa-onnx(phonemeCodepoints) → Float32 PCM
 *
 * 三个由实测得出的硬约束：
 *  1. espeak-ng 的 WASM 会把 process.argv[1] 当可执行文件名，任何非 latin1 字符都会
 *     触发断言崩溃 —— 本文件必须位于纯 ASCII 路径下（插件目录满足）。
 *  2. espeak 的 IPA 输出会丢掉标点，模型因此不会在标点处停顿 —— 停顿由本文件插入。
 *  3. style_dim[0] = 510 限制单句音素数，长文本按标点切段后再合成。
 *
 * 协议（与上游一致）：
 *   init { type:'init', id, kind:'kokoro', paths:{ model, voices, tokens } }
 *   { type:'synth', id, text, sid, speed } → { id, ok, sampleRate, samples(base64 f32) }
 *   { type:'close', id }                   → { id, ok }
 */
import sherpa_node from 'sherpa-onnx-node'
import { sanitizeForTts } from './segmenter'

interface NativeTts {
  generate(input: { text: string; generationConfig: unknown }): { samples: Float32Array; sampleRate: number }
  free(): void
  readonly sampleRate: number
}

const { OfflineTts, GenerationConfig } = sherpa_node as unknown as {
  OfflineTts: new (config: Record<string, unknown>) => NativeTts
  GenerationConfig: new (opts: Record<string, unknown>) => Record<string, unknown>
}

type EspeakModule = {
  FS: { readFile(path: string, opts?: { encoding?: string }): string }
}
type EspeakFactory = (opts: { arguments: string[] }) => Promise<EspeakModule>

const send = (msg: Record<string, unknown>): void => {
  if (typeof process.send === 'function') process.send(msg)
}

// espeak 输出的、模型词表里没有的 IPA 符号（按 178 槽词表实测）：
//   ɭ  — 俄语硬 «л»（голоса, булок）
//   ɵ  — ч/щ 之后的 «ё»（ещё, чёрный）
//   ʑ  — 浊化的 «щ»
const CHAR_MAP = new Map<string, string>([
  ['\u026d', 'l'],
  ['\u0275', 'o'],
  ['\u0291', '\u0255'],
  ['\u0261', '\u0261'],
  ['\u0067', '\u0261'],
])
const ZWJ = '\u200d' // espeak 的 IPA 连字符，不是词表符号
const MAX_PHONEMES_PER_SENTENCE = 400 // < style_dim[0] = 510

const PAUSE_MS = new Map<string, number>([
  [',', 150],
  [';', 220],
  [':', 220],
  ['\u2014', 220],
  ['\u2013', 220],
  ['.', 350],
  ['!', 350],
  ['?', 350],
  ['\u2026', 450],
  ['\u3002', 350],
])
const TAIL_PAUSE_MS = 300
const PAUSE_SCALE = ((): number => {
  const raw = Number(process.env.DSHVM_RU_PAUSE)
  return Number.isFinite(raw) && raw > 0 ? Math.min(3, raw) : 1
})()

let tts: NativeTts | null = null
let espeakFactory: EspeakFactory | null = null
let sampleRate = 24000

/** 按标点切段，并记住每个标点该配多长的停顿。过短的片段并入前一段，避免逐词调用模型。 */
function splitByPunctuation(text: string): Array<{ text: string; pauseMs: number }> {
  const src = String(text ?? '').replace(/\s+/g, ' ').trim()
  if (!src) return []
  const items: Array<{ text: string; pauseMs: number }> = []
  let buf = ''
  for (const ch of src) {
    buf += ch
    const pause = PAUSE_MS.get(ch)
    if (pause === undefined) continue
    const piece = buf.trim()
    buf = ''
    if (piece) items.push({ text: piece, pauseMs: Math.round(pause * PAUSE_SCALE) })
  }
  const tail = buf.trim()
  if (tail) items.push({ text: tail, pauseMs: Math.round(TAIL_PAUSE_MS * PAUSE_SCALE) })

  const merged: Array<{ text: string; pauseMs: number }> = []
  for (const item of items) {
    const prev = merged[merged.length - 1]
    if (prev && [...item.text].length < 12) {
      prev.text = `${prev.text} ${item.text}`
      prev.pauseMs = Math.max(prev.pauseMs, item.pauseMs)
      continue
    }
    merged.push({ ...item })
  }

  const limit = MAX_PHONEMES_PER_SENTENCE / 1.15
  const capped: Array<{ text: string; pauseMs: number }> = []
  for (const item of merged) {
    if ([...item.text].length <= limit) {
      capped.push(item)
      continue
    }
    let current = ''
    for (const word of item.text.split(' ')) {
      const candidate = current ? `${current} ${word}` : word
      if ([...candidate].length > limit && current) {
        capped.push({ text: current, pauseMs: TAIL_PAUSE_MS })
        current = word
      } else {
        current = candidate
      }
    }
    if (current) capped.push({ text: current, pauseMs: item.pauseMs })
  }
  return capped.length > 0 ? capped : [{ text: src, pauseMs: TAIL_PAUSE_MS }]
}

async function phonemize(text: string): Promise<string> {
  if (!espeakFactory) throw new Error('espeak not initialized')
  // espeak 只接受实例化时传入的参数，因此每段新建一个 32MB WASM 实例（几十毫秒）。
  const espeak = await espeakFactory({
    arguments: ['-q', '--ipa=3', '-v', 'ru', text, '--phonout', 'phon.txt'],
  })
  const raw = espeak.FS.readFile('phon.txt', { encoding: 'utf8' })
  // espeak 遇到 латиницу переключает язык и вставляет служебные пометки вида
  // "(en)…(ru)". Все их символы есть в словаре модели, поэтому она озвучивала
  // их как «ен» и «ру» (призвук «ру» в конце латинских слов). Вырезаем пометки.
  const ipa = raw.replace(/\([a-z]{2,3}(?:-[a-z0-9]+)?\)/gi, ' ')
  let out = ''
  for (const ch of ipa) {
    if (ch === ZWJ) continue
    const mapped = CHAR_MAP.get(ch)
    out += mapped !== undefined ? mapped : ch
  }
  return out.replace(/\s+/g, ' ').trim()
}

function toCodepoints(ipa: string): number[] {
  const cps: number[] = []
  for (const ch of ipa) {
    if (ch === ' ') continue
    cps.push(ch.codePointAt(0) ?? 0)
  }
  return cps
}

/** 模型每段输出前后都带静音，留着会与我们的标点停顿叠加，两端各裁一次。 */
function trimSilence(samples: Float32Array, threshold = 0.006): Float32Array {
  let start = 0
  while (start < samples.length && Math.abs(samples[start]) < threshold) start++
  let end = samples.length
  while (end > start && Math.abs(samples[end - 1]) < threshold) end--
  if (end - start <= 0) return samples
  return start === 0 && end === samples.length ? samples : samples.subarray(start, end)
}

async function init(paths: Record<string, string>): Promise<void> {
  if (tts) return
  // espeak-ng 的 WASM 把 process.argv[1] 当作可执行文件名，并对它做 latin1 断言：
  // 路径里只要有 кириллица（例如 «Документы»），初始化就会 Aborted(Assertion failed)
  // 直接砸掉整个子进程。这里的断言发生在我们 import 之前，所以先把它换成 ASCII 常量。
  // 顺便清掉环境变量里的非 latin1 值（同一断言路径）。
  if (/[^\x00-\xff]/.test(process.argv[1] ?? '')) process.argv[1] = 'dsh-voice-mode-ru-worker'
  for (const key of Object.keys(process.env)) {
    const value = process.env[key]
    if (/[^\x00-\xff]/.test(key) || (typeof value === 'string' && /[^\x00-\xff]/.test(value))) delete process.env[key]
  }
  const factory = (await import('espeak-ng')) as unknown as { default: EspeakFactory }
  espeakFactory = factory.default

  tts = new OfflineTts({
    model: {
      kokoro: {
        model: paths.model,
        voices: paths.voices,
        tokens: paths.tokens,
        lexicon: '',
      },
      // 4 线程实测 RTF≈0.89；语音输入同时跑 ASR 时可下调到 2。
      numThreads: Math.max(1, Math.min(8, Number(process.env.DSHVM_RU_THREADS) || 4)),
      debug: 0,
      provider: 'cpu',
    },
    ruleFsts: '',
    maxNumSentences: 1,
  })
  sampleRate = tts.sampleRate
}

async function synth(text: string, sid: number, speed: number): Promise<Float32Array> {
  if (!tts) throw new Error('tts child not initialized')
  const clean = sanitizeForTts(String(text ?? ''))
  if (!clean.trim()) return new Float32Array(Math.floor(sampleRate * 0.2))

  const parts: Array<{ samples: Float32Array; silence: number }> = []
  for (const fragment of splitByPunctuation(clean)) {
    const cps = toCodepoints(await phonemize(fragment.text))
    if (cps.length === 0) continue
    const gc = new GenerationConfig({ sid, speed, silenceScale: 0.2 })
    gc.phonemeCodepoints = [cps]
    const audio = tts.generate({ text: '', generationConfig: gc })
    if (audio.samples.length > 0) {
      parts.push({
        samples: trimSilence(audio.samples),
        silence: Math.floor(sampleRate * (fragment.pauseMs / 1000)),
      })
    }
  }
  if (parts.length === 0) return new Float32Array(Math.floor(sampleRate * 0.2))

  let total = 0
  for (const part of parts) total += part.samples.length + part.silence
  const out = new Float32Array(total)
  let offset = 0
  for (const part of parts) {
    out.set(part.samples, offset)
    offset += part.samples.length + part.silence
  }
  return out
}

const reply = (id: unknown, payload: Record<string, unknown>): void => {
  send({ id: typeof id === 'number' ? id : 0, ...payload })
}

process.on('message', (msg: unknown) => {
  if (!msg || typeof msg !== 'object') return
  const m = msg as { id?: unknown; type?: unknown; text?: unknown; sid?: unknown; speed?: unknown; paths?: unknown }
  const id = m.id
  if (m.type === 'init') {
    init((m.paths ?? {}) as Record<string, string>)
      .then(() => reply(id, { ok: true, sampleRate }))
      .catch((e: unknown) => reply(id, { ok: false, error: String(e instanceof Error ? e.message : e) }))
    return
  }
  if (m.type === 'synth') {
    const sid = Number(m.sid ?? 0)
    const speed = Number(m.speed ?? 1)
    synth(String(m.text ?? ''), sid, speed)
      .then((samples) => {
        const buf = Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength)
        reply(id, { ok: true, sampleRate, samples: buf.toString('base64') })
      })
      .catch((e: unknown) => reply(id, { ok: false, error: String(e instanceof Error ? e.message : e) }))
    return
  }
  if (m.type === 'close') {
    try {
      tts?.free()
    } catch {
      /* ignore */
    }
    tts = null
    reply(id, { ok: true })
    return
  }
  reply(id, { ok: false, error: `unknown message type: ${String(m.type)}` })
})

process.on('uncaughtException', (e: unknown) => {
  console.error('[ru-tts-worker] uncaught:', e instanceof Error ? e.stack ?? e.message : e)
  process.exit(1)
})
