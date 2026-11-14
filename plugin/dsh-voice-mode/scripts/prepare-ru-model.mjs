#!/usr/bin/env node
/**
 * fork：准备俄语 Kokoro 模型（首次安装后跑一次即可）。
 *
 * 目标目录与插件运行时一致：<cacheDir>/zaakirio/kokoro-ru
 *   cacheDir 默认 %LOCALAPPDATA%\dsh-voice-mode\models（Windows）或
 *   ~/.cache/dsh-voice-mode/models（其他平台）。
 *
 * 从 Hugging Face 取三样东西（不落盘临时文件，直接写目标目录）：
 *   config.json                 → 生成 tokens.txt（"符号 编号" 每行一个）
 *   voices/sveta.bin, masha.bin → 拼成 voices.bin（[2,510,256] float32）
 *   onnx/model_quantized.onnx   → 追加 sherpa 必需的 metadata_props
 *
 * 用法：node scripts/prepare-ru-model.mjs [--dir <缓存目录>] [--force]
 * 断点：已存在且非空的文件跳过，可重复执行。
 */
import fs from 'node:fs'
import path from 'node:path'
import os from 'node:os'

const REPO = 'zaakirio/kokoro-ru'
const VOICES = ['sveta', 'masha']
const MODEL = 'onnx/model_quantized.onnx'

const argv = process.argv.slice(2)
const flag = (name, fallback) => {
  const i = argv.indexOf(name)
  return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback
}
const defaultCacheDir = () =>
  process.platform === 'win32'
    ? path.join(process.env.LOCALAPPDATA ?? path.join(os.homedir(), 'AppData', 'Local'), 'dsh-voice-mode', 'models')
    : path.join(os.homedir(), '.cache', 'dsh-voice-mode', 'models')

const OUT = path.join(flag('--dir', defaultCacheDir()), REPO)
const FORCE = argv.includes('--force')
const BASE = `https://huggingface.co/${REPO}/resolve/main`

/** 下载到 dest；已存在且非空则跳过（除非 --force）。 */
async function grab(rel, dest) {
  if (!FORCE && fs.existsSync(dest) && fs.statSync(dest).size > 0) {
    console.log(`  = ${rel} (уже есть)`)
    return
  }
  fs.mkdirSync(path.dirname(dest), { recursive: true })
  for (let attempt = 1; attempt <= 4; attempt++) {
    try {
      const res = await fetch(`${BASE}/${rel}`, { redirect: 'follow' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      fs.writeFileSync(dest, Buffer.from(await res.arrayBuffer()))
      console.log(`  + ${rel}`)
      return
    } catch (e) {
      if (attempt === 4) throw new Error(`${rel}: ${e instanceof Error ? e.message : String(e)}`)
      await new Promise((r) => setTimeout(r, 500 * attempt))
    }
  }
}

// ---- protobuf: 给 onnx 追加 metadata_props（field 14）----
function writeVarint(n) {
  const out = []
  let v = n
  while (v >= 0x80) {
    out.push((v & 0x7f) | 0x80)
    v = Math.floor(v / 128)
  }
  out.push(v & 0x7f)
  return Buffer.from(out)
}
const pbField = (tag, payload) => Buffer.concat([writeVarint(tag), writeVarint(payload.length), payload])
const metaEntry = (key, value) =>
  pbField(0x72, Buffer.concat([pbField(0x0a, Buffer.from(key, 'utf8')), pbField(0x12, Buffer.from(value, 'utf8'))]))

function hasMeta(buf, wanted) {
  const found = new Set()
  let pos = 0
  const varint = (b, p) => {
    let r = 0
    let s = 0
    let byte
    do {
      byte = b[p++]
      r += (byte & 0x7f) * 2 ** s
      s += 7
    } while (byte & 0x80)
    return [r, p]
  }
  while (pos < buf.length) {
    let tag
    ;[tag, pos] = varint(buf, pos)
    const field = Math.floor(tag / 8)
    const wire = tag & 7
    if (wire === 2) {
      let len
      ;[len, pos] = varint(buf, pos)
      if (field === 14) {
        const chunk = buf.subarray(pos, pos + len)
        let q = 0
        while (q < chunk.length) {
          let t2, l2
          ;[t2, q] = varint(chunk, q)
          ;[l2, q] = varint(chunk, q)
          if (Math.floor(t2 / 8) === 1) found.add(chunk.subarray(q, q + l2).toString('utf8'))
          q += l2
        }
      }
      pos += len
    } else if (wire === 0) {
      ;[, pos] = varint(buf, pos)
    } else if (wire === 5) pos += 4
    else if (wire === 1) pos += 8
    else break
  }
  return wanted.every((k) => found.has(k))
}

console.log(`модель → ${OUT}`)
console.log('1/4 загрузка config.json')
const cfgPath = path.join(OUT, 'config.json')
await grab('config.json', cfgPath)

console.log('2/4 загрузка голосовых паков')
for (const name of VOICES) await grab(`voices/${name}.bin`, path.join(OUT, 'voices', `${name}.bin`))

console.log('3/4 загрузка весов (~132 МБ)')
const rawModel = path.join(OUT, 'model.onnx')
await grab(MODEL, rawModel)

console.log('4/4 сборка files')
const cfg = JSON.parse(fs.readFileSync(cfgPath, 'utf8'))
fs.writeFileSync(path.join(OUT, 'tokens.txt'), Object.entries(cfg.vocab).map(([s, id]) => `${s} ${id}`).join('\n') + '\n')

const packs = VOICES.map((name) => fs.readFileSync(path.join(OUT, 'voices', `${name}.bin`)))
for (const [i, pack] of packs.entries()) {
  if (pack.length !== 510 * 256 * 4) throw new Error(`voice ${VOICES[i]} has ${pack.length} bytes, expected ${510 * 256 * 4}`)
}
fs.writeFileSync(path.join(OUT, 'voices.bin'), Buffer.concat(packs))

const weights = fs.readFileSync(rawModel)
const meta = [
  ['sample_rate', '24000'],
  ['n_speakers', String(VOICES.length)],
  ['style_dim', '510,1,256'],
  ['has_espeak', '1'],
  ['voice', 'ru'],
  ['speaker_names', VOICES.join(',')],
]
if (!hasMeta(weights, meta.map(([k]) => k))) {
  fs.writeFileSync(rawModel, Buffer.concat([weights, ...meta.map(([k, v]) => metaEntry(k, v))]))
  console.log('  metadata_props добавлены')
}

console.log('\nготово:')
for (const f of ['model.onnx', 'voices.bin', 'tokens.txt']) {
  const st = fs.statSync(path.join(OUT, f))
  console.log(`  ${f.padEnd(12)} ${(st.size / 1024 / 1024).toFixed(1)} МБ`)
}
console.log('\nтеперь включите движок Kokoro в настройках плагина и нажмите прослушать.')
