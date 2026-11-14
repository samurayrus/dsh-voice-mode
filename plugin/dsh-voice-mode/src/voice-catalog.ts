/**
 * 音色目录：host（name ↔ sid 解析）与客户端（设置面板下拉的显示标签）共用的**单一数据源**。
 *
 * 此前 host 的 tts-local.ts 与客户端 settings-form.tsx 各抄一份（含中文标签），靠注释保证「同源」，且标签写死中文，
 * 英文界面下拉里满是「女 / 男声 / 常用男声」。现在这里只放**数据**（说话人名称的中/英写法、性别、口音、F0），
 * 显示用的描述词（性别、口音、风格）由客户端经词典翻译（见 voice-labels.ts），本文件不含任何界面文案。
 */

export type Gender = 'F' | 'M'

/** 说话人名称（专有名词）的中/英写法：英文为拼音/官方英文名，语言包缺失的语言回落英文。 */
export interface BiName {
  zh: string
  en: string
}

// ── Edge 云端常用音色（ShortName 取自 msedge-tts getVoices 实测权威清单）──────────────
export type EdgeAccent = 'mandarin' | 'northeast' | 'shaanxi' | 'cantonese' | 'taiwan' | 'english'
export interface EdgeCommonVoice extends BiName {
  v: string
  gender: Gender
  accent: EdgeAccent
}
export const EDGE_COMMON_VOICES: ReadonlyArray<EdgeCommonVoice> = [
  { v: 'zh-CN-XiaoxiaoNeural', zh: '晓晓', en: 'Xiaoxiao', gender: 'F', accent: 'mandarin' },
  { v: 'zh-CN-XiaoyiNeural', zh: '晓伊', en: 'Xiaoyi', gender: 'F', accent: 'mandarin' },
  { v: 'zh-CN-YunxiNeural', zh: '云希', en: 'Yunxi', gender: 'M', accent: 'mandarin' },
  { v: 'zh-CN-YunjianNeural', zh: '云健', en: 'Yunjian', gender: 'M', accent: 'mandarin' },
  { v: 'zh-CN-YunyangNeural', zh: '云扬', en: 'Yunyang', gender: 'M', accent: 'mandarin' },
  { v: 'zh-CN-YunxiaNeural', zh: '云夏', en: 'Yunxia', gender: 'M', accent: 'mandarin' },
  { v: 'zh-CN-liaoning-XiaobeiNeural', zh: '小北', en: 'Xiaobei', gender: 'F', accent: 'northeast' },
  { v: 'zh-CN-shaanxi-XiaoniNeural', zh: '小妮', en: 'Xiaoni', gender: 'F', accent: 'shaanxi' },
  { v: 'zh-HK-HiuMaanNeural', zh: '晓曼', en: 'HiuMaan', gender: 'F', accent: 'cantonese' },
  { v: 'zh-HK-WanLungNeural', zh: '云龙', en: 'WanLung', gender: 'M', accent: 'cantonese' },
  { v: 'zh-TW-HsiaoYuNeural', zh: '小雨', en: 'HsiaoYu', gender: 'F', accent: 'taiwan' },
  { v: 'zh-TW-YunJheNeural', zh: '云哲', en: 'YunJhe', gender: 'M', accent: 'taiwan' },
  { v: 'en-US-AriaNeural', zh: 'Aria', en: 'Aria', gender: 'F', accent: 'english' },
  { v: 'en-US-GuyNeural', zh: 'Guy', en: 'Guy', gender: 'M', accent: 'english' },
]

// ── 本地 VITS（vits-zh-ll 五说话人；性别按 2026-08 用户听测纠正：顾念/冰娇/霸总为男声、傅斯遇为女声）──
export interface VitsSpeaker extends BiName {
  name: string
  sid: number
  gender: Gender
}
export const VITS_SPEAKERS: ReadonlyArray<VitsSpeaker> = [
  { name: 'suyingxue', sid: 0, zh: '素映雪', en: 'Su Yingxue', gender: 'F' },
  { name: 'gunian', sid: 1, zh: '顾念', en: 'Gu Nian', gender: 'M' },
  { name: 'fushiyu', sid: 2, zh: '傅斯遇', en: 'Fu Siyu', gender: 'F' },
  { name: 'bingjiao', sid: 3, zh: '冰娇', en: 'Bing Jiao', gender: 'M' },
  { name: 'bazong', sid: 4, zh: '霸总', en: 'Ba Zong', gender: 'M' },
]

// ── fork：本地 Kokoro 换成俄语模型 zaakirio/kokoro-ru（两个女声，sid 0/1）──
/** 各 sid 的基频 F0（Hz，实测）。俄语包只有 sveta 与 masha 两个音色。 */
export const KOKORO_F0: ReadonlyArray<number | null> = [196, 208]

/** 俄语音色（name 为引擎侧英文名，host 侧 kokoroVoiceToSid 按它做映射）。 */
export const KOKORO_NAMED: Readonly<Record<number, { name: string } & BiName>> = {
  0: { name: 'sveta', zh: 'Света', en: 'Sveta' },
  1: { name: 'masha', zh: 'Маша', en: 'Masha' },
}

/** 俄语包没有男声，保留类型与空表以兼容上层引用。 */
export type KokoroStyle = 'deep' | 'rich' | 'clear' | 'magnetic'
export const KOKORO_POPULAR_MALE: Readonly<Record<number, KokoroStyle>> = {}

/** 置顶顺序：默认音色 masha(1) 排第一，其次 sveta(0)。 */
export const KOKORO_PINNED: ReadonlyArray<number> = [1, 0]

/** host 侧用：Kokoro 音色 name↔sid 表（置顶在前）。name = 引擎英文名或数字字符串。 */
export const KOKORO_VOICES: ReadonlyArray<{ name: string; sid: number }> = [
  ...KOKORO_PINNED.map((sid) => ({ name: KOKORO_NAMED[sid]?.name ?? String(sid), sid })),
  ...KOKORO_F0.map((_, sid) => ({ name: KOKORO_NAMED[sid]?.name ?? String(sid), sid })).filter(
    (v) => !KOKORO_PINNED.includes(v.sid),
  ),
]
