/**
 * fork 新增：say 工具 —— агент сам решает, что произнести вслух.
 *
 * Про формат. Хост-пакет `@deepseek-ai/dsh-tools` из папки плагина недоступен: его нет
 * ни в профиле, ни в зависимостях, а тянуть его целиком нельзя — он тянет peer-зависимости
 * хоста (dsh-agent, dsh-llm, dsh-scope…). `defineTool` из него лишь преобразует внутренний
 * формат в JSON Schema и оборачивает execute валидацией. Поэтому собираем описание сразу
 * в том виде, который ожидает `ctx.tools.register`: parameters — настоящая JSON Schema,
 * output.schema — JSON Schema результата, output.render — отрисовка. Валидацию аргументов
 * делаем сами в execute.
 */

/** Текст описания = политика использования (работает и в новых сессиях). */
const DESCRIPTION = [
  'Озвучить короткую фразу вслух локальным русским голосом.',
  'Вызывай по своей инициативе: главный вывод или итог, предупреждение,',
  'или короткая отметка о ходе работы, если она длится дольше нескольких секунд.',
  'Одна-две фразы, до 15-20 слов, без markdown, списков, кода и ссылок.',
  'Не дублируй голосом то, что и так читается автоматически.',
  'Возвращает сразу: синтез идёт в фоне, звук приходит в браузер по мере готовности.',
].join(' ')

export interface SayToolOptions {
  /** Фактическая озвучка: инъекция из index.ts (тот же путь, что у маршрута /say). */
  say: (text: string, voice?: string, rate?: number) => void
}

/** Описание инструмента в формате, пригодном для ctx.tools.register без defineTool. */
export function makeSayTool(options: SayToolOptions): Record<string, unknown> {
  return {
    name: 'say',
    description: DESCRIPTION,
    parameters: {
      type: 'object',
      properties: {
        text: { type: 'string', description: 'Что произнести: короткая фраза, до 15-20 слов.' },
        voice: { type: 'string', description: 'Необязательно: masha (по умолчанию) или sveta.' },
        rate: { type: 'number', description: 'Необязательно: скорость 0.5..2, по умолчанию 1.' },
      },
      required: ['text'],
    },
    output: {
      schema: { type: 'string' },
      render: (_args: unknown, value: unknown): Array<{ type: string; text: string }> => [
        { type: 'text', text: String(value) },
      ],
    },
    isConcurrencySafe: (): boolean => true,
    timeoutMs: 15000,
    async execute(args: unknown): Promise<string> {
      const a = (args ?? {}) as { text?: unknown; voice?: unknown; rate?: unknown }
      const text = String(a.text ?? '').trim()
      if (!text) return 'say: пустой текст, ничего не сказано'
      if (text.length > 2000) return 'say: текст длиннее 2000 символов, сократи'
      const voice = typeof a.voice === 'string' && a.voice.trim() ? a.voice.trim() : undefined
      const rate =
        typeof a.rate === 'number' && Number.isFinite(a.rate) ? Math.min(2, Math.max(0.5, a.rate)) : undefined
      options.say(text, voice, rate)
      return `say: принято к озвучке (${String(text.length)} символов)`
    },
  }
}
