# dsh-voice-mode-ru

Русский форк [dsh-voice-mode](https://github.com/qishuilalala/dsh-voice-mode) для DeepSeek Harness: локальная озвучка на русском языке (Kokoro + `zaakirio/kokoro-ru`) и инструмент `say`, позволяющий агенту самому выбирать, что произнести вслух.

Всё, что относится к русскому голосу — возможности, установка, настройки, инструмент `say`, ограничения — описано в **[plugin/dsh-voice-mode/README.md](plugin/dsh-voice-mode/README.md)**.

Коротко:

- два русских голоса: `masha` (по умолчанию) и `sveta`;
- фонемизация через `espeak-ng`, паузы по знакам препинания, регулируемая скорость;
- модель ≈132 МБ готовится один раз: `node scripts/prepare-ru-model.mjs`;
- инструмент `say` — агент говорит главное, остальное выводит текстом;
- код под GPL-3.0-or-later (из-за `espeak-ng`), веса модели — OpenRAIL.

Структура репозитория: пакет для npm лежит в `plugin/dsh-voice-mode`, публиковать нужно оттуда.
