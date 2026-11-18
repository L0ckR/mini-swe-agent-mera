# SWE MERA: MERA Forge

Сабмит для **Stage II**: адаптированный форк mini-swe-agent и локальная inference-модель
**Qwen/Qwen3.8-27B-FP8**, три независимые реплики на GPU организаторов.

Архив собирается командой ниже в `dist/mera-mini-stage2.zip`. Это кандидат:
проверки кода и упаковки не заменяют запуск модели на GPU и измерение pass@1.
Непроверенный результат соревнования не заявляется.

## Что внутри

- `submission_stage2/` — только загружаемые файлы: YAML, Docker build context,
  пакет агента. Веса скачиваются при сборке образа; runtime работает offline.
- `custom_agent/react_bash_agent.py` — адаптер Harbor для настоящего цикла
  mini-swe-agent 2.4.6. Пакет `_vendor/mini_swe_agent` содержит зафиксированный
  upstream, MIT-лицензию и provenance. Добавлены literal search, точный editor,
  сжатие контекста по бюджету, учёт реально выполненных проверок и завершение
  с учётом контракта конкретной постановки, включая экспорт `change.diff`.
- `scripts/` — проверка структуры, воспроизводимая упаковка, запуск официального
  runner, проверка API/нагрузки и чтение настоящих результатов Harbor.
- `vendor/stage2/` — неизменённые runner, API-check и requirements организаторов
  с commit и SHA256 в `SOURCE.json`. Они не попадают в ZIP.
- `vendor/mini_swe_agent/` — исходники upstream до адаптации;
  `experiments/legacy_react_agent.py` сохраняет предыдущий агент для сравнения.
- `data/public_statements/` и `data/public_task_inventory.json` — 49 публичных
  постановок и разрешённые metadata с immutable source URLs и хешами.
  Полные постановки хранятся только локально; для отдельного checkout они
  скачиваются `scripts/fetch_public_statements.py` по pinned URL с проверкой SHA256.
  `experiments/diagnostics/` и `experiments/results/` — воспроизводимые fixtures
  и компактные результаты локальных диагностик. Эти данные не входят в ZIP.

[GitHub-форк](https://github.com/L0ckR/mini-swe-agent-mera) ·
[Выбор harness, изменения и план сравнения](docs/harness-fork.md)

Модель выбрана по официальной карточке: FP8 checkpoint занимает около 31 ГБ и
оставляет место для KV cache на каждой 96-ГБ карте. Независимые реплики позволяют
обслуживать несколько задач без tensor-parallel обмена. Контекст — 65536 токенов;
агент начинает сжимать историю раньше, сохраняя исходную задачу и недавние шаги.
Reasoning включён для решения задач, но выключен по умолчанию на сервере: это
позволяет пройти короткую проверку tool call с `max_tokens=256`.

Результаты из карточки модели относятся к другим наборам и harness; они не
предсказывают pass@1 на SWE MERA. Окончательный выбор подтверждается прогоном.

## Что проверено локально

Живые запуски выполнены на небольшой **`qwen3:8b-q4_K_M` через Ollama**,
а не на целевой 27B-модели. Первые две серии на трёх synthetic задачах дали
0/3 против 0/3 и 1/3 против 1/3 для сохранённых before/after snapshots.
Третья серия только нового snapshot с дополнительным reviewer дала 1/3;
reviewer ошибался, поэтому по умолчанию выключен. Две отдельные попытки
публичной Click py-009 также не решили задачу: обе сохранили 4/6 публичных
критериев. Во второй корректный `change.diff` доставлен и применён к чистой
копии, но целевые ошибки остались. В ней одновременно включён thinking и
увеличен вывод модели, поэтому изменения нельзя приписать одному harness.
Это диагностические наблюдения, а не измерение конкурсного pass@1;
доказанного прироста числа решённых задач пока нет.
[Результаты и ограничения](docs/local-quality-results.md).

По этим запускам добавлены защита от случайного удаления исходного репозитория,
literal search для строк вида `--help`, подсказки при повторных неуспешных edit
и проверка свежести тестов относительно текущего patch. Независимый semantic
review доступен как экспериментальная опция `semantic_review=True`, по умолчанию
выключен: проверяющая модель тоже может ошибаться. Новые механизмы требуют
отдельного измерения на одинаковом backend и наборе задач.

Локальные команды агента исполняются через bubblewrap в `/testbed`, без сети;
при отсутствии доступной изоляции runner прекращает запуск. Нужны Linux,
`bwrap` и отдельно запущенный OpenAI-compatible inference endpoint. Например,
только текущую версию можно проверить без старого ZIP:

```bash
.venv/bin/python scripts/run_local_diagnostics.py --variants after --base-url http://127.0.0.1:18000/v1 --model LOCAL_MODEL --output jobs/diagnostics-live-NEXT
```

Для before/after требуется явно выбрать сохранённый `--before-zip`; каждый
запуск использует новый output-каталог. `scripts/ollama_bridge.py` предоставляет
локальный адаптер API и записывает фактические настройки backend, в том числе
неподдерживаемые параметры. Скрипт не делает Ollama эквивалентом vLLM.

## Проверить и собрать

Python 3.12, `uv`; GPU и Docker для этих команд не нужны:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/fetch_public_statements.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/validate_submission.py submission_stage2
.venv/bin/python scripts/package_submission.py submission_stage2 --output dist/mera-mini-stage2.zip
.venv/bin/python scripts/validate_submission.py dist/mera-mini-stage2.zip
```

В корне ZIP должны быть `submission.yaml`, `custom_agent/`, `inference/`, без
обёртки `submission_stage2/`. Рядом с ZIP сохраняется manifest с хешами файлов.
Траектории, данные задач, `.env`, ключи и локальные runner в архив не включаются.

## Проверить на GPU-сервере

Целевая среда: Linux x86_64, Docker Engine + Compose + NVIDIA Container Toolkit,
**3 × RTX PRO 6000 Blackwell 96 ГБ**, достаточная RAM хоста для vLLM и свободное
место под модель, Docker build layers и образы задач. Практический стартовый
ориентир — от 140 ГиБ свободного места; Docker может хранить данные на другом диске.
Ресурсы inference-хоста и лимиты CPU/RAM отдельных task-контейнеров — разные вещи.

```bash
.venv/bin/python scripts/preflight.py
git clone https://github.com/MERA-Evaluation/SWE-MERA-CHALLENGE.git official_checkout
git -C official_checkout checkout 070bcb675d96845745e28b40f3830c98139217ea
.venv/bin/python scripts/run_local.py --username mera-forge-001 --tasks official_checkout/tasks --concurrency 20
```

`run_local.py` временно размещает доверенные файлы рядом с сабмитом и вызывает
официальный runner. Он собирает образ, проверяет tool calls, запускает Harbor с
**одной попыткой на задачу**, затем удаляет inference container/image. Результаты
остаются в `jobs/stage2/mera-forge-001/`. До полной оценки стоит проверить один
публичный task в отдельном каталоге, переданном через `--tasks`, чтобы обнаружить
ошибки инфраструктуры без запуска всего набора.

В текущем upstream runner default concurrency — **20**, хотя README организаторов
указывает 10. Здесь проверяется более тяжёлый режим 20; при заданном организаторами
другом значении необходимо измерить и его.

Для отдельной проверки пропускной способности, пока inference работает:

```bash
.venv/bin/python scripts/check_inference.py --concurrency 20 --output jobs/inference-check.json
.venv/bin/python scripts/summarize_run.py jobs/stage2/mera-forge-001/ИМЯ_ПАПКИ_ПРОГОНА --json
```

Проверка API использует настоящий запрос организаторов, повторяет tool call после
результата инструмента, проверяет инструменты агента и запускает параллельные
запросы с reasoning. Это короткая проверка протокола и нагрузки; скорость при
длинных контекстах и укладывание в 30 минут определяются реальными задачами. Итоговый
score читается из оригинальных `verifier/reward.txt`; скрипты их не меняют.
В начале решать проблемы запуска и протокола, затем сравнивать модель и настройки
по полному публичному набору: solved/total, missing rewards, exceptions, время и
число токенов. Публичный и приватный score считать раздельно.

## Источники и ограничения

- [Официальный контракт Stage II](https://github.com/MERA-Evaluation/SWE-MERA-CHALLENGE/tree/070bcb675d96845745e28b40f3830c98139217ea/example_stage2).
- [Карточка Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8).
- [vLLM recipe](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-27B.yaml).
- [Harbor 0.21.0](https://github.com/harbor-framework/harbor/tree/v0.21.0).

Не использовать gold patches, скрытые тесты или готовые решения задач. Сохранять
оригинальные jobs и trajectories. В ZIP загружается программа решения и inference,
а не результаты локальной оценки. Загрузка на площадку выполняется отдельно.
