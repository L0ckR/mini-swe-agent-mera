# SWE MERA: готовая RL-модель под Pi

Текущий кандидат Stage II использует **Qwen/Qwen3.8-27B в BF16 и готовый
harness-hub/Qwen3.8-27B-pi-adapted LoRA**, работающие с настоящим
**Pi SDK 1.5.2**. Собственное дообучение не запускается.

Это адаптация готовой модели к контракту соревнования. Полноценный запуск
выбранных весов, GPU startup, пропускная способность и конкурсный pass@1 пока
**не проверены**. Прежние прогоны маленькой 8B-модели относятся к старому mini
harness и не являются результатом этого кандидата.

## Как устроен сабмит

Harbor импортирует обязательный класс
custom_agent.react_bash_agent:ReactBashAgent; он указывает на PiAgent.
Python-адаптер управляет отдельной нативной Pi-сессией через сервис в
inference-контейнере. Установка Node на Harbor worker не требуется.

Внутри inference-образа supervisor запускает private vLLM на 127.0.0.1:8001
и HTTP bridge на порту 8000. Bridge передаёт /v1/* в vLLM и запускает отдельный
Node-процесс для каждой Pi-сессии. Pi сохраняет собственные управляющий цикл,
промпт, четыре инструмента **read, bash, edit, write**, формат результатов и
компактацию истории. Его операции чтения, записи и команд передаются Python
адаптеру и исполняются только через Harbor environment.exec в /testbed.
Task filesystem в inference-контейнер не монтируется.

Для Go/PHP images наличие Python внутри задачи не требуется: при его
отсутствии чтение, запись, проверка изменений и экспорт diff используют
Bash/coreutils/Git. Настоящий Python interpreter нужен только worker Harbor.

Адаптер сохраняет исходную постановку, учитывает ограничения на non-test
изменения и экспортирует /testbed/change.diff, когда его требует задача.
В 22 из 49 публичных постановок собирается только этот файл. Применимость
diff и ограничения файлов проверяются отдельно от поведения реализации.

[Подробности модели и интеграции](docs/pi-ready-model.md).

## Зафиксированные настройки

| Параметр | Значение |
|---|---|
| Основа | Qwen/Qwen3.8-27B, BF16 |
| Base revision | 37561816e66b5b9153b1b64c10968c18c1e6f4a0 |
| RL-адаптер | harness-hub/Qwen3.8-27B-pi-adapted, LoRA rank 32 |
| Adapter revision | e786f8491742268f9b2bc729697b13bfac26d4ad |
| Рабочий model alias | qwen38-27b-pi-mera |
| vLLM / Node / Pi SDK | 0.31.0 / 22.23.5 / 1.5.2 |
| Thinking / temperature | выключен / 0.8 |
| GPU replicas / tensor parallel | 3 / 1 |
| Контекст / максимальный вывод | 65536 / 8192 токенов |
| KV cache / max sequences на replica | BF16 auto / 4 |
| Лимит агента | 1680 секунд, до 100 ходов |

Модель и адаптер скачиваются при сборке, по immutable revision; никаких весов
в ZIP нет. Adapter SHA256, размер, конфигурация, 512 F32 LoRA-тензоров и их
пары проверяются до завершения сборки. Runtime offline. Base и adapter имеют
разные API aliases, поэтому запрос к рабочему alias не должен случайно попасть
на основу без LoRA.

Три BF16 replicas и max sequences 4 — начальная настройка под 3 × 96 ГБ.
Это не измеренный результат нагрузки при 20 одновременных задачах. Локальная
RTX 4070 Ti SUPER 16 ГБ не вмещает выбранную BF16-связку. В доступной Yandex
конфигурации проверены ограничения квот: квота A100 равна нулю, доступная T4
имеет 16 ГБ; подходящий GPU-прогон пока не выполнен.

## Проверить и собрать

Нужен Python 3.12. Для CPU-проверок Docker и GPU не требуются.

~~~bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/fetch_public_statements.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/validate_submission.py submission_stage2
.venv/bin/python scripts/package_submission.py submission_stage2 --output dist/mera-pi-stage2.zip
~~~

Для проверки настоящего SDK нужны Node 22.19+ и npm. Зависимости ставятся в
отдельную локальную копию, чтобы node_modules не попали в upload context:

~~~bash
mkdir -p local_runs/pi-native
cp -a submission_stage2/inference/pi/. local_runs/pi-native/
npm ci --prefix local_runs/pi-native --ignore-scripts --omit=dev --no-audit --no-fund
node local_runs/pi-native/worker.mjs --check
npm test --prefix local_runs/pi-native
.venv/bin/python scripts/check_pi_bridge.py --worker-path local_runs/pi-native/worker.mjs --output jobs/pi-bridge-smoke-NEXT
~~~

SDK smoke использует настоящий Pi и подготовленный локальный SSE endpoint.
Он проверяет loop, native schemas, обработку инструментов и остановку.
Он не проверяет качество весов.

На сервере с подходящими GPU и Docker:

~~~bash
.venv/bin/python scripts/preflight.py
.venv/bin/python scripts/run_local.py --username mera-pi-001 --tasks official_checkout/tasks --concurrency 20
~~~

Официальные runner, API-check и requirements сохранены без изменений в
vendor/stage2 с provenance. Неизменённый runner собирает inference, проверяет
bash/pwd и запускает ровно одну попытку на задачу. Для отдельной проверки
уже запущенного сервера:

~~~bash
.venv/bin/python scripts/check_inference.py --model qwen38-27b-pi-mera --concurrency 20 --output jobs/pi-api-check.json
~~~

Эта проверка использует actual Pi schemas, normal-text completion и thinking
off; она остаётся короткой проверкой API, а не бенчмарком длинных контекстов.

## Сравнение и сохранённые результаты

Публичный набор содержит 18 Go, 15 PHP и 16 Python задач. Следующее сравнение
должно отдельно показывать solved/total по каждому языку и общее время,
ошибки инструментов и таймауты при одинаковых ресурсах.

Предыдущий mini-swe-agent кандидат сохранён вместе с SHA256 в
experiments/mini_before_pi/submission_stage2 и source_sha256.json.
[Исторические mini/8B-диагностики](docs/local-quality-results.md) и
[его описание](docs/harness-fork.md) сохраняются без переименования в результаты Pi.
Ни hidden verifiers, ни gold patches соревнования в обучение и сабмит не входят.

Публичные постановки хранятся локально; для воспроизводимости опубликован
inventory с pinned URL/SHA256 и scripts/fetch_public_statements.py.
Данные задач, jobs, траектории, ключи, npm cache и node_modules в ZIP не входят.
Загрузка на площадку выполняется отдельно.
