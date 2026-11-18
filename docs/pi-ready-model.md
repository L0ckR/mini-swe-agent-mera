# Готовая RL-модель и нативный Pi для SWE MERA

Кандидат использует готовую RL-LoRA, обученную внутри Pi. Код не запускает
собственное обучение и не переносит её без проверки на прежний интерфейс mini
с семью инструментами. Родной Pi SDK сохраняет свой loop и четыре определения
инструментов, а операции исполняются через Harbor в отдельном task container.

## Какие именно веса и runtime

| Артефакт | Зафиксированная версия |
|---|---|
| BF16 base | Qwen/Qwen3.8-27B |
| Base revision | 37561816e66b5b9153b1b64c10968c18c1e6f4a0 |
| Готовый adapter | harness-hub/Qwen3.8-27B-pi-adapted |
| Adapter revision | e786f8491742268f9b2bc729697b13bfac26d4ad |
| Adapter weights | 637610624 байта, 512 F32 tensors, LoRA rank 32 / alpha 64 |
| Adapter SHA256 | 869b2a089882c0dac4683993f11081bd70edc30b32c64a988c688d6ac26b792a |
| vLLM | 0.31.0, Docker digest c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3 |
| Pi package | @earendil-works/pi-coding-agent 1.5.2 |
| Pi release commit | b4afdddf44e059c67ccf5097e5317d913521b470 |
| Node | 22.23.5, Docker digest e9f565d32859f0cbfb52125eafe8be39390f24dac302dc3cdf566e8ba3ab21ee |

Adapter config требует именно Qwen/Qwen3.8-27B, а не FP8 repository прежнего
сабмита. Весы не сливаются с основой и не переквантуются: vLLM загружает LoRA
через отдельный model alias qwen38-27b-pi-mera, с auto dtype BF16.
Основа дополнительно видна как qwen38-27b-base для контроля. Tokenizer и
chat template берутся из pinned adapter snapshot.

Thinking выключен в Pi и по умолчанию на сервере, как в обучении. Temperature
0.8. Используются qwen3_xml parser, OpenAI chat completions, prefix cache и
mamba-cache-mode align. Последний необходим для повторного использования
истории этой hybrid architecture. vLLM 0.31.0 явно поддерживает Qwen3.5 LoRA;
его registry относит qwen3_xml и qwen3_coder к одному parser engine.
Это проверка исходников и аргументов запуска, а не GPU smoke.

Условия авторской оценки отличаются от наших: Terminal-Bench 2.1 имел до
одного часа, SWE-bench Pro hard51 — до двух часов, при контексте 131072.
Мы ограничиваем агента 1680 секундами, 100 ходами, контекстом 65536 и выводом
8192, чтобы оставить время до общего 30-минутного ограничения. Результаты
карточки не являются прогнозом SWE MERA score.

Точная версия Pi, использованная авторами RL-обучения, в карточке не закреплена.
Наша версия 1.5.2 закреплена явно; эквивалентность training runtime не заявляется.

## Граница исполнения

1. Harbor импортирует custom_agent.react_bash_agent:ReactBashAgent, который
   экспортирует PiAgent. Он получает task instruction и model connection.
2. В inference-контейнере supervisor запускает vLLM на localhost:8001 и
   Python HTTP service на 8000. Все /v1/* запросы передаются в vLLM, включая SSE.
3. Service выделяет отдельный Node process и in-memory Pi session на trial.
   Включены только read, bash, edit и write, их родные definitions и prompt
   contributions; проектные extensions, skills, auth discovery и обновления
   не загружаются. Npm dependencies устанавливаются при сборке по package-lock.
4. Нативные operation callbacks отправляют JSON-lines RPC. Python адаптер
   выполняет операции только через environment.exec в соответствующем /testbed.
   Исходники задач не находятся в inference-контейнере, host filesystem не
   подменяет task filesystem. Для разных trials нет общей рабочей директории.
5. Текстовое завершение Pi сохраняется вместе с native events и usage. При
   завершении либо лимите выполняются проверки file constraints и экспорт
   change.diff, если его требует исходная постановка. Исправление тестов не
   маскируется удалением test hunks из патча.

Task container не обязан иметь Python или Node. При отсутствии Python
native file ports, Git snapshots и экспорт diff переходят на Bash/coreutils/Git;
это проверено на Go/PHP файлах в PATH без Python. Интерпретация результатов
и контроль выполняются существующим Python worker Harbor.

## Ресурсы и доступная проверка

Конфигурация содержит три независимые GPU replicas, tensor parallel 1,
max-num-seqs 4, BF16 KV cache и 90% memory utilization. BF16 base занимает
около 55.6 GB, исходный F32 adapter — около 0.64 GB. При загрузке LoRA
downcast выполняет vLLM. Настройки оставляют место под KV/linear states,
но память, latency и concurrency 20 ещё необходимо измерить на GPU.

Локальная RTX 4070 Ti SUPER имеет 16 GB и не вмещает эту BF16-связку.
В проверенной Yandex конфигурации квота A100 равна нулю; доступная T4 также
имеет 16 GB. Поэтому статус полного model startup, actual model tool calls,
throughput и полного публичного прогона — NOT_RUN. Альтернативная маленькая
модель не выдаётся за проверку выбранных готовых весов.

Проверяются CPU-only launch/download contracts, хеши и структура adapter,
настоящий Pi SDK на подготовленном SSE endpoint, передача операций в task
environment, изоляция сессий и доставка patch. Ответы scripted endpoint
не являются результатом решения задач моделью.

Тестовый API-check сохраняет короткий bash/pwd probe, повторяет tool result,
использует tool-schema.json, сгенерированный реальным SDK и сравнённый с его
wire payload, проверяет все четыре tool schemas и обычное текстовое завершение.
Никакого finish tool у Pi нет. Все запросы и concurrent smoke используют
thinking off. Здоровье сервиса проверяется отдельно от качества ответа модели.

Следующая полноценная оценка должна сравнить готовые Pi+base и Pi+adapter
при одинаковых задачах и ресурсах, затем прежний mini candidate. В публичном
SWE MERA 18 Go, 15 PHP, 16 Python; solved/total нужно учитывать отдельно по
языкам. Обучающие данные и внешние траектории не добавлены в runtime.

## Первичные источники

- [Готовый RL adapter, pinned revision](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted/tree/e786f8491742268f9b2bc729697b13bfac26d4ad).
- [BF16 base, pinned revision](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0).
- [Pi SDK release source](https://github.com/earendil-works/pi/tree/b4afdddf44e059c67ccf5097e5317d913521b470).
- [vLLM Qwen3.5 implementation](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/model_executor/models/qwen3_5.py).
- [vLLM dense Qwen3.5 LoRA tests](https://github.com/vllm-project/vllm/blob/v0.31.0/tests/lora/test_qwen35_densemodel_lora.py).
- [vLLM tool parser registry](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/tool_parsers/__init__.py).

Исходная mini-версия и SHA256 сохранены в experiments/mini_before_pi.
Её старые отчёты не переименовываются в результаты Pi или новой модели.
