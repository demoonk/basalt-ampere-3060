# Basalt-3060: задания по правке кода (раунд 2, work order)

> Адресовано агенту с правами на правку `C:\Neurolib\Tools\basalt-ampere-3060` (ветка `ampere-3060`,
> HEAD `6ea194ed`). Каждая правка — отдельный коммит, сборка `build-ampere.cmd 6`, проверка по
> указанной команде. Ничего не пушить.
> Контекст и числа — в `FORK-3060-v0.5.0.md`, `SPEEDUP-3060.md`, `basalt-3060-speedup-task-2.md`.

---

## 0. Сводка правок

| id | файл | суть | риск | приоритет |
|---|---|---|---|---|
| E-01 | `kernels/native_moe.cu:190-229` | переделать S1-патч: решение о give-up на весь блок + жёсткий отказ на хосте | средний | 1 |
| E-02 | `decode/verify/expert_resources.cpp:216` | `stage_gpu_us` 250 → 25 µs (модель никогда не выбирает k>0) | низкий | 1 |
| E-03 | `decode/verify/record_layer.cpp:329` | взводить `stage_src` только когда есть что стейджить | средний | 1 |
| E-04 | `kernels/dec_moe*` / `execution.cpp:432` | при нуле staged-групп не ждать флаг B → снять `never rang` | средний | 1 |
| E-05 | `cli/report.cpp:146` + счётчик | «pcie experts 0.00» → считать staged в dm-пути | низкий | 2 |
| E-06 | `kernels/cpu/pool.cpp` | `ms_multi_q` не накапливается → мерить или убрать печать | низкий | 2 |
| E-07 | `cli/gpu_modes.cpp:261` + `help` | `--gpu-only-full` не работает на этом билде → честный отказ | низкий | 2 |
| E-08 | `kernels/cpu/pool.cpp:34` | `pin_pool` по умолчанию на Windows + пин хоста | низкий | 2 |
| E-09 | `platform/cpu.cpp:378` | `pin_thread`: `% 64` алиасит на >64 CPU | низкий | 3 |
| E-10 | `run-arms*.cmd` | `--exp` в обёртке ломается (не движок) | — | 3 |
| E-11 | доки | `n_ff 320→640`, «пул быстрее в 3.4×», 4852 vs 3000 слотов, 4/10 | — | 1 |
| E-12 | `flags.cpp` | lossy-флаги в serve требуют `--allow-lossy` | низкий | 3 |

**Порядок:** E-02 + E-03 + E-04 одним коммитом (это один механизм), затем E-01, затем E-05..E-08,
E-11 — сразу.

---

## 1. Ревью двух новых коммитов

### 1.1 `8b4ae8d0` (S6, пиннинг пула) — принять с правками

Работает и сделано аккуратно (флаг `--exp pin_pool=1`, платформенная обёртка, лог на каждый пин).
Замечания:

1. `src/platform/cpu.cpp:381` — `1ull << (processor % 64)`: на машине с >64 логическими CPU
   `processor=64` молча сядет на CPU 0. Заменить на `if (processor < 0 || processor >= 64) return false;`
   (для 16 CPU этого достаточно; группы процессоров всё равно не поддержаны — так и написать).
2. Хост-поток не пинован: он крутит `_mm_pause` в wait-цикле и может уехать на ядро, занятое
   пинованным воркером. Добавить пин хоста на последнее физическое ядро (или на ядро 0) — там же,
   где создаётся пул.
3. Дефолт: если `--exp pin_pool` не задан на Windows и физических ядер ≥ 8 — включать. Замер S6
   в SPEEDUP не проводился (в отчёте его нет), поэтому включать только после 3 прогонов × 1200
   токенов против baseline; иначе оставить выключенным и записать «не измерено».

### 1.2 `6ea194ed` (S1, give-up в combine) — **переделать**, это опасно в текущем виде

Что не так:

1. **`gave_up` — локальная переменная потока.** В ветке `WAITC` её ставит либо поток 0 блока (0,0)
   (строка 201), либо каждый поток сам в своём цикле (209-216). Значит колонки одного и того же
   токена могут посчитать разные наборы экспертов: один поток подставил нули, соседний прочитал
   реальные строки. Результат недетерминированный и молча неверный — хуже падения.
2. **Нет жёсткого отказа на хосте.** Комментарий говорит «a result the host is about to discard»,
   но в коде нет ничего, что гарантирует отбрасывание окна: если `wait_give_up` записал код в
   `abort`, хост обязан **упасть/отклонить запрос**, а не продолжать генерацию. Один «нулевой»
   токен отравляет весь дальнейший ответ (авторегрессия).
3. **Бюджет 4 мс** (`wait_abort.hpp:20`, `kWaitGiveUpNs = 4'000'000`) против реального времени
   пула на слой ≈0.93 мс (44.5 мс / 48) — запас всего 4×. Любой всплеск (антивирус, orphan-процессы,
   page-in) поднимет give-up на **здоровом** окне и молча обнулит экспертов.

### E-01. Переделать give-up (правка)

Файл: `src/models/qwen4exp/kernels/native_moe.cu`, шаблон `dm_combine4_kernel`, строки 172-229.

1. Перенести весь блок `if constexpr (WAITC) { ... }` (сейчас 192-218) **выше** раннего выхода
   `if (!live) return;` (строка 173) — сразу после `__syncthreads();` (строка 172). Причина:
   `__syncthreads_or` ниже недостижим для потоков, которые уже вернулись.
2. Внутри блока: локальный `bool gave_up` заполняется как сейчас, затем одна строка на весь блок:

```cpp
        // every column of this token must add the same experts: one decision for the block
        gave_up = __syncthreads_or(gave_up);
```
   (`__syncthreads_or` — sm_70+, есть на sm_86.)
3. Публикацию `cpu_seen = ring` (строка 204) **оставить безусловной** — иначе deadlock; но при
   `gave_up == true` дополнительно записать слово ошибки:

```cpp
            if (gave_up && abort != nullptr)
                st_release_gpu(abort /* или отдельное слово given_up */, wait_abort_code_of(kWaitCombine, 9, 0u, 0u));
```
   Если `abort` менять нельзя — добавить в аргументы `unsigned* given_up` (mapped host word),
   поднять его, и читать на хосте.
4. Хост (`decode/verify/execution.cpp`, после `cudaStreamSynchronize`/перед использованием `h_ymiss_`):
   если слово ошибки поднято — `err = "verify: the CPU pool did not publish layer N within X ms";
   return -1;` То есть **окно с нулями никогда не доходит до сэмплера**.
5. Бюджет: `kWaitGiveUpNs = 4'000'000` → 15'000'000 и вынести в `--exp wait_combine_us` (дефолт 15000).
   Обоснование: 0.93 мс/слой × 16 = 15 мс — 16× запас на всплеск, при этом зависание по-прежнему
   диагностируется, а не висит вечно.
6. Лог: при срабатывании писать отдельную строку `basalt verify: combine gave up on layer N (pool word
   after X ms)` — сейчас отличить give-up от обычного окна по логам нельзя.

**Проверка:** `--max-new 1200` × 3 прогона: decode в пределах baseline (26–28), в логе **нет** строк
give-up; затем искусственно: `--exp wait_combine_us=1` — движок должен вернуть ошибку запроса
(а не молча выдать токен), сервер — 500, процесс не падает.

---

## 2. Стейджинг PCIe: одна причина, три правки

Диагноз по коду: устройство-promoted kind 1 делает `dm_stage_split` (`kernels/dec_moe.cuh:149-190`)
по таблице `stage_k[m]`, которую считает `expert_resources.cpp:219-226`. Модель стоимостей с
дефолтом `stage_gpu_us = 250` **не выбирает k>0 ни при одном m** (при реальных 115 µs CPU и
106 µs линка staged-эксперт «стоит» 250 µs GPU, а он стоит 23 µs). Поэтому:

- стейджинг мёртв даже с `--exp solo_stage=1`;
- но `ra.stage_src` всё равно взводится (`record_layer.cpp:329`) → запускается `dec_moe_stage`
  с нулём групп, а граф ждёт флаг B (`execution.cpp:432`) → ровно тот `never rang` на слое 43/6,
  который наблюдался в E1a/E1b.

### E-02. Калибровка модели стоимостей

`decode/verify/expert_resources.cpp:216`:

```cpp
const double kStageGpuUs = (double) basalt::core::exp_flag("stage_gpu_us", 250.0);
```
→
```cpp
// measured: 14.71 ms of GPU FFN per round over 634 records = 23.2 us per staged expert
const double kStageGpuUs = (double) basalt::core::exp_flag("stage_gpu_us", 25.0);
```
И рядом, после построения `kt`: если таблица целиком нулевая — предупредить, а не молчать:

```cpp
int st_max = 0; for (int m = 0; m <= kDmCap; ++m) st_max = std::max(st_max, kt[(size_t) m]);
if (st_max == 0)
    basalt::log::warn("basalt verify", "staging is off: the cost model picks no share (cpu %.0f us, link %.0f us, "
                      "gpu %.0f us per expert)", cpu_us, link_us, kStageGpuUs);
```
При 12/13/25 GB/s/µs и m=6.5 оптимум даёт **k≈2–3** — это и есть цель.

### E-03. Не взводить стейдж, когда стейджить нечего

`decode/verify/record_layer.cpp:329`:

```cpp
if (v_.dm_->stage_src != nullptr && (v_.batch_rec_ || v_.solo_stage_)) {
```
→ добавить условие по числу процессорных миссов окна (его публикует dm-план в `h_ring[k].n_cpu`,
`counts[3]`):

```cpp
const bool stage_here = v_.dm_->stage_src != nullptr &&
                        (v_.batch_rec_ || v_.solo_stage_) &&
                        v_.dm_->stage_min > 0 &&
                        v_.dm_->h_ring[k].n_cpu >= v_.dm_->stage_min;
if (stage_here) { ... }
```
(`stage_min` уже вычисляется в `expert_resources.cpp:247-248` и сейчас не используется — задействовать.)

### E-04. Инвариант: каждое ожидание графа освобождается всегда

Файл: `kernels/dec_moe*.cuh` (`dm_stage_kernel`/`dec_moe_stage`, `kernels/dec_moe.hpp:105`) и
`decode/verify/execution.cpp:432`.

Требование, которое надо обеспечить кодом: **если staged-групп ноль, `dec_moe_stage` не запускается
и флаг B публикуется роутером сразу с count=0**. Сейчас путь такой: граф ждёт B, стейдж-ядро с
`n_groups=0` ничего не публикует → вечное ожидание до give-up.

1. В `dec_moe_stage(...)`: `if (a.n_groups == 0 || a.stage_src == nullptr) { /* publish B with 0 and return */ }`.
2. В `dm_route_kernel`/роутере — публиковать B безусловно (тем же счётчиком, что и для staged).
3. В `execution.cpp:432` — заменить «планируем PCIe-группы всегда» на «планируем, но публикуем B
   с тем числом, что реально будет скопировано».

**Проверка E-02..E-04 (один коммит):**
`--max-new 1200 … --exp solo_stage=1` (без `stage_gpu_us`): в логе `solo staging: ... (pool 12 GB/s,
link 13 GB/s, a staged expert's GPU FFN 25 us, 16 slots)`; в `--profile` бакет `PCIe grp` > 0;
`pool` мс/раунд 44.5 → ≤29; decode ≥30 tok/s; 10/10 в serve.

### E-05. Счётчик `pcie experts` больше не врёт

`cli/report.cpp:146` печатает `pcie_experts`, который инкрементируется только в **host-плане**
(`expert_dispatch.cpp:152-165`), а в dm-режиме (`ext`-ветка, `expert_dispatch.cpp:116`) host-план не
выполняется никогда → в логе всегда `0.00` при `share 93/256`. Это не «tier мёртв», а «счётчик мёртв».

1. В `dm_stage_split` (или в `dec_moe_stage`) добавить накопитель promoted-записей в `dm_->wstat`
   (поле рядом с существующими) и в `report.cpp` печатать его как `staged (PCIe) N/layer`.
2. Если `d.plan->ext_kind != nullptr` и host-план не выполнялся — печатать `pcie experts n/a (device plan)`,
   а не `0.00`.

**Проверка:** при `solo_stage=0` — `n/a` или 0 при явной пометке; при `solo_stage=1` — число > 0.

---

## 3. Мелкие правки

### E-06. `ms_multi_q` (баг-репорт из SPEEDUP §5.2 подтверждён)

`include/basalt/kernels/cpu/pool.hpp:110` объявляет `ms_multi_q`, `cli/report.cpp:127` его печатает,
и **нигде не накапливает**. Вариант A (предпочтительно): в `pool.cpp` вокруг квантования
промежуточного (`native_quantize` / `act_quant_any` в native-пути) поставить
`std::chrono` и `ms_multi_q += ...`. Вариант B: убрать поле и из печати `quantize`, чтобы не врать.
A, потому что квантование — единственная фаза без своего счётчика.

### E-07. `--gpu-only-full`

`cli/gpu_modes.cpp:261` вызывает `session_replay_full`, который требует `gr.posts != nullptr`
(`decode/session.cpp:427`), а decode-путь этого билда пост-графы не пишет. Вариант A: печатать
явный отказ и указывать `--profile` как замену (пол уже считается по стампам). Вариант B: сложить
пол из `--profile` и назвать флаг `--floor-from-profile`. Убрать `--gpu-only-full` из `help` как
рабочий инструмент — сейчас он обещает «THE TRUE PER-TOKEN GPU FLOOR» и падает.

### E-08. `pin_pool` по умолчанию (после замера)

`kernels/cpu/pool.cpp:95-105`: если `exp_flag("pin_pool", -1) < 0` и `_WIN32` и физических ядер ≥ 8 —
пинить. Плюс пин хоста (см. §1.1). `--exp pin_pool=0` — выключатель.

### E-09. `pin_thread` guard

`src/platform/cpu.cpp:378-390`: `if (processor < 0 || processor >= 64) return false;` и убрать `% 64`.

### E-10. Обёртки `--exp`

`logs-speedup\run-arms*.cmd` — E12 упал не из-за движка (`flags.cpp:160-169` принимает
`NAME=VALUE[,NAME=VALUE...]`), а из-за кавычек в обёртке. Одна строка:
`--exp "gu2=4,grouped_v1=1"`. Проверить на одном прогоне, что флаги доходят (в логе `gu2`).

### E-11. Доки (правки в двух отчётах)

- `FORK-3060-v0.5.0.md:14` — `n_ff 320` → `n_ff 640` (иначе не сходятся 1.38 МБ и 4.9 М весов).
- `FORK-3060-v0.5.0.md:104-105` — «CPU-пул в 3.4 раза быстрее» → «вдвое меньше байт: adaptive tier
  свапнул 4608 экспертов против 42; GB/s тот же (12.9)». Ссылка: `cmp-*-conv.log`.
- `FORK-3060-v0.5.0.md:106` — «expert cache auto: 4852–4955» относится к другому прогону; в
  сравнительном — `--expert-cache 3000` (3.86 ГиБ).
- `FORK-3060-v0.5.0.md:136` и `SPEEDUP §4` — serve: **4/10 против 10/10 у 0.1.0** — это регрессия,
  а не «менее стабилен».
- `SPEEDUP-3060.md:80` — пометить `--exp no_lazy=1` как **невалидный** (GDN не коммитятся).
- `SPEEDUP-3060.md:69-74` — «GPU-пол» получен из профиля холодного прогона (99 мс/раунд против
  76–81 у base-r*), пересчитать от base-r*.

### E-12. Lossy-флаги в serve

`src/models/qwen4exp/flags.cpp`: при `--route-tail-skip`/`--route-prior` > 0 **и** режиме serve —
требовать явный `--allow-lossy`, иначе `code = 2` с текстом «this flag ignores part of the routed
experts; pass --allow-lossy to accept a different model». Сейчас `o.lossy` только печатает
предупреждение, а рука уже попала в таблицу отчёта как «+43%».

---

## 4. Что не трогать

- `kernels/s2_expert_grouped.cu` (dp4a) — менять только после роуфлайн-гейта: `ncu` ≥50% DRAM BW и
  ≤10 µs при ne=1. Стенд без `cp.async`/`ldmatrix` доказательством не считается.
- `expert_dispatch.cpp` host-план — оставить как есть; в dm-режиме он не выполняется, «починить» его
  можно только сломав device-план (см. E-05: чинить счётчик, а не план).
- `--route-tail-skip` не делать дефолтом ни при каких числах.
- Модель/пак, `--kv`, `--expert-cache`, `--spec` — не менять в рамках этих правок.

---

## 5. Что добавить до правок, чтобы приёмка была измеримой

1. Счётчик staged-записей в dm-пути (для E-05) — нужен уже для E-02..E-04.
2. Строка лога на give-up (для E-01) — иначе не отличить от обычного окна.
3. `--exp stage_trace=1`: печать `m`, выбранного `k`, `cpu_us/link_us/gpu_us` на первых 8 окнах —
   чтобы видеть, что модель вообще выбирает k>0 (сейчас это невидимо).

**Команды проверки после каждого коммита:**

```
C:\Neurolib\Tools\basalt-ampere-3060\build-ampere.cmd 6
taskkill /f /im basalt.exe & timeout /t 5
bin\basalt.exe --model C:\Neurolib\Models\Basalt\GSQ-RCO-Q2_0\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0.basalt ^
  --tokens-file data\bench\conv.ids --max-new 1200 --greedy --seed 42 ^
  --kv int8 --max-context 8192 --spec 8 --profile > logs-edits\<id>-<n>.log 2>&1
```
Три прогона на точку, медиана; сравнивать только с медианой baseline 26.3 (SPEEDUP §1).
