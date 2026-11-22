# Задание: ускорить Basalt-ampere на RTX 3060 (sm_86)

> Самодостаточное ТЗ. Отдаётся целиком другой нейросети/агенту, у которого есть shell-доступ
> к этой машине (Windows 11, HOMEPC). Всё, что нужно для старта, — внутри файла; перепроверять
> код с нуля не требуется, но все числа из §3 надо воспроизвести самому перед правками.

---

## 0. TL;DR для исполнителя

Движок уже написан так, что умеет отдавать часть экспертов GPU по PCIe. На этой машине
**ни один из трёх механизмов не срабатывает**, и GPU простаивает ~70–80% раунда декода.
Задача — включить/откалибровать их и измерить. Ожидаемый итог: decode 26 → 37–41 tok/s
без замены железа. Пять главных шагов:

| # | Что | Где | Ожидание |
|---|---|---|---|
| E1 | `--exp solo_stage=1,stage_cpu_gbps=12.6,stage_link_gbps=13.2` | флаг | +30–50% decode |
| E2 | Почему `pcie experts 0.00` при `pcie_frac 0.36` | `src/core/expert_source.cpp:2029` | +10–25% |
| E3 | `--batch 2` (амортизация + включает стейджинг) | флаг | +50–90% на пропускную способность |
| E4 | `--pool-workers 15 --pool-affinity all` (SMT) | флаг | 0…+20%, замерить |
| E5 | Rebase Ampere-порта на upstream 0.5.0 | ветка | +13–25% |
| E9 | IMMA (tensor cores) вместо `dp4a` в GPU-ядре экспертов Q2_0, по мотивам Marlin | `s2_expert_grouped.cu` / готовый MMQ | +6–9% decode, и снимает потолок с E1 |

---

## 1. Роль и цель

Ты — инженер-оптимизатор локального инференс-движка. Цель: **поднять decode tok/s на
RTX 3060 12 GB для пакета Qwen3.8-Flash-Next Q2_0 без замены железа и без потери
осмысленности вывода**, не сломав стабильность serve-режима.

Формат работы: гипотеза → минимальное изменение → A/B на одинаковых фикстурах → цифры →
решение (принять/откатить). Любое изменение без замера считается невыполненным.

---

## 2. Среда (проверено 2026-11-21)

| | |
|---|---|
| ОС | Windows 11 Pro 26300, x64 |
| GPU | RTX 3060 12 GB, sm_86, 28 SM, драйвер 621.14, CUDA UMD 13.5, power limit дефолт |
| PCIe | **3.0 x16**, замер движка `PCIe probe: 13.2 GB/s host->device` → `pcie_frac 0.36` |
| CPU | **AMD Ryzen 7 1700** (Zen 1), 8C/16T, 3.0–3.7 ГГц, **AVX-512 нет** (только AVX2+FMA+F16C) |
| RAM | 64 GiB, dual-channel DDR4 |
| Движок | `C:\Neurolib\Tools\basalt-ampere`, git HEAD `0636c6c5` (2026-10-30), база — upstream Basalt ~**v0.1.0** |
| Бинарь | `C:\Neurolib\Tools\basalt-ampere\bin\basalt.exe` (63 МБ, собран `-DCMAKE_CUDA_ARCHITECTURES=86`, MSVC 19.44 + CUDA 13.4.3) |
| Модель | `C:\Neurolib\Models\Basalt-ampere\Q2_0-v1.basalt` (70.4 ГБ: experts 31.6 GiB, dense 1.49 GiB, hc8, mtp, vision; заголовок v1 + синтетическая секция `profile`) |
| Фикстуры | `data\bench\{conv,struct,prose,prefill}.ids` (претокенизированные) |
| Готовые скрипты | `bench_headline.py`, `bench_prefill.py`, `bench_basalt_matched2.py`, `bench_stability*.py`, `bench_compare.py` |
| Отчёт по прошлым замерам | `BENCHMARK-3060.md` (обязателен к прочтению) |

Архитектура: 48 слоёв × 512 экспертов, top-10, n_embd 2560, FF 640, обученный контекст 262144.
Эксперт = 1 382 400 Б (Q2_0, 2.25 бит/вес). Кэш экспертов в VRAM — LRU+адаптивный тир.

---

## 3. Baseline, который надо воспроизвести (не принимать на веру)

Прямой прогон (одна фикстура, холодный старт):

```bat
cd /d C:\Neurolib\Tools\basalt-ampere
bin\basalt.exe generate --model C:\Neurolib\Models\Basalt-ampere\Q2_0-v1.basalt ^
  --tokens-file data\bench\conv.ids --max-new 400 --greedy --seed 42 ^
  --kv q4_0 --max-context 8192 --spec 8 --expert-cache 1500 > base-conv.log 2>&1
```

Что должно появиться в логе (порядок величин, ±10%):

```
pool multi        gate/up 60-77  quantize 0.000  down 30-39 ms/round; 9-12.6 GB/s
verify window     wait for rings 22-25  pool 89-115  host 0.2-0.7  commit 0.1 ms/round
                  CPU experts 15-17 distinct / 22-25 routed per layer
speculation       180-192 rounds of 8, drafts accepted 63-69%, 2.1-2.2 tokens per round
decode            17-18 tok/s (прямой прогон), 24-27 tok/s (serve, прогретый)
```

Ключевые инварианты, которые нельзя терять:
- `PCIe probe: ~13 GB/s` в начале лога; `FileExpertSource: ... page-locked and mapped`;
- `warning: this CPU has no AVX-512: the expert kernels run on AVX-2`;
- `loading weights ... cuda0≈6.9-10 GiB, ram≈25-31 GiB`, суммарно 35 GiB, старт 35–40 с;
- в serve на 10 запросах по 600 токенов: **10/10 успешных**, mean 26–27 tok/s.

Если baseline не воспроизводится — сначала разобраться с этим, дальше не идти.

Профиль раунда (это и есть предмет оптимизации):

| Фаза | мс | доля |
|---|---:|---:|
| CPU-пул экспертов | ~89 | 79% |
| GPU-часть + рукопожатие по кольцу | ~23 | 20% |
| хост/commit | ~0.3 | 1% |

`nvidia-smi` показывает 93–99% util — **это не загрузка**: GPU крутит spin-ядра ожидания
флага (`wait_flag_ge`), полезной работы ~20 мс из 112.

---

## 4. Диагноз (что уже известно, не переоткрывать)

1. **CPU-пул упирается в память, а не в инструкции.** 12.6 GB/s одинаковы при nt 1.2–1.4 и
   на префилле; Strata 0.1.59 на этой же машине даёт те же 12.6 GB/s в своём пуле.
   Пул читает ~1.1 ГБ/раунд (16.7 distinct × 48 слоёв × 1.38 МБ).
2. **Линк по полосе равен пулу** (13.2 vs 12.6 GB/s) и простаивает. Значит правильный ход —
   не «ускорить ядро», а распараллелить чтение экспертов между CPU и PCIe.
3. **PCIe-тир не выдаёт ни одного эксперта**: в логах `pcie experts 0.00 distinct experts per
   layer read over PCIe (share 93/256 of the misses)` — при том что `pcie_frac=0.36` посчитан
   и арена запинена. У Strata на этой же машине: **1.74–5.11** экспертов/слой.
   Код: `src/core/expert_source.cpp:2029-2084` (ветка `kd=1`).
4. **Стейджинг выключен**: `dec_moe_stage` (копия k миссов в VRAM-стейдж на отдельном потоке,
   счёт на GPU) включается только при `batch_rec_` (режим `--batch`) или
   `stage_solo = ampere_device() && exp_flag("solo_stage")` — `src/core/verify.cpp:749,2793`.
   Ёмкость: `kStagingBlobs=16` слотов × 1.38 МБ ≈ 22 МБ VRAM (`src/core/verify.hpp:618`).
5. **Модель стоимостей стейджинга не откалибрована**: `src/core/verify.cpp:763` по умолчанию
   `stage_cpu_gbps=45`, `stage_link_gbps=18`; реально 12.6 и 13.2. Оптимум при реальных
   числах — k≈8 из 16 миссов (при дефолтных — k≈4), т.е. линк недоиспользуется вдвое.
6. На 3090 авторы форка мерили этот же рычаг: **+47% generation** (56.2 → 82.7 tok/s) —
   `docs/ampere.md`, раздел «Experimental RAM staging».
7. **Профиль экспертов синтетический**: секция `profile` в пакете добавлена скриптом
   `tools\add_profile.py`, а не получена из трейса маршрутизации → стартовое размещение мусорное.
8. **На sm_86 отключены быстрые пути GPU**: `dense_tc` требует `major >= 9`
   (`src/kernels/cuda/dense_tc.cu`, дифф `83954ed3`); `hc_fused`/`mid_chain` требуют sm_90+/
   major==12 → в логе `mid_chain: device not covered (28 SMs)` и «the unfused kernels run».
   int8-MMA для экспертов в VRAM включается только при `ncols ≥ 5` (IQ3_S) / `≥ 7` (IQ4_XS),
   а у нас окно 2–3 токена → идёт dp4a (34.3 µs против 22.3 µs по A/B авторов).
9. **Форк отстал**: upstream Basalt на 2026-11-21 = **v0.5.0** (19.11). В 0.2.0 — «+13% decode»,
   в 0.4.1 — план RAM для 64-ГБ машин (#339), в 0.5.0 — per-layer expert copies.
10. Падения `verify: layer N never rang (illegal memory access)` есть и у Strata на этой же
    карте, т.е. это verify-граф, а не Ampere-порт. Следы: basalt PR #5 (`/Zc:__cplusplus` →
    misaligned `CUtensorMap` на Windows), issue #76 (peer-карта, Xid 13), issue #34.

---

## 5. Задачи по приоритету

Для каждой: **гипотеза → действие → метрика → критерий приёмки → откат**.
Все A/B — на одинаковых фикстурах, `--greedy --seed 42`, свежий процесс движка.

### E0. Пол и потолок (без этого остальное бессмысленно)

- Снять истинный GPU-пол: `--gpu-only-full` (см. `help.txt`, «THE TRUE PER-TOKEN GPU FLOOR»),
  отдельно от `--graph-only`. Записать в отчёт мс/токен.
- Замерить реальную полосу DDR4: одноразовый стенд **вне репозитория** (`C:\Neurolib\Tools\tmp\bw.c`,
  8 потоков, потоковое чтение 2 ГиБ, `cl /O2`), плюс `cudaMemcpyAsync` 1 ГиБ pinned.
  Нужно понять, 25 или 30+ GB/s потолок.
- **Приёмка:** в отчёте есть два числа (GPU-пол и потолок DRAM) и вывод, какой сплит
  CPU/PCIe оптимален арифметически (`m × 1.38 МБ / 12.6` против `k × 1.38 МБ / 13.2`).

### E1. Включить стейджинг с честной моделью стоимостей ⭐ главное

- Добавить к запуску: `--exp solo_stage=1,stage_cpu_gbps=12.6,stage_link_gbps=13.2`
  (для `serve` — те же флаги в командной строке после `--config`; проверить, что доходит).
- **Проверка, что механизм жив:** в логе появляется `basalt verify: solo staging: N+ CPU misses,
  1382400-byte slots`; в `--stats` растёт число staged, падает `pool` мс/раунд.
- **Приёмка:** decode ↑ ≥15% на `conv`+`prose`, `pool` ↓, ни одного нового падения на 10×600.
  Учесть: числа меняются (документировано «Greedy identity across window widths is unvalidated»),
  поэтому осмысленность проверять функционально (см. §7), а не побайтовым сравнением.
- **Откат:** флаг убрать. Кода не трогал — откат бесплатный.
- Свип: `stage_cpu_gbps ∈ {10, 12.6, 15}`, `stage_link_gbps ∈ {11, 13.2, 15}`, по 3 прогона.

### E2. Разобраться, почему zero-copy PCIe-тир даёт 0.00 экспертов

- Трасса: `expert_pool_dispatch` (`src/core/expert_source.cpp:1852`), ветка
  `if (d.plan != nullptr && n <= d.plan->cap)` — проверить, что `d.plan` не nullptr,
  что `staging_cap > 0`, что `d.src->pinned(layer, expert)` истинно для миссов
  (в логе `page-locked and mapped` есть, значит `complement_pinned_` true).
- Сравнить с поведением Strata 0.1.59 на этой же машине (там `pcie experts 1.74-5.11`):
  чем отличается путь — `ext_kind` из device-плана или host-план?
- **Приёмка:** либо `pcie experts > 0` и decode ↑ ≥10%, либо в отчёте — точная причина,
  почему ветка недостижима на этой конфигурации, с номерами строк.
- **Важно:** не заменять E1; это независимый (zero-copy) путь, они могут конфликтовать.

### E3. Пропускная способность: `--batch 2` (и 3)

- `--batch N` (2..8) — несколько запросов декодят вместе. Побочно включает стейджинг
  (`batch_rec_`), т.е. E3 частично пересекается с E1 — ставить оба варианта: только batch,
  batch+solo_stage.
- Стоимость: каждый слот — своя сессия; при ctx 8192 KV ≈ 0.3 ГиБ/слот (`q4_0`) →
  −220 слотов экспертов на слот (−7% hit rate).
- **Приёмка:** суммарные tok/s на 2 параллельных запросах ↑ ≥50%, при этом каждый запрос
  не деградирует больше чем на 15%; 10/10 успешных.
- Свип ctx: 4096 / 8192 / 16384 × batch 2.

### E4. CPU-пул: SMT, аффинность, число воркеров

- Сейчас `pool=7` (физические ядра минус хост), воркеры **не пинованы**
  (`--pool-affinity` влияет только на подсчёт, см. `help.txt`).
- Проверить: `--pool-workers 15 --pool-affinity all` (SMT), `--pool-workers 8`, `--pool-workers 6`.
- Отдельно: пиннинг воркеров на ядра 1..7 через `SetThreadAffinityMask` (Windows-специфика,
  планировщик сам этого не делает) — но это уже правка кода, см. E6.
- **Приёмка:** таблица GB/s в пуле по вариантам; принять лучший, если ≥+8%.

### E5. Выученный профиль экспертов вместо синтетического

- Прогнать serve с `--expert-profile-save P` (и `--adapt-every 1`), дать ему 20–30 запросов
  вашей реальной нагрузки (русский чат), затем стартовать с готовым `P`.
- **Приёмка:** hit rate 46.8% → ≥55%, `pool` мс/раунд ↓, decode ↑ ≥3%.
- Проверить, что после загрузки профиля `pcie`/`staged` доли не поехали.

### E6. Локальные правки кода (только после E1–E5)

Порядок по соотношению выигрыш/риск:

1. **Пиннинг воркеров пула** на физические ядра (`src/kernels/cpu/pool.cpp`, `worker()`):
   `SetThreadAffinityMask`/`SetThreadPriority` на Windows. Дешёво, проверяемо.
2. **Ядро Q2_0 на AVX2** (`src/kernels/cpu/q2_avx2.cpp`): сейчас распаковка 2-битных кодов —
   4×`and` + 3×`srl` + 4×`unpack` на 64 кода. Кандидаты: LUT-распаковка через `vpshufb`
   (таблица 256→4 байта), вынос `cvtepi32_ps`+`fmadd` из внутреннего цикла, обработка 2 рядов
   за проход для MLP. **Сначала подтвердить, что ядро вообще упирается** (E0 + профиль
   `perf`/`VTune`, RDTSC-таймеры вокруг `s2_expert_gu_rows_multi`).
3. **Большие страницы** для арены экспертов (26 ГиБ на 4К-страницах = 6.6М страниц):
   `MEM_LARGE_PAGES` (нужна привилегия `SeLockMemoryPrivilege`) либо
   `VirtualAlloc2` + `MEM_EXTENDED_PARAMETER`. Требует согласия владельца машины
   (изменение политики ОС).
4. **Пересборка матрицы стоимостей стейджинга из реальных замеров** (не константы
   `45`/`18`), чтобы планировщик сам выбирал k.

### E7. Крупное: rebase Ampere-порта на upstream 0.5.0

- Форк заморожен на базе ~v0.1.0. Ampere-правки сосредоточены в одном коммите `83954ed3`
  (~600 строк: гейты `__CUDA_ARCH__ >= 900`, `ampere_device()`, отключение TMA/cluster/FP4,
  диспетчеризация). Есть готовые диффы: `ampere-port.diff`, `ampere-cmake.diff` в корне форка.
- План: `git remote add upstream https://github.com/jesdga95/basalt`, ветка от `v0.5.0`,
  перенос гейтов, сборка `-DCMAKE_CUDA_ARCHITECTURES=86`, прогон тестов
  (`ctest -L cpu`, `ctest -L gpu`), затем baseline-замеры по §3.
- Мешает: 0.3+ ввёл формат model-file v2 и `profile`; наш пакет руками приведён к v1
  (`tools\add_profile.py`) — при rebase либо пересобрать пакет конвертером
  (`tools\convert\convert.py`), либо перенести чтение v1.
- **Приёмка:** decode ≥ baseline + 10%, 10/10 стабильность, `pcie`/`staged` не сломаны.
- **Риск:** высокий. Делать в отдельной ветке; ничего не пушить без разрешения.

### E8. Русский драфтер (самое дешёвое из «нового»)

- В Strata есть `--draft-vocab cyrillic` (замер: 180 → 227.6 tok/s, +26%) и `en` для карт <14 ГБ.
- У Basalt такого флага **нет**. Аналог: словарь-маска для MTP-драфтера / n-gram lookup
  (`--suffix-draft`, `src/ngram/`), ограниченная частотным списком кириллических токенов.
- Проверить косвенно: замерить долю принятых драфтов на русской фикстуре (нужна своя
  `ru.ids`) — если она заметно ниже 67%, чем на английской, рычаг есть.
- **Приёмка:** acceptance ↑ на русском, decode ↑ ≥10% на русской фикстуре.

---


### E9. IMMA вместо `dp4a` в GPU-ядре экспертов Q2_0 (по мотивам IST-DASLab/marlin)

**Гипотеза.** GPU-ядро экспертов для нашего пакета считает **только через `__dp4a`** —
`src/kernels/cuda/s2_expert_grouped.cu` (см. `expand_codes` + `chunk_s`: ~30 uops на 32 веса
на строку) и `src/kernels/cuda/dec_moe.cuh:1410`. Тензор-коровых путей для Q2_0 в декоде нет
вообще. Прикидка: 4.9 М весов на эксперта → ~23 µs на эксперт-токен против 3.8 µs, которые
нужны, чтобы просто прочитать 1.38 МБ из VRAM при 360 ГБ/с → **ядро в ~6 раз instruction-bound**.
Значит MAC'и там не бесплатны, и это ровно тот случай, под который Marlin и написан
(near-ideal 3.87× на батче 1–16 за счёт M-padding до 16 и IMMA).

**Что уже есть в дереве, чтобы не портировать Marlin.** `src/prefill/mmq/` содержит IMMA-ядра
`mma.sync.aligned.m16n8k32` для **GGML_TYPE_Q2_0** (`mmq_config.cuh:10-18`, минимальный M = 8).
То есть машинерия int8-MMA для нашего формата уже собрана и отлажена на префилле — её надо
переиспользовать в декоде с M, дополненным до 8.

**Что брать из Marlin конкретно:**

1. **M-padding вместо гейта по батчу.** Marlin работает на батче 1 за счёт дополнения M до 16.
   Наш порог MMA в `iq_kernels.cu`/`native_mmvq.cu` — `ncols ≥ 5/7`, из-за чего окно 2–3 токена
   уходит на медленный путь. Дополнить M до 8 и всегда идти в IMMA.
2. **Оффлайн-reshuffle в layout фрагментов.** Обязательное условие Marlin, и здесь главный
   конфликт: тот же 1.38 МБ блоб кормит и CPU-ядро AVX2 (`q2_avx2.cpp`, раскладка
   codes/scales), и GPU. Перекладка под MMA потребует второй копии (34 → ~68 ГБ, места нет)
   либо конвертации на лету. Вывод: **не делать оффлайн-репак**, ограничиться конверсией
   в shared-memory на лету (как сейчас), но с записью сразу в fragment-порядок.
3. **`ldmatrix.x4` + XOR-swizzled shared** — чтобы расконвертированные коды читались в
   регистры без конфликтов банков (в Marlin: `transform_a`, `ldsm4`).
4. **`createpolicy.fractional.L2::evict_first`** на загрузках весов (`cp_async4_stream`):
   эксперт читается ровно один раз, он не должен вытеснять активации/выход из L2. Одна строка,
   забирать без раздумий.
5. **Персистентная сетка `blocks = sms` + striped-партиционирование** (сейчас на слой
   запускается ~1600 мелких блоков; Marlin запускает 28 CTA и крутит цикл). Снижает churn,
   полезно в паре с 4-стадийным `cp.async`.
6. **Конфиг, подобранный под 8.6:** Marlin'овские `THREADS=256`, `STAGES=4`,
   `SHARED_MEM=96*1024` — в исходнике прямо помечено «max shared memory on compute capability 8.6».
   Наш движок тоже упирается в 101 376 Б optin — бюджет тот же, конфиг переносится как есть.

**Чего НЕ ждать.** Marlin не уменьшает число байт: он про VRAM→SM, а наш упор — DDR4 и PCIe
(1.1 ГБ/раунд никуда не денутся). Marlin — 4-битный (2-битного нет, наш пакет 2.25 бит/вес),
и он требует fp16-активаций; нам нужен int8×int8 (Q8_1), поэтому берём IMMA m16n8k32, а не
HMMA, — заодно это сохраняет точность целочисленного накопления (int8×int8→int32 точен так же,
как `dp4a`).

**Метрика и приёмка:**
- Сначала измерить долю экспертного ядра в GPU-части: `--gpu-stages`, `--stage-timing`,
  либо A/B с `--no-pool --no-capture` на фикстуре.
- Ожидание: эксперт-токен 23 µs → 6–9 µs; GPU-часть 20–23 мс → 13–17 мс; **decode +6–9%**.
- Приёмка: decode ↑ ≥5%, `--stats` не показывает роста отклонений, 10/10 стабильность,
  и — важное — **пересчитать оптимум E1**: подешевевший GPU-счёт эксперта позволяет отдать
  линку больше миссов (k ближе к 10–12 из 16).
- **Риск:** средний. Числа меняются в пределах fp32-порядка (int-суммы точны, порядок
  применения масштабов — нет); путь должен идти и в verify, и в plain decode одновременно.

**Лицензия.** Marlin — Apache-2.0 (`LICENSE`), файл `marlin/marlin_cuda_kernel.cu`
самодостаточен («no dependencies beyond base CUDA») и переносим в C++-движок; при переносе
кода — сохранить copyright-нотис. Поддерживаемая C++-версия того же ядра — в vLLM:
`csrc/quantization/gptq_marlin/marlin_template.h`.

## 6. Протокол измерений

1. **Перед серией:** `taskkill /f /im basalt.exe`, пауза 6 с, закрыть браузер/оверлеи
   (в одном прошлом логе было `only 145 MiB of VRAM free`) — иначе VRAM-шум съедает 5–10%.
2. **Каждая точка — 3 прогона**, в отчёт min/median/max. Разброс >10% → точка невалидна.
3. **Прямые прогоны** (`generate`, холодные) — для сравнения ядер; **serve** (прогретый) —
   для реального пользовательского числа. Не смешивать в одной таблице.
4. **Всегда фиксировать:** `pool multi ... GB/s`, `verify window ...`, `speculation ...`,
   `pcie experts`, `expert cache hit rate`, `drafts accepted`, VRAM peak, мощность.
5. Один вариант — один конфиг-файл, логи в `logs-speedup\<E>\<arm>-<n>.log`.
6. Итоговые числа — в `SPEEDUP-3060.md` рядом с `BENCHMARK-3060.md`, с указанием
   флагов, даты, температуры/питания.

---

## 7. Правила и ограничения

- **Не пушить, не публиковать, не релизить** ничего без явного разрешения владельца.
  Никаких `--amend` и force-push; только новые коммиты.
- **Не менять глобальные настройки машины** (реестр, power limit, MSI mode, политика
  больших страниц) без отдельного разрешения. Всё, что меняет ОС, — только с согласия.
- **Никаких облаков, загрузок моделей и «апгрейдов» драйвера/тулкита** в рамках этой задачи.
- **Стабильность важнее скорости.** Любой выигрыш, после которого serve падает чаще
  (10×600: было 10/10), не принимается.
- **Качество:** прямые прогоны на `--seed 42 --greedy` должны совпадать с baseline
  побайтно, **кроме** явно документированных lossy-путей (`solo_stage`, `--relaxed-drafting`,
  `--route-prior`). Для них — функциональная проверка: точный ответ на «напиши фразу X»,
  `{"answer": 323}` на «17 × 19», фибоначчи для 0/1/2/10/20/−1, валидный JSON на `struct`.
- **Один эксперимент — одна переменная.** Исключение: E3 (batch+solo) как отдельная явная рука.
- **Не переписывать ядро движка «заодно».** Только то, что нужно для принятой гипотезы.
- **Не писать инструменты, о которых не просили.** Одноразовые стенды — вне репозитория
  (`C:\Neurolib\Tools\tmp\`).

### Anti-goals (проверено, не тратить время)

- **Не пинить экспертов в L2**: у 3060 L2 = 3 МБ, эксперт 1.38 МБ, поток 1.1 ГБ/раунд →
  hit ≈ 0. Полезно обратное: пометить поток весов как streaming/`evict_first`.
- **Не строить UVM-тир на Windows**: managed-память нельзя выделить больше объёма VRAM,
  `cudaMemAdvise`/`cudaMemPrefetchAsync` применимы только к managed.
- **Не ждать выигрыша от TMA, thread-block clusters, DSMEM, `elect.sync`, PDL/`griddepcontrol`,
  device-side `cudaGraphLaunch`, DPX, GPUDirect Storage** — на sm_86/Windows их нет.
  TCC на GeForce не существует, только WDDM.
- **Не включать `--use_fast_math`** — меняет деквант, выигрыша на Q2_0-скалярных произведениях нет.
- **Не использовать:** `--no-pool` (пакету нужен CPU-hook `session_token`), `--adapt-swaps 0`
  (hit rate 47% → 21%), `--mmap-experts` (медленнее resident при 64 ГБ RAM).
- **Не менять пакет на IQ3** для 12 ГБ: в официальном замере на 3060 Q2_0 = 45 tok/s,
  IQ3_XXS = 26, IQ3_S = 18–21.
- **Не тащить Marlin как есть**: он 4-битный, требует fp16-активаций и оффлайн-репака
  (второй экземпляр 34 ГБ — места нет) и оптимизирован под батч 1–32; наш упор — DDR4/PCIe,
  а не VRAM→SM. Из него берём только IMMA с M-padding, `evict_first` на веса, персистентную
  сетку и smem-конфиг под 8.6 (E9).

---

## 8. Формат отчёта

Файл `SPEEDUP-3060.md` + коммиты в ветке. Обязательные разделы:

1. **Baseline воспроизведён** — таблица с числами из §3 (3 прогона, min/median/max).
2. **E0** — GPU-пол, потолок DRAM, вывод про оптимальный сплит.
3. **Таблица экспериментов:** id, что менялось, decode tok/s (min/med/max), prefill,
   pool мс/раунд, `pcie`/`staged` экспертов/слой, hit rate, стабильность n/10, вердикт.
4. **Что принято в дефолт** (конфиг + флаги) и **что откатано почему**.
5. **Открытые вопросы:** почему `pcie experts 0.00`, что показал `compute-sanitizer`
   (если запускался) по `never rang`, где именно упёрлось профилирование ядра.
6. **Одна страница «как повторить»:** точные команды.

Тон: цифры, а не «стало быстрее». Если гипотеза не подтвердилась — так и написать,
с числами и с указанием, какая часть диагноза из §4 оказалась неверной.

---

## 9. Приложение: где что лежит

| Что | Путь |
|---|---|
| Замеры прошлой сессии | `BENCHMARK-3060.md` |
| Опции движка (полный help) | `help.txt` |
| CPU-пул (протокол, режимы) | `src/kernels/cpu/pool.cpp` |
| Ядро Q2_0 AVX2 | `src/kernels/cpu/q2_avx2.cpp` |
| Ядро Q2_0 AVX-512 | `src/kernels/cpu/expert.cpp` |
| Диспетчеризация ISA | `src/kernels/cpu/expert_layout.cpp:134` |
| Планировщик экспертов / PCIe-ветка | `src/core/expert_source.cpp:2006-2099` |
| Пиннинг арены (cudaHostRegisterMapped) | `src/core/expert_source.cpp:1145-1188` |
| Стейдж-модель и `solo_stage` | `src/core/verify.cpp:744-793`, `:2793-2823` |
| GPU-ядро экспертов Q2_0 (**dp4a**) | `src/kernels/cuda/s2_expert_grouped.cu` (`expand_codes`/`chunk_s`), `src/kernels/cuda/dec_moe.cuh:1410` |
| Готовые IMMA-ядра для Q2_0 | `src/prefill/mmq/` (`mma.cuh:97` = `m16n8k32`, `mmq_config.cuh:10-18`, min M=8) |
| Константы стейджинга | `include/basalt/kernels/dec_moe.hpp:23,51`, `include/basalt/core/verify.hpp:618` |
| Гейты Ampere | `include/basalt/kernels/device_caps.hpp`, коммит `83954ed3`, `docs/ampere.md` |
| Диффы порта | `ampere-port.diff`, `ampere-cmake.diff` |
| Скрипты замеров | `bench_*.py`, `bench_direct.ps1` |

### Источники по идеям (Strata / форки / upstream)

- Strata: [HOW_IT_WORKS](https://github.com/Niko1221/Strata/blob/main/docs/HOW_IT_WORKS.md),
  [DETAILS](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md)
  (`--pcie-frac`, `--calibrate`, `--draft-vocab`, `--pool-workers`, expert profile),
  [отчёт по 3060](https://github.com/Niko1221/Strata/blob/main/docs/bench-results/2026-10-09-community-rtx-3060-0141/README.md)
  (Q2_0 → 41.8–45.3 tok/s на i7-12700).
- Basalt upstream: [v0.5.0](https://github.com/jesdga95/basalt/releases/tag/v0.5.0),
  changelog 0.2.0 (+13% decode), 0.4.1 (RAM-план для 64 ГБ).
- Падения: [PR #5](https://github.com/jesdga95/basalt/pull/5),
  [issue #76](https://github.com/jesdga95/basalt/issues/76),
  [issue #34](https://github.com/jesdga95/basalt/issues/34).
- CUDA на sm_86: [L2 cache control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html),
  [async copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html),
  [Unified Memory on Windows](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html),
  [Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html).
- Marlin (E9): [репозиторий](https://github.com/IST-DASLab/marlin) (Apache-2.0),
  [marlin_cuda_kernel.cu](https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda_kernel.cu)
  (`cp_async4_stream` с `L2::evict_first`, `dequant` через `lop3`, `ldsm4`, striped-партиционирование,
  `SHARED_MEM = 96*1024` с пометкой «max shared memory on compute capability 8.6»),
  поддерживаемая C++-версия: vLLM
  [`marlin_template.h`](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/gptq_marlin/marlin_template.h),
  статья [arXiv:2408.11743](https://arxiv.org/abs/2408.11743).
