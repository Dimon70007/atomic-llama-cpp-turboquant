# TASK-066.18 — Mode B: влияние каждого изменения (подробная таблица)

**Audience:** любой читатель (не только автор патчей).  
**Вопрос:** если на «голом» master включить **только одно** tip-изменение — что станет лучше / хуже и **на сколько**?  
**Policy:** RED != KEEP. «Число стало меньше» != GREEN. GREEN = бит-exact / suite PASS.

**Stamps (Terminal Metal, не agent sandbox):**

| Stamp | Что |
|-------|-----|
| [`logs/task066-18-ablation-20261109T211533Z-fullB/`](../logs/task066-18-ablation-20261109T211533Z-fullB/) | Mode B все lever’ы (старый tip; для `L_size_swa_ub` колонка **устарела**) |
| [`logs/task066-18-ablation-20261110T001626Z-modeB-size-swa/`](../logs/task066-18-ablation-20261110T001626Z-modeB-size-swa/) | Mode B alone-ON **`PAD128(n_batch)`** (актуальный product headroom) |
| Mode A tip−1 (доп. causality): [`…193709Z-fullA`](../logs/task066-18-ablation-20261109T193709Z-fullA/), tip−1 size_swa OFF [`…232638Z`](../logs/task066-17-size-swa-off-20261109T232638Z/) |

Parent pointer: [`TASK-066.18-src-lever-ablation.md`](TASK-066.18-src-lever-ablation.md).

---

## 1. Как читать эксперимент (Mode B)

```text
Шаг 0. MASTER = все tip-изменения ВЫКЛ (как upstream master в этих местах).
Шаг 1. Включить РОВНО одно tip-изменение.
Шаг 2. Прогнать те же suite’ы.
Шаг 3. Сравнить с master: GREEN/RED + числа.
```

- **Alone helps** = на master было RED, с одним изменением стало **GREEN** (это изменение само чинит suite).
- **Still RED, better** = оба RED, но ошибка **меньше** (мягче, но suite всё ещё провален).
- **Still RED, (=)** = те же числа — изменение на этот suite не влияет.
- **Alone harms** = было GREEN, стало RED (в нашей матрице **не было**).

---

## 2. Словарь чисел (что значит каждая метрика)

| Метрика | Простыми словами | Хорошо / плохо |
|---------|------------------|----------------|
| **GREEN** | Тест пройден: ответы/логиты **совпали** с эталоном пары (bit-exact или suite PASS). | Цель. |
| **RED** | Тест провален: есть расхождение. | Баг / незакрытый gate. **Нельзя** объявить KEEP. |
| **mad** (mean abs diff) | «Средняя величина ошибки» между двумя прогонами (разный `-ub` при том же `-b`). Чем ближе к **0**, тем ближе ответы. | **0** = идеально. **4.7** сильно хуже **0.25**. |
| **ulp** | Насколько «далеко» float-числа в единицах точности железа. Грубо: сколько ступенек float между эталоном и фактом. | **0** отлично; **8** маленькая дрожь; **10735** огромный разъезд. |
| **abs=…** | Макс. абсолютная разница в residual/mul-mat тестах (рядом с ulp). | Меньше = лучше. |
| **Δ (delta)** | Насколько число изменилось vs master: `новое − старое`. Отрицательный mad/ulp при RED = **лучше, но всё ещё RED**. | Смотри знак и GREEN/RED отдельно. |

**Аналогия:** GREEN = контрольная написана без ошибок. RED + меньший mad = «ошибок меньше, но двойку всё равно поставили».

### Какие suite’ы смотрим

| Suite id | Что проверяет (простыми словами) |
|----------|----------------------------------|
| `S_fa` | Flash-Attention: одинаковый ли путь/результат при разных размерах батча. |
| `S_mul_mat` | Обычное матричное умножение Metal: N=1 vs rem — численно одинаково? |
| `S_mul_mat_res` | «Хвосты» rem и политика mul_mm: насколько дрожит vs эталон N=8. |
| `S_cb_fusion` | Склейка command-buffer / fusion границ. |
| `S_moe_out` | MoE выход: row0 parity. |
| `S_iswa_unit` | CPU-юнит: формула размера SWA-кольца (без GPU модели). |
| `S_ub_c2pf16` | Prefill logits: ubatch **256 vs 512** при KV **f16/f16** (чистый численный ubatch). |
| `S_ub_c2pq8` | То же при KV **q8/q8**. |
| `S_ub_c2p` | То же при product KV **q8/turbo4** (как у оператора). |
| `S_b1_shared` | Аудит API shared-KV (есть ли символ в дереве) — не GPU-качество. |

---

## 3. Сводная таблица влияния (Mode B alone-ON)

Числа для `L_size_swa_ub` — из stamp **`…001626Z`** (`PAD128(n_batch)`).  
Остальные lever’ы — из **`…211533Z-fullB`**.

| # | Изменение (lever) | Что делает в библиотеке | Alone vs master (главное) | Насколько (числа) | Вердикт |
|---|-------------------|------------------------|---------------------------|-------------------|---------|
| 1 | `L_fa_tile` | FA всегда tile (не shape-vec) + pin nsg | **Чинит** `S_fa` | RED → **GREEN** (mad=0 на exact rows) | **KEEP** |
| 2 | `L_no_mv_ext` | Глушит `mul_mv_ext` для ne11∈[2..8] → обычный `mul_mv` | **Чинит** `S_mul_mat` | abs **2.38e-7 / ulp 5 → 0 / 0** | **KEEP** |
| 3 | `L_float_tile` | Float-tile + больший smem для q8/iq4 | Residual и ubatch **мягче**, но RED | ulp **10735 → 8** (~1340× меньше); c2pf16 mad **4.70 → 0.255**; c2pq8 **4.78 → 3.88**; c2p **8.09 → 7.63** | **KEEP soft** (не GREEN) |
| 4 | `L_size_swa_ub` = **`PAD128(n_batch)`** | Размер SWA-headroom от `-b`, **не** от `-ub` | **Чинит** product ubatch | c2pf16/c2pq8/c2p mad **4.70 / 4.78 / 8.09 → 0** (GREEN); residual **(=)**; iswa still RED | **KEEP product** |
| 5 | `L_mm_min` | Политика MUL_MM_MIN / MoE id min | Нет alone-эффекта здесь | Все смотренные suite (=) master | undecided / host |
| 6 | `L_fusion_cb` | Snap fusion_max / CB | На этом хосте cb уже GREEN на master | Нет RED→GREEN | no alone need |
| 7 | `L_moe_view_add` | MoE out всегда view+ADD | moe уже GREEN на master | Нет RED→GREEN | no alone need |
| 8 | `L_iswa_pads` | pad `n_swa` до 128 | Ubatch не чинит | mad ubatch (=) master | secondary |
| 9 | `L_shared_kv` | API shared cells | b1 всё ещё MISSING | RED → RED | needs combo |
| 10 | `L_swa_hwm_log` | Телеметрия HWM | Не suite-gate | нет влияния на PASS/FAIL | telemetry |

**Alone-harms:** нет (ни один lever не сломал GREEN suite).

---

## 4. Подробно по каждому изменению

### 4.1 `L_fa_tile` — Flash Attention always-tile

**Зачем в tip:** убрать разный FA-путь (tile vs vec) при разных размерах запроса.

| Suite | Master | Alone-ON | Смысл |
|-------|--------|----------|-------|
| `S_fa` | **RED** | **GREEN** | Одно это изменение **полностью** чинит FA suite. |
| Остальные смотренные | как master | как master | На ubatch/residual само не влияет. |

**Вывод:** без FA-tile tip **ломает** FA batch-invariance. Это обязательный KEEP.

---

### 4.2 `L_no_mv_ext` — запрет mul_mv_ext на малых ne11

**Зачем:** не прыгать между `mul_mv_ext` и `mul_mv` на rem 2..8.

| Suite | Master | Alone-ON | Смысл |
|-------|--------|----------|-------|
| `S_mul_mat` | RED abs≈**2.38e-7** ulp=**5** | **GREEN** abs=**0** ulp=**0** | Ошибка исчезла полностью. |
| Остальные | (=) | (=) | |

**Вывод:** обязательный KEEP для mul-mat invariance.

---

### 4.3 `L_float_tile` — float tiles + smem

**Зачем:** численно стабильнее matmul на q8/iq4 (меньше «дрожи»).

| Suite | Master | Alone-ON | Δ | Смысл для ребёнка |
|-------|--------|----------|---|-------------------|
| `S_mul_mat_res` | RED ulp=**10735** | RED ulp=**8** | ulp ÷ ~1340 | Было «очень далеко», стало «чуть-чуть», но тест **ещё красный**. |
| `S_ub_c2pf16` | RED mad=**4.703** | RED mad=**0.255** | Δ **−4.45** | Ошибка уменьшилась ~18×, но **не ноль**. |
| `S_ub_c2pq8` | RED mad=**4.779** | RED mad=**3.878** | Δ **−0.90** | Чуть лучше, всё ещё RED. |
| `S_ub_c2p` | RED mad=**8.094** | RED mad=**7.635** | Δ **−0.46** | Чуть лучше, всё ещё RED. |

**Вывод:** KEEP **soft** — сильно помогает качеству чисел, **один** не закрывает ubatch GREEN. Mode A tip−1: выключить float_tile → residual/ubatch **хуже** (ulp 8→10729 и т.д.).

---

### 4.4 `L_size_swa_ub` — product headroom `PAD128(n_batch)`

**Зачем:** размер SWA-кольца не должен зависеть от `-ub` при равном `-b`. Иначе 256 vs 512 ubatch считают разную физику KV → разные logits.

| Suite | Master (headroom∝ub) | Alone-ON `PAD128(n_batch)` | Δ | Смысл |
|-------|----------------------|----------------------------|---|--------|
| `S_ub_c2pf16` | RED mad=**4.703** | **GREEN mad=0** | −4.703 → exact | Prefill f16: **бит-exact**. |
| `S_ub_c2pq8` | RED mad=**4.779** | **GREEN mad=0** | −4.779 → exact | Prefill q8: **бит-exact**. |
| `S_ub_c2p` | RED mad=**8.094** | **GREEN mad=0** | −8.094 → exact | Product turbo4: **бит-exact**. |
| `S_mul_mat_res` | RED ulp=10735 | RED ulp=10735 | (=) | На matmul residual **не влияет** (и не должно). |
| `S_iswa_unit` | RED | RED | still RED | Юнит ждёт полный tip-набор формулы/pads; одного headroom мало для unit GREEN. |

**Старый tip `max(ub,b/4)` alone-ON (fullB):** ubatch mad **(=) master** — **не** чинил. Поэтому product KEEP = именно **`PAD128(n_batch)`**, не старая формула.

**Обратная проверка (Mode A tip−1 OFF @ tip с новой формулой, `…232638Z`):** ubatch GREEN→RED mad **0 → 0.255 / 3.878 / 7.635**. Выключили — сломалось. Значит рычаг настоящий.

---

### 4.5 `L_mm_min` — MUL_MM_MIN / MoE min

| Наблюдение Mode B | Смысл |
|-------------------|--------|
| Нет RED→GREEN, mad/ulp (=) | На этой матрице suite’ов **одно** это изменение ничего не чинит и не ломает. |
| Product KEEP DEFAULT=1 живёт в 066.15/066.16 (скорость vs MTP), не в этой Mode B таблице ubatch. | Смотри отдельный SSOT perf/triage. |

---

### 4.6 `L_fusion_cb` / `L_moe_view_add`

На master `S_cb_fusion` и `S_moe_out` уже **GREEN**. Alone-ON нечему «помогать» в Mode B на этом хосте.  
Не значит «удалить из tip» — значит **нет alone-доказательства** на этих suite’ах.

---

### 4.7 `L_iswa_pads`

Pad окна SWA до блока TurboQuant 128. Alone не двигает ubatch mad. Нужен вместе с корректным headroom / unit-ожиданиями. Secondary.

---

### 4.8 `L_shared_kv`

Аудит `#27781` / shared cells: alone всё ещё **MISSING**. Не GPU-mad история.

---

### 4.9 `L_swa_hwm_log`

Только логи occupancy. На PASS/FAIL suite’ов не влияет.

---

## 5. Что оставить в tip (склейка)

| Роль | Lever | Доказательство |
|------|-------|----------------|
| FA invariance | `L_fa_tile` | Mode B RED→GREEN `S_fa` |
| Mul-mat path | `L_no_mv_ext` | Mode B RED→GREEN, abs/ulp → 0 |
| Численная мягкость mm | `L_float_tile` | Mode B still-RED but ulp 10735→8, mad ubatch ↓ |
| Product ubatch bit-exact | **`PAD128(n_batch)`** | Mode B RED→GREEN mad=0 на c2pf16/c2pq8/c2p |
| Не закрыто alone | residual ulp=8 на tip | класс 066.7 / DEFAULT policy — не size_swa |

**Коротко:** четыре опоры tip для A1. Float-tile уменьшает «дрожь», size_swa делает ubatch **ноль ошибки**, FA и no_mv_ext чинят свои suite’ы начисто.

**Next (speed gates):** cumulative enable of this stack + nomtp/MTP decode floors + final suite report → [`TASK-066.19-keep-stack-speed-gates.md`](TASK-066.19-keep-stack-speed-gates.md) (floors: nomtp ≥23 tg, MTP ≥30 tg; floor miss → **NEGATIVE** hard stop, no further pillars / no Phase F; NEGATIVE ≠ KEEP).

---

## 6. Чего в этой таблице нет (чтобы не путать)

- Stress C1 16↔1024 (066.5/066.6) — другой gate, не Mode B alone.
- Long-ctx KLD / 066.10 — не Mode B.
- 066.15 скорость pp/tg — другие stamps.
- Mode A tip−1 полная матрица — зеркало («что будет, если выключить»), см. `TASK-066.18-src-lever-ablation.md`; здесь фокус Mode B «включить одно».

---

## 7. Источники цифр (fail-closed)

1. `logs/task066-18-ablation-20261109T211533Z-fullB/tables.md`  
2. `logs/task066-18-ablation-20261110T001626Z-modeB-size-swa/tables.md` + `keep_drop.md`  
3. Operator logs: `logs/run-modeB-full.log`, `logs/run-06618-modeB-size-swa-alone.log`

Если цифра не в stamp — её **нет** (не додумывать).
