# TradeSavvy Sandbox Starter

Учебный репозиторий для запуска торгового бота в песочнице T-Invest, проведения бэктестов и ведения общего лидерборда на GitHub.

<!-- LEADERBOARD:START -->
## Актуальный Лидерборд

Автоматически обновляется после каждого бэктеста. Последнее обновление: `20260219T104200Z` UTC.

| Место | Участник | CAGR % | Макс. просадка % | Сделки | EMA Fast | EMA Slow | BB Window | BB Dev | ТФ (мин) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | DmitriiOrel | 8.34 | -12.17 | 65 | 20 | 50 | 20 | 2.0 | 60 |
| 2 | dmitrii | 8.22 | -12.17 | 65 | 20 | 50 | 20 | 2.0 | 60 |
| 3 | smoke_ru | -0.80 | -2.56 | 1 | 20 | 50 | 20 | 2.0 | 240 |
| 4 | smoke2 | -6.12 | -8.61 | 4 | 21 | 55 | 20 | 2.0 | 240 |
| 5 | smoke | -6.12 | -8.61 | 4 | 20 | 50 | 20 | 2.0 | 240 |

<!-- LEADERBOARD:END -->

## Быстрый Старт (Windows PowerShell)

```powershell
git clone https://github.com/DmitriiOrel/winter_school_project.git
cd .\winter_school_project
.\quickstart.ps1 -Token "t.ВАШ_API_ТОКЕН"
```

Токен указывается в сыром виде: `t.xxxxx` (без `< >` и без пробелов).

## Что Делает quickstart

- создает `.venv`;
- устанавливает зависимости;
- устанавливает SDK T-Invest (`tinkoff.invest`);
- создает или находит sandbox-счет;
- записывает `.env` с полями `TOKEN`, `ACCOUNT_ID`, `SANDBOX=True`.

## Единый Сценарий Запуска

В проекте используется один основной скрипт: `run_backtest_manual.ps1`.

Скрипт выполняет полный цикл:
1. ручной бэктест за 3 года;
2. обновление лидерборда в GitHub (режим `best-only`: сохраняется только лучший результат участника);
3. обновление таблицы лидерборда в `README.md`;
4. сохранение артефактов trial в `reports/<user>/trial_<run_id>`;
5. обязательный запуск sandbox-бота.

### Команда запуска

```powershell
$env:GITHUB_TOKEN="github_pat_ВАШ_PAT"
.\run_backtest_manual.ps1 20 50 20 2.0 60 -Name dmitrii
```

Порядок параметров:
- `ema_fast` в диапазоне `8..30`;
- `ema_slow` в диапазоне `35..120`, при этом `ema_fast < ema_slow`;
- `bb_window` в диапазоне `10..40`;
- `bb_dev` в диапазоне `1.0..3.5` с шагом `0.25`;
- `timeframe_min` один из: `5, 15, 30, 60, 120, 240, 720, 1440`.

## Структура Артефактов

### Лидерборд

- Основной файл лидерборда в GitHub: `reports/leaderboard.json`.
- Формат: JSON-массив записей.
- Логика отбора: `best-only` по полю участника `name`.

### Артефакты участника

Для каждого участника формируется папка:
- `reports/<user>/trials_index.json` — индекс запусков, отсортированный по доходности;
- `reports/<user>/trial_<run_id>/summary.json` — параметры и метрики стратегии;
- `reports/<user>/trial_<run_id>/backtest.png` — график бэктеста;
- `reports/<user>/trial_<run_id>/trades.csv` — журнал сделок.

## Требования к GitHub

Для публикации нужен GitHub PAT с правами:
- репозиторий: `DmitriiOrel/winter_school_project`;
- permission: `Contents: Read and write`.

Логин/пароль GitHub не поддерживаются.

## Если Скрипты Блокируются Политикой ExecutionPolicy

```powershell
powershell -ExecutionPolicy Bypass -File .\quickstart.ps1 -Token "t.ВАШ_API_ТОКЕН"
```

## Примечания

- Режим торговли: только песочница (`SANDBOX=True`).
- Не коммитьте `.env`, `market_data_cache`, `stats.db`.
