# TradeSavvy Sandbox Starter

A minimal training repository for running the bot in T-Invest Sandbox.

## Quick Start (Windows PowerShell)

1. Clone repository:

```powershell
git clone https://github.com/DmitriiOrel/winter_school_project.git
cd .\winter_school_project
```

2. Run one-shot setup (creates `.venv`, installs deps, creates/finds sandbox account, writes `.env`):

```powershell
.\quickstart.ps1 -Token "t.<YOUR_API_TOKEN>"
```

3. Run bot:

```powershell
.\run_sandbox.ps1
```

Stop with `Ctrl+C`.

## What You Fill Manually

Only API token in `quickstart.ps1 -Token ...`.

`ACCOUNT_ID` is written automatically to `.env`.

## Core Files

- `app/main.py` - bot entrypoint.
- `instruments_config_scalpel.json` - traded instrument and strategy params.
- `tools/get_accounts.py` - sandbox accounts helper.
- `tools/get_figi.py` - FIGI by ticker.
- `tools/plot_scalpel_report.py` - backtest and charts.

## Strategy Params

In `instruments_config_scalpel.json`:

- `days_back_to_consider`
- `quantity_limit`
- `check_data`

EMA logic in code (`app/strategies/scalpel/scalpel.py`):

- `EMA_fast = 20`
- `EMA_slow = 50`

## Backtest Example

```powershell
.\.venv\Scripts\python.exe .\tools\plot_scalpel_report.py --source api --days-back 730 --interval 5min
```

## Publish Notes

- Do not commit `.env`.
- Do not commit `stats.db`, `market_data_cache`, `reports`.

## Disclaimer

Training project only. Test in sandbox before any real account use.
