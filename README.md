# TradeSavvy Sandbox Starter

Minimal training project to run a trading bot in T-Invest Sandbox.

<!-- LEADERBOARD:START -->
## Live Leaderboard

Auto-updated by backtest script. Last update: `20260219T100204Z` UTC.

| Place | Name | CAGR % | Max DD % | Trades | EMA Fast | EMA Slow | BB Window | BB Dev | TF (min) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | DmitriiOrel | 8.34 | -12.17 | 65 | 20 | 50 | 20 | 2.0 | 60 |
| 2 | smoke2 | -6.12 | -8.61 | 4 | 21 | 55 | 20 | 2.0 | 240 |
| 3 | smoke | -6.12 | -8.61 | 4 | 20 | 50 | 20 | 2.0 | 240 |

<!-- LEADERBOARD:END -->

## Quick Start (Windows PowerShell)

```powershell
git clone https://github.com/DmitriiOrel/winter_school_project.git
cd .\winter_school_project
.\quickstart.ps1 -Token "t.YOUR_API_TOKEN" -Run
```

Token format: use the raw token only (`t.xxxxx`).
Do not wrap token with `< >`.

## What quickstart does

- Creates `.venv`
- Installs dependencies
- Installs T-Invest SDK (`tinkoff.invest`)
- Creates/finds sandbox account
- Writes `.env` with `TOKEN`, `ACCOUNT_ID`, `SANDBOX=True`

## Run later

```powershell
.\run_sandbox.ps1
```

Stop: `Ctrl+C`.

## Manual Backtest + Leaderboard + Run Sandbox

Run with manual params (space-separated):

```powershell
$env:GITHUB_TOKEN="ghp_your_pat"
.\run_backtest_manual.ps1 20 50 20 2.0 60 -Name dima
```

Params order:
- `ema_fast` in `8..30`
- `ema_slow` in `35..120` and `ema_fast < ema_slow`
- `bb_window` in `10..40`
- `bb_dev` in `1.0..3.5` (step `0.25`)
- `timeframe_min` in `5, 15, 30, 60, 120, 240, 720, 1440`

Artifacts are saved into `reports/`:
- `reports/scalpel_backtest_plot.png`
- `reports/leaderboard.csv` (local mirror of GitHub leaderboard)
- `reports/trades_<name>_<run_id>.csv`
- `reports/summary_<name>_<run_id>.json`

The script:
- runs backtest for last 3 years (`1095` days);
- shows entry/exit points on chart;
- prints your leaderboard place in terminal;
- updates leaderboard on GitHub (required);
- updates leaderboard section in `README.md` on GitHub;
- keeps only one best result per participant name (`-Name`);
- writes selected EMA params into `instruments_config_scalpel.json`;
- starts sandbox bot.

GitHub leaderboard publish (mandatory):
- requires GitHub PAT (GitHub login/password is not supported);
- pass token as `-GitHubToken "<PAT>"` or set env `GITHUB_TOKEN` (recommended);
- target file by default: `DmitriiOrel/winter_school_project` -> `reports/leaderboard.csv`.

Useful flags:
- `-NoSandboxRun` - run only backtest + leaderboard.
- `-NoChartOpen` - do not auto-open chart window.

## If script execution is blocked

```powershell
powershell -ExecutionPolicy Bypass -File .\quickstart.ps1 -Token "t.YOUR_API_TOKEN" -Run
```

## Notes

- Sandbox only (`SANDBOX=True`): virtual trades.
- Do not commit `.env`, `stats.db`, `market_data_cache`, `reports`.
