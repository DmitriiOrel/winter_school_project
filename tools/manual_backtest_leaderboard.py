import argparse
import base64
import io
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import matplotlib
import pandas as pd
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands
from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.constants import INVEST_GRPC_API, INVEST_GRPC_API_SANDBOX
from tinkoff.invest.utils import now

from app.config import settings
from app.utils.quotation import quotation_to_float

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


SECONDS_IN_YEAR = 365.25 * 24 * 60 * 60
ALLOWED_TIMEFRAMES = {5, 15, 30, 60, 120, 240, 720, 1440}

BASE_TIMEFRAME_MAP = {
    5: 5,
    15: 15,
    30: 30,
    60: 60,
    120: 120,
    240: 240,
    720: 240,
    1440: 1440,
}

INTERVAL_BY_BASE_TIMEFRAME = {
    5: CandleInterval.CANDLE_INTERVAL_5_MIN,
    15: CandleInterval.CANDLE_INTERVAL_15_MIN,
    30: CandleInterval.CANDLE_INTERVAL_30_MIN,
    60: CandleInterval.CANDLE_INTERVAL_HOUR,
    120: CandleInterval.CANDLE_INTERVAL_2_HOUR,
    240: CandleInterval.CANDLE_INTERVAL_4_HOUR,
    1440: CandleInterval.CANDLE_INTERVAL_DAY,
}

LEADERBOARD_COLUMNS = [
    "place",
    "name",
    "annual_return_pct",
    "ema_fast",
    "ema_slow",
    "bb_window",
    "bb_dev",
    "timeframe_min",
    "max_drawdown_pct",
    "trades",
    "total_return_pct",
    "days_back",
    "figi",
    "timestamp_utc",
    "run_id",
]
README_LB_START = "<!-- LEADERBOARD:START -->"
README_LB_END = "<!-- LEADERBOARD:END -->"


@dataclass
class BacktestResult:
    df: pd.DataFrame
    equity: pd.Series
    trades: pd.DataFrame
    entries: list[tuple[pd.Timestamp, float]]
    exits: list[tuple[pd.Timestamp, float]]
    metrics: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual backtest for leaderboard with chart and required GitHub publish."
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--ema-fast", type=int, required=True)
    parser.add_argument("--ema-slow", type=int, required=True)
    parser.add_argument("--bb-window", type=int, required=True)
    parser.add_argument("--bb-dev", type=float, required=True)
    parser.add_argument("--timeframe-min", type=int, required=True)
    parser.add_argument("--days-back", type=int, default=1095)
    parser.add_argument("--backcandles", type=int, default=15)
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--config-path", type=Path, default=Path("instruments_config_scalpel.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--leaderboard-path", type=Path, default=Path("reports/leaderboard.csv"))
    parser.add_argument("--write-live-config", action="store_true")
    parser.add_argument("--github-owner", default="")
    parser.add_argument("--github-repo", default="")
    parser.add_argument("--github-path", default="reports/leaderboard.csv")
    parser.add_argument("--github-readme-path", default="README.md")
    parser.add_argument("--github-branch", default="main")
    parser.add_argument("--github-token", default="")
    parser.add_argument("--require-github", action="store_true")
    parser.add_argument("--table-limit", type=int, default=20)
    parser.add_argument("--readme-table-limit", type=int, default=10)
    return parser.parse_args()


def ensure_valid_params(args: argparse.Namespace) -> None:
    if not (8 <= args.ema_fast <= 30):
        raise ValueError("ema_fast must be in range 8..30")
    if not (35 <= args.ema_slow <= 120):
        raise ValueError("ema_slow must be in range 35..120")
    if args.ema_fast >= args.ema_slow:
        raise ValueError("ema_fast must be less than ema_slow")
    if not (10 <= args.bb_window <= 40):
        raise ValueError("bb_window must be in range 10..40")
    if not (1.0 <= args.bb_dev <= 3.5):
        raise ValueError("bb_dev must be in range 1.0..3.5")
    scaled = round(args.bb_dev * 100)
    if scaled % 25 != 0:
        raise ValueError("bb_dev step must be 0.25")
    if args.timeframe_min not in ALLOWED_TIMEFRAMES:
        raise ValueError(f"timeframe_min must be one of {sorted(ALLOWED_TIMEFRAMES)}")
    if args.days_back < 30:
        raise ValueError("days_back must be >= 30")


def sanitize_name(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_\-.]", "_", name.strip())
    return clean[:64] if clean else "anonymous"


def load_instrument_params(config_path: Path) -> tuple[str, dict]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    instruments = payload.get("instruments", [])
    if not instruments:
        raise ValueError("No instruments found in instruments_config_scalpel.json")
    first = instruments[0]
    figi = first.get("figi")
    if not figi:
        raise ValueError("Missing figi in config")
    params = first.get("strategy", {}).get("parameters", {})
    return figi, params


def fetch_candles(figi: str, days_back: int, timeframe_min: int) -> pd.DataFrame:
    base_tf = BASE_TIMEFRAME_MAP[timeframe_min]
    interval = INTERVAL_BY_BASE_TIMEFRAME[base_tf]
    target = INVEST_GRPC_API_SANDBOX if settings.sandbox else INVEST_GRPC_API

    print(
        f"Fetching candles: figi={figi}, days_back={days_back}, timeframe={timeframe_min}m (base {base_tf}m)",
        flush=True,
    )

    rows = []
    with Client(settings.token, target=target) as client:
        for candle in client.get_all_candles(
            figi=figi,
            from_=now() - timedelta(days=days_back),
            to=now(),
            interval=interval,
        ):
            rows.append(
                {
                    "Time": candle.time,
                    "Open": quotation_to_float(candle.open),
                    "High": quotation_to_float(candle.high),
                    "Low": quotation_to_float(candle.low),
                    "Close": quotation_to_float(candle.close),
                    "Volume": candle.volume,
                }
            )
            if len(rows) % 2000 == 0:
                print(f"Fetched candles: {len(rows)}", flush=True)

    if not rows:
        raise ValueError("No candles returned from API")

    df = pd.DataFrame(rows)
    df = df[df["High"] != df["Low"]].copy()
    df.sort_values("Time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if timeframe_min == base_tf:
        return df
    return resample_ohlcv(df, timeframe_min)


def resample_ohlcv(df: pd.DataFrame, timeframe_min: int) -> pd.DataFrame:
    frame = df.copy()
    frame["Time"] = pd.to_datetime(frame["Time"], utc=True)
    frame.set_index("Time", inplace=True)
    rule = f"{timeframe_min}min"
    agg = frame.resample(rule).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    agg.dropna(inplace=True)
    agg = agg[agg["High"] != agg["Low"]]
    agg.reset_index(inplace=True)
    print(f"Resampled candles to {timeframe_min}m: {len(agg)}", flush=True)
    return agg


def add_signals(df: pd.DataFrame, ema_fast: int, ema_slow: int, bb_window: int, bb_dev: float, backcandles: int) -> pd.DataFrame:
    out = df.copy()
    out["EMA_fast"] = EMAIndicator(close=out["Close"], window=ema_fast).ema_indicator()
    out["EMA_slow"] = EMAIndicator(close=out["Close"], window=ema_slow).ema_indicator()

    bbands = BollingerBands(close=out["Close"], window=bb_window, window_dev=bb_dev)
    out["bbihband"] = bbands.bollinger_hband_indicator()
    out["bbilband"] = bbands.bollinger_lband_indicator()

    above = out["EMA_fast"] > out["EMA_slow"]
    below = out["EMA_fast"] < out["EMA_slow"]
    above_all = above.rolling(window=backcandles).apply(lambda x: x.all(), raw=True).fillna(0).astype(bool)
    below_all = below.rolling(window=backcandles).apply(lambda x: x.all(), raw=True).fillna(0).astype(bool)

    out["EMASignal"] = 0
    out.loc[above_all, "EMASignal"] = 2
    out.loc[below_all, "EMASignal"] = 1

    buy = (out["EMASignal"] == 2) & (out["bbilband"] == 1)
    sell = (out["EMASignal"] == 1) & (out["bbihband"] == 1)
    out["TotalSignal"] = 0
    out.loc[buy, "TotalSignal"] = 2
    out.loc[sell, "TotalSignal"] = 1
    return out


def run_backtest(df: pd.DataFrame, initial_capital: float, stop_loss_percent: float) -> BacktestResult:
    cash = float(initial_capital)
    qty = 0.0
    entry_price = 0.0
    entry_time: Optional[pd.Timestamp] = None
    entry_value = 0.0

    entries: list[tuple[pd.Timestamp, float]] = []
    exits: list[tuple[pd.Timestamp, float]] = []
    trade_rows: list[dict] = []
    equity_values: list[float] = []
    equity_times: list[pd.Timestamp] = []

    for row in df.itertuples(index=False):
        ts = pd.Timestamp(row.Time)
        price = float(row.Close)
        signal = int(row.TotalSignal)

        if qty == 0.0 and signal == 2 and price > 0:
            qty = cash / price
            entry_value = cash
            cash = 0.0
            entry_price = price
            entry_time = ts
            entries.append((ts, price))
        elif qty > 0.0:
            stop_hit = price <= entry_price * (1.0 - stop_loss_percent)
            if signal == 1 or stop_hit:
                cash = qty * price
                pnl_abs = cash - entry_value
                pnl_pct = pnl_abs / entry_value if entry_value else 0.0
                exits.append((ts, price))
                trade_rows.append(
                    {
                        "entry_time": entry_time,
                        "entry_price": entry_price,
                        "exit_time": ts,
                        "exit_price": price,
                        "pnl_abs": pnl_abs,
                        "pnl_pct": pnl_pct,
                        "reason": "stop_loss" if stop_hit else "signal",
                    }
                )
                qty = 0.0
                entry_price = 0.0
                entry_time = None
                entry_value = 0.0

        equity_times.append(ts)
        equity_values.append(cash + qty * price)

    if qty > 0.0 and not df.empty:
        ts = pd.Timestamp(df.iloc[-1]["Time"])
        price = float(df.iloc[-1]["Close"])
        cash = qty * price
        pnl_abs = cash - entry_value
        pnl_pct = pnl_abs / entry_value if entry_value else 0.0
        exits.append((ts, price))
        trade_rows.append(
            {
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": ts,
                "exit_price": price,
                "pnl_abs": pnl_abs,
                "pnl_pct": pnl_pct,
                "reason": "eod",
            }
        )
        equity_values[-1] = cash

    equity = pd.Series(equity_values, index=pd.to_datetime(equity_times), dtype=float)
    if equity.empty:
        raise ValueError("Empty equity curve")

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())
    total_return = float(equity.iloc[-1] / initial_capital - 1.0)

    total_seconds = max((equity.index[-1] - equity.index[0]).total_seconds(), 24 * 60 * 60)
    years = total_seconds / SECONDS_IN_YEAR
    cagr = float((equity.iloc[-1] / initial_capital) ** (1.0 / years) - 1.0)

    trades_df = pd.DataFrame(trade_rows)
    n_trades = int(len(trades_df))
    win_rate = float((trades_df["pnl_pct"] > 0).mean()) if n_trades > 0 else 0.0

    metrics = {
        "final_equity": float(equity.iloc[-1]),
        "total_return": total_return,
        "cagr": cagr,
        "average_annual_return": cagr,
        "max_drawdown": max_dd,
        "number_of_trades": n_trades,
        "win_rate": win_rate,
    }

    return BacktestResult(
        df=df,
        equity=equity,
        trades=trades_df,
        entries=entries,
        exits=exits,
        metrics=metrics,
    )


def build_plot(result: BacktestResult, figi: str, params_text: str, output_path: Path) -> None:
    fig, (ax_price, ax_equity) = plt.subplots(
        2, 1, figsize=(18, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_price.plot(result.df["Time"], result.df["Close"], label=f"{figi} Close")
    if result.entries:
        et, ep = zip(*result.entries)
        ax_price.scatter(et, ep, marker="^", color="green", s=80, label="Buy")
    if result.exits:
        xt, xp = zip(*result.exits)
        ax_price.scatter(xt, xp, marker="v", color="red", s=80, label="Sell/Exit")
    ax_price.set_title(
        (
            f"Manual Backtest | {params_text} | "
            f"CAGR {result.metrics['cagr']*100:.2f}% | "
            f"Max DD {result.metrics['max_drawdown']*100:.2f}% | "
            f"Trades {result.metrics['number_of_trades']}"
        )
    )
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.2)
    ax_price.legend(loc="upper left")

    eq_norm = result.equity / result.equity.iloc[0]
    ax_equity.plot(eq_norm.index, eq_norm.values, color="#1f77b4")
    ax_equity.set_ylabel("Equity (x)")
    ax_equity.set_xlabel("Time")
    ax_equity.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def update_live_config(config_path: Path, ema_fast: int, ema_slow: int) -> None:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    params = payload["instruments"][0]["strategy"]["parameters"]
    params["ema_fast_window"] = ema_fast
    params["ema_slow_window"] = ema_slow
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def empty_leaderboard() -> pd.DataFrame:
    return pd.DataFrame(columns=LEADERBOARD_COLUMNS)


def ensure_leaderboard_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in LEADERBOARD_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[LEADERBOARD_COLUMNS]


def rank_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_leaderboard_columns(df)
    out["annual_return_pct"] = pd.to_numeric(out["annual_return_pct"], errors="coerce").fillna(-10_000.0)
    out["max_drawdown_pct"] = pd.to_numeric(out["max_drawdown_pct"], errors="coerce").fillna(-10_000.0)
    out["trades"] = pd.to_numeric(out["trades"], errors="coerce").fillna(0).astype(int)
    out.sort_values(
        by=["annual_return_pct", "max_drawdown_pct", "trades"],
        ascending=[False, False, False],
        inplace=True,
    )
    out.reset_index(drop=True, inplace=True)
    out["place"] = out.index + 1
    return out[LEADERBOARD_COLUMNS]


def load_local_leaderboard(path: Path) -> pd.DataFrame:
    if not path.exists():
        return empty_leaderboard()
    try:
        return pd.read_csv(path)
    except Exception:
        return empty_leaderboard()


def save_local_leaderboard(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def fetch_remote_text_file(owner: str, repo: str, path: str, branch: str, token: str) -> tuple[str, Optional[str]]:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{quote(path)}?ref={quote(branch)}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "winter-school-backtest-script",
    }
    req = Request(url, headers=headers, method="GET")
    try:
        with urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        if e.code == 404:
            return "", None
        raise
    content = payload.get("content", "")
    sha = payload.get("sha")
    decoded = base64.b64decode(content).decode("utf-8") if content else ""
    return decoded, sha


def fetch_remote_leaderboard(owner: str, repo: str, path: str, branch: str, token: str) -> tuple[pd.DataFrame, Optional[str]]:
    decoded, sha = fetch_remote_text_file(owner=owner, repo=repo, path=path, branch=branch, token=token)
    if not decoded.strip():
        return empty_leaderboard(), sha
    return pd.read_csv(io.StringIO(decoded)), sha


def push_remote_text_file(
    owner: str,
    repo: str,
    path: str,
    branch: str,
    token: str,
    text: str,
    sha: Optional[str],
    message: str,
) -> None:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{quote(path)}"
    body = {
        "message": message,
        "content": base64.b64encode(text.encode("utf-8")).decode("utf-8"),
        "branch": branch,
    }
    if sha:
        body["sha"] = sha
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "winter-school-backtest-script",
        "Content-Type": "application/json",
    }
    req = Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="PUT")
    with urlopen(req, timeout=30):
        pass


def push_remote_leaderboard(owner: str, repo: str, path: str, branch: str, token: str, df: pd.DataFrame, sha: Optional[str], message: str) -> None:
    push_remote_text_file(
        owner=owner,
        repo=repo,
        path=path,
        branch=branch,
        token=token,
        text=df.to_csv(index=False),
        sha=sha,
        message=message,
    )


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def render_readme_leaderboard(df: pd.DataFrame, table_limit: int, generated_utc: str) -> str:
    cols = ["place", "name", "annual_return_pct", "max_drawdown_pct", "trades", "ema_fast", "ema_slow", "bb_window", "bb_dev", "timeframe_min"]
    top = df[cols].head(table_limit).copy()
    top["annual_return_pct"] = top["annual_return_pct"].map(lambda x: f"{float(x):.2f}")
    top["max_drawdown_pct"] = top["max_drawdown_pct"].map(lambda x: f"{float(x):.2f}")
    top["trades"] = top["trades"].map(lambda x: f"{int(x)}")
    lines = [
        README_LB_START,
        "## Live Leaderboard",
        "",
        f"Auto-updated by backtest script. Last update: `{generated_utc}` UTC.",
        "",
        "| Place | Name | CAGR % | Max DD % | Trades | EMA Fast | EMA Slow | BB Window | BB Dev | TF (min) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if top.empty:
        lines.append("| - | - | - | - | - | - | - | - | - | - |")
    else:
        for _, row in top.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md_cell(row["place"]),
                        _md_cell(row["name"]),
                        _md_cell(row["annual_return_pct"]),
                        _md_cell(row["max_drawdown_pct"]),
                        _md_cell(row["trades"]),
                        _md_cell(row["ema_fast"]),
                        _md_cell(row["ema_slow"]),
                        _md_cell(row["bb_window"]),
                        _md_cell(row["bb_dev"]),
                        _md_cell(row["timeframe_min"]),
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append(README_LB_END)
    return "\n".join(lines)


def inject_readme_leaderboard(readme_text: str, leaderboard_block: str) -> str:
    pattern = re.compile(rf"{re.escape(README_LB_START)}.*?{re.escape(README_LB_END)}", flags=re.S)
    matches = list(pattern.finditer(readme_text))
    if matches:
        first = matches[0]
        prefix = readme_text[: first.start()]
        suffix = readme_text[first.end() :]
        suffix_without_duplicates = pattern.sub("", suffix)
        return f"{prefix}{leaderboard_block}{suffix_without_duplicates}"

    suffix = "" if readme_text.endswith("\n") else "\n"
    return f"{readme_text}{suffix}\n{leaderboard_block}\n"


def print_leaderboard(df: pd.DataFrame, table_limit: int) -> None:
    cols = ["place", "name", "annual_return_pct", "ema_fast", "ema_slow", "bb_window", "bb_dev", "timeframe_min"]
    top = df[cols].head(table_limit).copy()
    top["annual_return_pct"] = top["annual_return_pct"].map(lambda x: f"{float(x):.2f}%")
    print("\n=== Leaderboard ===")
    print(top.to_string(index=False))


def main() -> None:
    args = parse_args()
    ensure_valid_params(args)
    if not args.require_github:
        raise ValueError("Local leaderboard mode is disabled. Run with --require-github.")
    if not args.github_token:
        args.github_token = os.getenv("GITHUB_TOKEN", "")
    if not args.github_owner or not args.github_repo or not args.github_token:
        raise ValueError(
            "GitHub leaderboard requires --github-owner, --github-repo and --github-token "
            "(or GITHUB_TOKEN env variable)."
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    figi, cfg_params = load_instrument_params(args.config_path)
    stop_loss_percent = float(cfg_params.get("stop_loss_percent", 0.05))
    safe_name = sanitize_name(args.name)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    candles = fetch_candles(figi=figi, days_back=args.days_back, timeframe_min=args.timeframe_min)
    print(f"Candles ready: {len(candles)}", flush=True)

    with_signals = add_signals(
        candles,
        ema_fast=args.ema_fast,
        ema_slow=args.ema_slow,
        bb_window=args.bb_window,
        bb_dev=args.bb_dev,
        backcandles=args.backcandles,
    )

    result = run_backtest(
        with_signals,
        initial_capital=args.initial_capital,
        stop_loss_percent=stop_loss_percent,
    )

    params_text = (
        f"EMA {args.ema_fast}/{args.ema_slow} | BB {args.bb_window},{args.bb_dev} | TF {args.timeframe_min}m"
    )
    unique_plot = output_dir / f"backtest_{safe_name}_{run_id}.png"
    latest_plot = output_dir / "scalpel_backtest_plot.png"
    build_plot(result, figi=figi, params_text=params_text, output_path=unique_plot)
    build_plot(result, figi=figi, params_text=params_text, output_path=latest_plot)

    trades_path = output_dir / f"trades_{safe_name}_{run_id}.csv"
    result.trades.to_csv(trades_path, index=False)

    summary = {
        "run_id": run_id,
        "name": safe_name,
        "figi": figi,
        "days_back": args.days_back,
        "timeframe_min": args.timeframe_min,
        "ema_fast": args.ema_fast,
        "ema_slow": args.ema_slow,
        "bb_window": args.bb_window,
        "bb_dev": args.bb_dev,
        "annual_return_pct": result.metrics["cagr"] * 100.0,
        "total_return_pct": result.metrics["total_return"] * 100.0,
        "max_drawdown_pct": result.metrics["max_drawdown"] * 100.0,
        "trades": int(result.metrics["number_of_trades"]),
        "win_rate_pct": result.metrics["win_rate"] * 100.0,
        "plot_png": str(unique_plot),
        "latest_plot_png": str(latest_plot),
        "trades_csv": str(trades_path),
    }
    summary_path = output_dir / f"summary_{safe_name}_{run_id}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    record = {
        "place": 0,
        "name": safe_name,
        "annual_return_pct": summary["annual_return_pct"],
        "ema_fast": args.ema_fast,
        "ema_slow": args.ema_slow,
        "bb_window": args.bb_window,
        "bb_dev": args.bb_dev,
        "timeframe_min": args.timeframe_min,
        "max_drawdown_pct": summary["max_drawdown_pct"],
        "trades": summary["trades"],
        "total_return_pct": summary["total_return_pct"],
        "days_back": args.days_back,
        "figi": figi,
        "timestamp_utc": run_id,
        "run_id": run_id,
    }

    try:
        remote_df, remote_sha = fetch_remote_leaderboard(
            owner=args.github_owner,
            repo=args.github_repo,
            path=args.github_path,
            branch=args.github_branch,
            token=args.github_token,
        )
        remote_df = ensure_leaderboard_columns(remote_df)
        if remote_df.empty:
            merged = pd.DataFrame([record], columns=LEADERBOARD_COLUMNS)
        else:
            merged = pd.concat([remote_df, pd.DataFrame([record])], ignore_index=True)
        ranked = rank_leaderboard(merged)
        push_remote_leaderboard(
            owner=args.github_owner,
            repo=args.github_repo,
            path=args.github_path,
            branch=args.github_branch,
            token=args.github_token,
            df=ranked,
            sha=remote_sha,
            message=f"Update leaderboard: {safe_name} {run_id}",
        )

        readme_text, readme_sha = fetch_remote_text_file(
            owner=args.github_owner,
            repo=args.github_repo,
            path=args.github_readme_path,
            branch=args.github_branch,
            token=args.github_token,
        )
        leaderboard_block = render_readme_leaderboard(
            df=ranked,
            table_limit=args.readme_table_limit,
            generated_utc=run_id,
        )
        updated_readme = inject_readme_leaderboard(readme_text=readme_text, leaderboard_block=leaderboard_block)
        if updated_readme != readme_text:
            push_remote_text_file(
                owner=args.github_owner,
                repo=args.github_repo,
                path=args.github_readme_path,
                branch=args.github_branch,
                token=args.github_token,
                text=updated_readme,
                sha=readme_sha,
                message=f"Update README leaderboard: {safe_name} {run_id}",
            )
    except Exception as ex:
        raise RuntimeError(f"GitHub leaderboard update failed: {ex}") from ex

    # Keep a local mirror for convenience, but source of truth is GitHub.
    save_local_leaderboard(ranked, args.leaderboard_path)

    my_row = ranked[ranked["run_id"] == run_id]
    my_place = int(my_row.iloc[0]["place"]) if not my_row.empty else -1

    if args.write_live_config:
        update_live_config(args.config_path, args.ema_fast, args.ema_slow)

    print("\n=== Backtest Summary ===")
    print(f"Name: {safe_name}")
    print(f"FIGI: {figi}")
    print(f"Params: {params_text}")
    print(f"Average annual return (CAGR): {summary['annual_return_pct']:.2f}%")
    print(f"Total return: {summary['total_return_pct']:.2f}%")
    print(f"Max drawdown: {summary['max_drawdown_pct']:.2f}%")
    print(f"Trades: {summary['trades']}")
    print(f"Your leaderboard place: {my_place}")
    print(f"Latest plot: {latest_plot}")
    print(f"Leaderboard file: {args.leaderboard_path}")
    print(f"README leaderboard path: {args.github_readme_path}")
    print("Published to GitHub: yes")

    print_leaderboard(ranked, args.table_limit)


if __name__ == "__main__":
    main()
