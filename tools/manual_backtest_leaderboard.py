import argparse
import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
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
GITHUB_TIMEOUT_SEC = 90
GITHUB_MAX_RETRIES = 5
GITHUB_RETRY_BASE_SEC = 2.0


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
        description="Ручной бэктест стратегии, обновление лидерборда и публикация артефактов на GitHub."
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
    parser.add_argument("--leaderboard-path", type=Path, default=Path("reports/leaderboard.json"))
    parser.add_argument("--write-live-config", action="store_true")
    parser.add_argument("--github-owner", default="")
    parser.add_argument("--github-repo", default="")
    parser.add_argument("--github-path", default="reports/leaderboard.json")
    parser.add_argument("--github-readme-path", default="README.md")
    parser.add_argument("--github-branch", default="main")
    parser.add_argument("--github-token", default="")
    parser.add_argument("--require-github", action="store_true")
    parser.add_argument("--table-limit", type=int, default=20)
    parser.add_argument("--readme-table-limit", type=int, default=10)
    return parser.parse_args()


def ensure_valid_params(args: argparse.Namespace) -> None:
    if not (8 <= args.ema_fast <= 30):
        raise ValueError("Параметр ema_fast должен быть в диапазоне 8..30")
    if not (35 <= args.ema_slow <= 120):
        raise ValueError("Параметр ema_slow должен быть в диапазоне 35..120")
    if args.ema_fast >= args.ema_slow:
        raise ValueError("Должно выполняться условие: ema_fast < ema_slow")
    if not (10 <= args.bb_window <= 40):
        raise ValueError("Параметр bb_window должен быть в диапазоне 10..40")
    if not (1.0 <= args.bb_dev <= 3.5):
        raise ValueError("Параметр bb_dev должен быть в диапазоне 1.0..3.5")
    scaled = round(args.bb_dev * 100)
    if scaled % 25 != 0:
        raise ValueError("Шаг параметра bb_dev должен быть 0.25")
    if args.timeframe_min not in ALLOWED_TIMEFRAMES:
        raise ValueError(f"Параметр timeframe_min должен быть одним из {sorted(ALLOWED_TIMEFRAMES)}")
    if args.days_back < 30:
        raise ValueError("Параметр days_back должен быть >= 30")


def sanitize_name(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_\-.]", "_", name.strip())
    return clean[:64] if clean else "anonymous"


def load_instrument_params(config_path: Path) -> tuple[str, dict]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    instruments = payload.get("instruments", [])
    if not instruments:
        raise ValueError("В файле instruments_config_scalpel.json не найдены инструменты")
    first = instruments[0]
    figi = first.get("figi")
    if not figi:
        raise ValueError("В конфигурации отсутствует поле figi")
    params = first.get("strategy", {}).get("parameters", {})
    return figi, params


def fetch_candles(figi: str, days_back: int, timeframe_min: int) -> pd.DataFrame:
    base_tf = BASE_TIMEFRAME_MAP[timeframe_min]
    interval = INTERVAL_BY_BASE_TIMEFRAME[base_tf]
    target = INVEST_GRPC_API_SANDBOX if settings.sandbox else INVEST_GRPC_API

    print(
        f"Загрузка свечей: figi={figi}, days_back={days_back}, timeframe={timeframe_min}m (базовый {base_tf}m)",
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
                print(f"Загружено свечей: {len(rows)}", flush=True)

    if not rows:
        raise ValueError("API не вернул свечи по заданным параметрам")

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
    print(f"Ресемплинг свечей в {timeframe_min}m: {len(agg)}", flush=True)
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
        raise ValueError("Кривая капитала пустая: невозможно рассчитать метрики")

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


def update_live_config(
    config_path: Path,
    ema_fast: int,
    ema_slow: int,
    bb_window: int,
    bb_dev: float,
    timeframe_min: int,
    backcandles: int,
) -> None:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    params = payload["instruments"][0]["strategy"]["parameters"]
    params["ema_fast_window"] = ema_fast
    params["ema_slow_window"] = ema_slow
    params["bb_window"] = bb_window
    params["bb_dev"] = bb_dev
    params["timeframe_min"] = timeframe_min
    params["backcandles"] = backcandles
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
    out["name"] = out["name"].astype(str).str.strip()
    out["name_norm"] = out["name"].str.lower()
    out["annual_return_pct"] = pd.to_numeric(out["annual_return_pct"], errors="coerce").fillna(-10_000.0)
    out["max_drawdown_pct"] = pd.to_numeric(out["max_drawdown_pct"], errors="coerce").fillna(-10_000.0)
    out["trades"] = pd.to_numeric(out["trades"], errors="coerce").fillna(0).astype(int)
    out["timestamp_utc"] = out["timestamp_utc"].astype(str)
    out.sort_values(
        by=["annual_return_pct", "max_drawdown_pct", "trades", "timestamp_utc"],
        ascending=[False, False, False, False],
        inplace=True,
    )
    # Best-only mode: keep one best row per participant name.
    out = out.drop_duplicates(subset=["name_norm"], keep="first")
    out.drop(columns=["name_norm"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    out["place"] = out.index + 1
    return out[LEADERBOARD_COLUMNS]


def load_local_leaderboard(path: Path) -> pd.DataFrame:
    if not path.exists():
        return empty_leaderboard()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return ensure_leaderboard_columns(pd.DataFrame(payload))
        return empty_leaderboard()
    except Exception:
        return empty_leaderboard()


def save_local_leaderboard(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = ensure_leaderboard_columns(df).to_dict(orient="records")
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _should_retry_http(code: int) -> bool:
    return code in {408, 409, 423, 425, 429, 500, 502, 503, 504}


def github_api_request(
    url: str,
    method: str,
    headers: dict,
    data: Optional[bytes] = None,
    timeout: int = GITHUB_TIMEOUT_SEC,
) -> bytes:
    last_error: Optional[Exception] = None
    for attempt in range(1, GITHUB_MAX_RETRIES + 1):
        req = Request(url, headers=headers, data=data, method=method)
        try:
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except HTTPError as ex:
            last_error = ex
            if attempt < GITHUB_MAX_RETRIES and _should_retry_http(ex.code):
                sleep_s = GITHUB_RETRY_BASE_SEC * attempt
                print(
                    f"[GitHub] HTTP {ex.code}, retry {attempt}/{GITHUB_MAX_RETRIES} in {sleep_s:.1f}s",
                    flush=True,
                )
                time.sleep(sleep_s)
                continue
            raise
        except (URLError, TimeoutError) as ex:
            last_error = ex
            if attempt < GITHUB_MAX_RETRIES:
                sleep_s = GITHUB_RETRY_BASE_SEC * attempt
                print(
                    f"[GitHub] Network error, retry {attempt}/{GITHUB_MAX_RETRIES} in {sleep_s:.1f}s: {ex}",
                    flush=True,
                )
                time.sleep(sleep_s)
                continue
            raise
    raise RuntimeError(f"GitHub request failed: {last_error}")


def fetch_remote_file_bytes(owner: str, repo: str, path: str, branch: str, token: str) -> tuple[bytes, Optional[str]]:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{quote(path)}?ref={quote(branch)}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "winter-school-backtest-script",
    }
    try:
        payload = json.loads(github_api_request(url=url, method="GET", headers=headers).decode("utf-8"))
    except HTTPError as e:
        if e.code == 404:
            return b"", None
        raise
    content = payload.get("content", "")
    sha = payload.get("sha")
    decoded = base64.b64decode(content) if content else b""
    return decoded, sha


def fetch_remote_text_file(owner: str, repo: str, path: str, branch: str, token: str) -> tuple[str, Optional[str]]:
    decoded, sha = fetch_remote_file_bytes(owner=owner, repo=repo, path=path, branch=branch, token=token)
    if not decoded:
        return "", sha
    return decoded.decode("utf-8"), sha


def fetch_remote_leaderboard(owner: str, repo: str, path: str, branch: str, token: str) -> tuple[pd.DataFrame, Optional[str]]:
    decoded, sha = fetch_remote_text_file(owner=owner, repo=repo, path=path, branch=branch, token=token)
    if not decoded.strip():
        return empty_leaderboard(), sha
    payload = json.loads(decoded)
    if not isinstance(payload, list):
        return empty_leaderboard(), sha
    return ensure_leaderboard_columns(pd.DataFrame(payload)), sha


def push_remote_bytes_file(
    owner: str,
    repo: str,
    path: str,
    branch: str,
    token: str,
    content: bytes,
    sha: Optional[str],
    message: str,
) -> None:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{quote(path)}"
    body = {
        "message": message,
        "content": base64.b64encode(content).decode("utf-8"),
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
    github_api_request(url=url, method="PUT", headers=headers, data=json.dumps(body).encode("utf-8"))


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
    push_remote_bytes_file(
        owner=owner,
        repo=repo,
        path=path,
        branch=branch,
        token=token,
        content=text.encode("utf-8"),
        sha=sha,
        message=message,
    )


def push_remote_leaderboard(owner: str, repo: str, path: str, branch: str, token: str, df: pd.DataFrame, sha: Optional[str], message: str) -> None:
    payload = ensure_leaderboard_columns(df).to_dict(orient="records")
    push_remote_text_file(
        owner=owner,
        repo=repo,
        path=path,
        branch=branch,
        token=token,
        text=json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        sha=sha,
        message=message,
    )


def delete_remote_file(owner: str, repo: str, path: str, branch: str, token: str, sha: str, message: str) -> None:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{quote(path)}"
    body = {
        "message": message,
        "sha": sha,
        "branch": branch,
    }
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "winter-school-backtest-script",
        "Content-Type": "application/json",
    }
    github_api_request(url=url, method="DELETE", headers=headers, data=json.dumps(body).encode("utf-8"))


def load_trials_index(payload_text: str) -> list[dict]:
    if not payload_text.strip():
        return []
    try:
        payload = json.loads(payload_text)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return payload


def sort_trials_index(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("annual_return_pct", -10_000.0)),
            float(row.get("max_drawdown_pct", -10_000.0)),
            str(row.get("run_id", "")),
        ),
        reverse=True,
    )


def push_local_file_to_github(owner: str, repo: str, branch: str, token: str, remote_path: str, local_path: Path, message: str) -> None:
    push_remote_bytes_file(
        owner=owner,
        repo=repo,
        path=remote_path,
        branch=branch,
        token=token,
        content=local_path.read_bytes(),
        sha=None,
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
        "## Актуальный Лидерборд",
        "",
        f"Автоматически обновляется после каждого бэктеста. Последнее обновление: `{generated_utc}` UTC.",
        "",
        "| Место | Участник | CAGR % | Макс. просадка % | Сделки | EMA Fast | EMA Slow | BB Window | BB Dev | ТФ (мин) |",
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
    print("\n=== Лидерборд ===")
    print(top.to_string(index=False))


def main() -> None:
    args = parse_args()
    ensure_valid_params(args)
    if not args.require_github:
        raise ValueError("Локальный режим лидерборда отключен. Используйте флаг --require-github.")
    if not args.github_token:
        args.github_token = os.getenv("GITHUB_TOKEN", "")
    if not args.github_owner or not args.github_repo or not args.github_token:
        raise ValueError(
            "Для публикации лидерборда в GitHub необходимы --github-owner, --github-repo и --github-token "
            "(или переменная окружения GITHUB_TOKEN)."
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    figi, cfg_params = load_instrument_params(args.config_path)
    stop_loss_percent = float(cfg_params.get("stop_loss_percent", 0.05))
    safe_name = sanitize_name(args.name)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    candles = fetch_candles(figi=figi, days_back=args.days_back, timeframe_min=args.timeframe_min)
    print(f"Свечи готовы: {len(candles)}", flush=True)

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
    user_dir = output_dir / safe_name
    trial_dir = user_dir / f"trial_{run_id}"
    user_dir.mkdir(parents=True, exist_ok=True)
    trial_dir.mkdir(parents=True, exist_ok=True)

    unique_plot = trial_dir / "backtest.png"
    latest_plot = output_dir / "scalpel_backtest_plot.png"
    build_plot(result, figi=figi, params_text=params_text, output_path=unique_plot)
    build_plot(result, figi=figi, params_text=params_text, output_path=latest_plot)

    trades_path = trial_dir / "trades.csv"
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
        "strategy_params": {
            "ema_fast": args.ema_fast,
            "ema_slow": args.ema_slow,
            "bb_window": args.bb_window,
            "bb_dev": args.bb_dev,
            "timeframe_min": args.timeframe_min,
            "backcandles": args.backcandles,
        },
        "trial_dir": str(trial_dir),
        "plot_png": str(unique_plot),
        "trades_csv": str(trades_path),
    }
    summary_path = trial_dir / "summary.json"
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
        if remote_df.empty and args.github_path.endswith(".json"):
            legacy_text, _ = fetch_remote_text_file(
                owner=args.github_owner,
                repo=args.github_repo,
                path="reports/leaderboard.csv",
                branch=args.github_branch,
                token=args.github_token,
            )
            if legacy_text.strip():
                try:
                    remote_df = ensure_leaderboard_columns(pd.read_csv(io.StringIO(legacy_text)))
                except Exception:
                    remote_df = empty_leaderboard()
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
            message=f"Обновление лидерборда: {safe_name} {run_id}",
        )

        # Миграция: удаляем устаревший CSV-лидерборд, если он существует в репозитории.
        _, legacy_csv_sha = fetch_remote_file_bytes(
            owner=args.github_owner,
            repo=args.github_repo,
            path="reports/leaderboard.csv",
            branch=args.github_branch,
            token=args.github_token,
        )
        if legacy_csv_sha:
            delete_remote_file(
                owner=args.github_owner,
                repo=args.github_repo,
                path="reports/leaderboard.csv",
                branch=args.github_branch,
                token=args.github_token,
                sha=legacy_csv_sha,
                message="Удаление устаревшего файла reports/leaderboard.csv",
            )

        trial_remote_root = f"reports/{safe_name}/trial_{run_id}"
        push_local_file_to_github(
            owner=args.github_owner,
            repo=args.github_repo,
            branch=args.github_branch,
            token=args.github_token,
            remote_path=f"{trial_remote_root}/summary.json",
            local_path=summary_path,
            message=f"Добавление артефактов trial: {safe_name} {run_id}",
        )
        push_local_file_to_github(
            owner=args.github_owner,
            repo=args.github_repo,
            branch=args.github_branch,
            token=args.github_token,
            remote_path=f"{trial_remote_root}/backtest.png",
            local_path=unique_plot,
            message=f"Добавление артефактов trial: {safe_name} {run_id}",
        )
        push_local_file_to_github(
            owner=args.github_owner,
            repo=args.github_repo,
            branch=args.github_branch,
            token=args.github_token,
            remote_path=f"{trial_remote_root}/trades.csv",
            local_path=trades_path,
            message=f"Добавление артефактов trial: {safe_name} {run_id}",
        )

        index_path_remote = f"reports/{safe_name}/trials_index.json"
        index_text, index_sha = fetch_remote_text_file(
            owner=args.github_owner,
            repo=args.github_repo,
            path=index_path_remote,
            branch=args.github_branch,
            token=args.github_token,
        )
        index_rows = load_trials_index(index_text)
        index_rows.append(
            {
                "run_id": run_id,
                "trial_path": trial_remote_root,
                "annual_return_pct": summary["annual_return_pct"],
                "max_drawdown_pct": summary["max_drawdown_pct"],
                "total_return_pct": summary["total_return_pct"],
                "trades": summary["trades"],
                "ema_fast": args.ema_fast,
                "ema_slow": args.ema_slow,
                "bb_window": args.bb_window,
                "bb_dev": args.bb_dev,
                "timeframe_min": args.timeframe_min,
                "timestamp_utc": run_id,
            }
        )
        index_rows = sort_trials_index(index_rows)
        (user_dir / "trials_index.json").write_text(
            json.dumps(index_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        push_remote_text_file(
            owner=args.github_owner,
            repo=args.github_repo,
            path=index_path_remote,
            branch=args.github_branch,
            token=args.github_token,
            text=json.dumps(index_rows, ensure_ascii=False, indent=2) + "\n",
            sha=index_sha,
            message=f"Обновление индекса trial: {safe_name} {run_id}",
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
                message=f"Обновление README-лидерборда: {safe_name} {run_id}",
            )
    except Exception as ex:
        raise RuntimeError(f"Ошибка обновления данных в GitHub: {ex}") from ex

    # Локальное зеркало для удобства, источник истины - GitHub.
    save_local_leaderboard(ranked, args.leaderboard_path)

    current_run_kept = not ranked[ranked["run_id"] == run_id].empty
    my_row = ranked[ranked["run_id"] == run_id]
    if my_row.empty:
        my_row = ranked[ranked["name"].astype(str).str.lower() == safe_name.lower()].head(1)
    my_place = int(my_row.iloc[0]["place"]) if not my_row.empty else -1

    if args.write_live_config:
        update_live_config(
            config_path=args.config_path,
            ema_fast=args.ema_fast,
            ema_slow=args.ema_slow,
            bb_window=args.bb_window,
            bb_dev=args.bb_dev,
            timeframe_min=args.timeframe_min,
            backcandles=args.backcandles,
        )

    print("\n=== Сводка Бэктеста ===")
    print(f"Участник: {safe_name}")
    print(f"FIGI: {figi}")
    print(f"Параметры: {params_text}")
    print(f"Среднегодовая доходность (CAGR): {summary['annual_return_pct']:.2f}%")
    print(f"Итоговая доходность: {summary['total_return_pct']:.2f}%")
    print(f"Максимальная просадка: {summary['max_drawdown_pct']:.2f}%")
    print(f"Количество сделок: {summary['trades']}")
    print(f"Текущий прогон сохранен в лидерборде: {'да' if current_run_kept else 'нет (оставлен прошлый лучший)'}")
    print(f"Ваше место в лидерборде: {my_place}")
    print(f"Последний график: {latest_plot}")
    print(f"Локальное зеркало лидерборда: {args.leaderboard_path}")
    print(f"Путь README с лидербордом: {args.github_readme_path}")
    print("Публикация в GitHub: выполнена")

    print_leaderboard(ranked, args.table_limit)


if __name__ == "__main__":
    main()
