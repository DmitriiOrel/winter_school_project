import argparse
import asyncio
from datetime import timedelta
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from tinkoff.invest import CandleInterval
from tinkoff.invest.utils import now

from app.client import client
from app.config import settings
from app.utils.quotation import quotation_to_float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SBER ARIMA price predictor.")
    parser.add_argument("--figi", default="BBG004730N88")
    parser.add_argument("--days-back", type=int, default=1095)
    parser.add_argument("--future-days", type=int, default=30)
    return parser.parse_args()


async def load_close_series(figi: str, days_back: int) -> pd.Series:
    rows = []
    async for candle in client.get_all_candles(
        figi=figi,
        from_=now() - timedelta(days=days_back),
        to=now(),
        interval=CandleInterval.CANDLE_INTERVAL_DAY,
    ):
        rows.append({"time": candle.time, "close": quotation_to_float(candle.close)})

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time")
    return pd.Series(df["close"].values, index=df["time"], name="close")


async def main() -> None:
    args = parse_args()
    await client.init()
    warnings.filterwarnings("ignore")

    close_series = await load_close_series(args.figi, args.days_back)
    log_series = np.log(close_series)

    auto_model = auto_arima(
        log_series,
        start_p=0,
        start_q=0,
        test="adf",
        max_p=3,
        max_q=3,
        m=1,
        d=None,
        seasonal=False,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    best_order = auto_model.order

    model = ARIMA(log_series, order=best_order)
    fitted = model.fit()

    next_log = float(fitted.forecast(steps=1).iloc[0])
    next_price = float(np.exp(next_log))
    last_price = float(close_series.iloc[-1])
    expected_return = next_price / last_price - 1.0

    future_log = fitted.forecast(steps=args.future_days)
    future_price = np.exp(future_log)
    future_index = pd.bdate_range(
        start=close_series.index[-1] + pd.Timedelta(days=1),
        periods=args.future_days,
        tz=close_series.index.tz,
    )
    future_series = pd.Series(future_price.values, index=future_index, name="forecast")

    output_dir = Path("reports/arima_sber")
    output_dir.mkdir(parents=True, exist_ok=True)
    future_series.to_csv(output_dir / "live_future_forecast.csv")

    print("=== ARIMA Price Forecast ===")
    print(f"FIGI: {args.figi}")
    print(f"Sandbox mode: {settings.sandbox}")
    print(f"Best ARIMA order: {best_order}")
    print(f"Last close: {last_price:.4f}")
    print(f"Next predicted close: {next_price:.4f}")
    print(f"Expected return: {expected_return*100:.4f}%")
    print(f"Saved: {output_dir / 'live_future_forecast.csv'}")


if __name__ == "__main__":
    asyncio.run(main())
