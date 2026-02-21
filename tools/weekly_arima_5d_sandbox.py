from datetime import timedelta
from pathlib import Path
from uuid import uuid4
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from tinkoff.invest import CandleInterval, Client, MoneyValue, OrderDirection, OrderType
from tinkoff.invest.constants import INVEST_GRPC_API_SANDBOX
from tinkoff.invest.exceptions import RequestError
from tinkoff.invest.utils import now

from app.config import settings
from app.utils.quotation import quotation_to_float

FIGI_SBER = "BBG004730N88"
DAYS_BACK = 1095
ARIMA_ORDER = (1, 1, 2)
FORECAST_HORIZON_DAYS = 5
MIN_HISTORY_DAYS = 252
COMMISSION_RATE = 0.001
INITIAL_CAPITAL = 100000
ORDER_LOTS = 1

OUTPUT_DIR = Path("reports/arima_sber")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")


def load_close_series(api: Client) -> pd.Series:
    rows = []
    for candle in api.get_all_candles(
        figi=FIGI_SBER,
        from_=now() - timedelta(days=DAYS_BACK),
        to=now(),
        interval=CandleInterval.CANDLE_INTERVAL_DAY,
    ):
        rows.append(
            {
                "Date": candle.time,
                "Close": quotation_to_float(candle.close),
            }
        )

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").set_index("Date")
    return df["Close"]


def run_price_backtest(close: pd.Series) -> pd.DataFrame:
    fit = ARIMA(close, order=ARIMA_ORDER).fit()
    pred = fit.get_prediction(start=1, end=len(close) - 1)

    return pd.DataFrame(
        {
            "actual_price": close.iloc[1:].values,
            "predicted_price": pred.predicted_mean.values,
            "lower_price": pred.conf_int(alpha=0.05).iloc[:, 0].values,
            "upper_price": pred.conf_int(alpha=0.05).iloc[:, 1].values,
        },
        index=close.index[1:],
    )


def run_weekly_strategy(close: pd.Series) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    weekly_close = close.resample("W-FRI").last().dropna()
    rebalance_dates = weekly_close.index[weekly_close.index >= close.index[MIN_HISTORY_DAYS]]

    records = []
    prev_position = 0

    for i in range(len(rebalance_dates) - 1):
        t = rebalance_dates[i]
        t_next = rebalance_dates[i + 1]

        history = close.loc[:t]
        fit = ARIMA(history, order=ARIMA_ORDER).fit()
        forecast_5d = float(fit.forecast(steps=FORECAST_HORIZON_DAYS).iloc[-1])

        close_t = float(weekly_close.loc[t])
        close_next = float(weekly_close.loc[t_next])

        position = 1 if forecast_5d > close_t else -1

        asset_return = close_next / close_t - 1.0
        gross_return = position * asset_return

        turnover = abs(position - prev_position)
        fee = COMMISSION_RATE * turnover
        net_return = gross_return - fee

        records.append(
            {
                "date": t,
                "next_date": t_next,
                "close_t": close_t,
                "forecast_5d": forecast_5d,
                "position": position,
                "asset_return": asset_return,
                "gross_return": gross_return,
                "fee": fee,
                "net_return": net_return,
            }
        )

        prev_position = position

    trades_df = pd.DataFrame(records).set_index("date")
    equity_no_fee = INITIAL_CAPITAL * (1.0 + trades_df["gross_return"]).cumprod()
    equity_with_fee = INITIAL_CAPITAL * (1.0 + trades_df["net_return"]).cumprod()

    return trades_df, equity_no_fee, equity_with_fee


def save_charts(
    close: pd.Series,
    forecast_df: pd.DataFrame,
    equity_no_fee: pd.Series,
    equity_with_fee: pd.Series,
) -> None:
    plt.figure(figsize=(16, 6))
    plt.plot(close.index, close.values, color="blue", label="Real Price (3Y)")
    plt.plot(
        forecast_df.index,
        forecast_df["predicted_price"],
        color="orange",
        label="Predicted Price (backtest)",
    )
    plt.fill_between(
        forecast_df.index,
        forecast_df["lower_price"],
        forecast_df["upper_price"],
        color="orange",
        alpha=0.12,
        label="95% CI",
    )
    plt.title("SBER Real vs ARIMA Backtest Prediction (3 Years)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "arima_3y_backtest_overlay.png", dpi=170)
    plt.show()

    plt.figure(figsize=(16, 5))
    plt.plot(equity_no_fee.index, equity_no_fee.values, color="blue")
    plt.title("Strategy Equity Without Commission")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "equity_without_commission.png", dpi=170)
    plt.show()

    plt.figure(figsize=(16, 5))
    plt.plot(equity_with_fee.index, equity_with_fee.values, color="orange")
    plt.title("Strategy Equity With Commission")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "equity_with_commission.png", dpi=170)
    plt.show()


def post_latest_sandbox_order(api: Client, close: pd.Series) -> None:
    fit = ARIMA(close, order=ARIMA_ORDER).fit()
    forecast_5d = float(fit.forecast(steps=FORECAST_HORIZON_DAYS).iloc[-1])
    close_t = float(close.iloc[-1])

    direction = (
        OrderDirection.ORDER_DIRECTION_BUY
        if forecast_5d > close_t
        else OrderDirection.ORDER_DIRECTION_SELL
    )

    accounts = api.users.get_accounts().accounts
    if not accounts:
        api.sandbox.open_sandbox_account()
        accounts = api.users.get_accounts().accounts
    account_id = accounts[0].id

    api.sandbox.sandbox_pay_in(
        account_id=account_id,
        amount=MoneyValue(currency="rub", units=1_000_000, nano=0),
    )

    signal = "LONG" if direction == OrderDirection.ORDER_DIRECTION_BUY else "SHORT"
    print("Latest close:", round(close_t, 4))
    print("Forecast 5d:", round(forecast_5d, 4))
    print("Signal:", signal)

    try:
        status = api.market_data.get_trading_status(instrument_id=FIGI_SBER)
        if not status.market_order_available_flag:
            print("Sandbox order skipped: market order is not available now.")
            return

        order = api.sandbox.post_sandbox_order(
            order_id=str(uuid4()),
            figi=FIGI_SBER,
            quantity=ORDER_LOTS,
            direction=direction,
            order_type=OrderType.ORDER_TYPE_MARKET,
            account_id=account_id,
        )
        print("Posted sandbox order_id:", order.order_id)
    except RequestError as e:
        print(f"Sandbox order skipped: {e}")


def main() -> None:
    with Client(settings.token, target=INVEST_GRPC_API_SANDBOX, app_name=settings.app_name) as api:
        close = load_close_series(api)
        forecast_df = run_price_backtest(close)
        trades_df, equity_no_fee, equity_with_fee = run_weekly_strategy(close)

        save_charts(close, forecast_df, equity_no_fee, equity_with_fee)

        trades_df.to_csv(OUTPUT_DIR / "weekly_trades_5d_forecast.csv")
        equity_no_fee.to_csv(OUTPUT_DIR / "equity_without_commission.csv")
        equity_with_fee.to_csv(OUTPUT_DIR / "equity_with_commission.csv")
        forecast_df.to_csv(OUTPUT_DIR / "forecast_backtest_3y.csv")

        post_latest_sandbox_order(api, close)


if __name__ == "__main__":
    main()
