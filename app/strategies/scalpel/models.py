from typing import Literal

from pydantic import BaseModel, Field


class ScalpelStrategyConfig(BaseModel):
    days_back_to_consider: int = Field(1, gt=0)
    stop_loss_percent: float = Field(0.05, ge=0.0, le=1.0)
    quantity_limit: int = Field(1, gt=0)
    check_data: int = Field(60, gt=0)
    ema_fast_window: int = Field(20, gt=0)
    ema_slow_window: int = Field(50, gt=0)
    bb_window: int = Field(14, ge=2)
    bb_dev: float = Field(2.0, gt=0.0)
    timeframe_min: Literal[5, 15, 30, 60, 120, 240, 720, 1440] = 5
    backcandles: int = Field(15, gt=0)
