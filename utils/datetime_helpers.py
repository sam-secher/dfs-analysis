from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
import pandas as pd

# ISO_Z_MILLIS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

def to_date_str(dt: datetime) -> str:
    """Return string in format 'YYYY-MM-DD'."""
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y-%m-%d")

def to_iso_z_minutes(dt: datetime) -> str:
    """Return string in format '`YYYY-MM-DDTHH:MMZ`'."""
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y-%m-%dT%H:%MZ")

def to_iso_z_ms(dt: datetime) -> str:
    """Return string in format 'YYYY-MM-DDTHH:MM:SS.mmmZ'."""
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    # if not ISO_Z_MILLIS.fullmatch(s):
    #     msg = f"Bad timestamp format: {s!r}"
    #     raise ValueError(msg)
    # return s

def datetime_to_sp(dt: datetime) -> int:
    return (dt.hour * 2 + dt.minute // 30) + 1 # 1-48

def datetime_to_sp_series(ts: "pd.Series[pd.Timestamp]") -> "pd.Series[int]":
    return (ts.dt.hour * 2 + ts.dt.minute // 30) + 1 # 1-48

def sp_to_timedelta_series(ts: "pd.Series[int]") -> "pd.Series[pd.Timedelta]":
    return pd.to_timedelta((ts-1) * 30, unit="m")

def round_down_hh_mm(dt: datetime) -> datetime:
    minute = 30 if dt.minute > 30 else 0
    return datetime(dt.year, dt.month, dt.day, dt.hour, minute, 0, tzinfo=dt.tzinfo)

def round_up_hh_mm(dt: datetime) -> datetime:
    minute = 30 if dt.minute < 30 else 0
    hour = dt.hour + 1 if minute == 0 else dt.hour
    return datetime(dt.year, dt.month, dt.day, hour, minute, 0, 0, tzinfo=dt.tzinfo)

def generate_weekly_dates(start_dt: datetime, end_dt: datetime) -> Iterator[tuple[datetime, datetime]]:
    current_dt = start_dt
    while current_dt <= end_dt:
        week_end_dt = min(current_dt + timedelta(days=6, hours=23, minutes=30), end_dt)
        yield current_dt, week_end_dt
        current_dt = week_end_dt + timedelta(minutes=30)

def generate_daily_dates(start_dt: datetime, end_dt: datetime) -> Iterator[datetime]:
    current_dt = start_dt
    while current_dt <= end_dt:
        yield current_dt
        current_dt = current_dt + timedelta(days=1)
