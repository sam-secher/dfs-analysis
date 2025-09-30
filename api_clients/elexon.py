import time
from datetime import datetime
from urllib import parse

import pandas as pd

from api_clients.base import BaseClient
from utils.datetime_helpers import generate_daily_dates, generate_weekly_dates, round_down_hh_mm, round_up_hh_mm, settlement_period, to_date_str, to_iso_z_minutes


class ElexonClient(BaseClient):
    def __init__(self, rate_limit: int):
        base_url = "https://data.elexon.co.uk/bmrs/api/v1/"
        super().__init__(base_url, rate_limit)
        self.system_forecast_url = "forecast/system/"
        self.balancing_settlement_url = "balancing/settlement/"

    def _sanitise_dt_params(self, start_dt: datetime, end_dt: datetime) -> tuple[datetime, datetime]:
        return round_down_hh_mm(start_dt), round_up_hh_mm(end_dt)

    def get_balancing_settlement_data(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        start_dt = datetime(start_dt.year, start_dt.month, start_dt.day, tzinfo=start_dt.tzinfo) # to day
        end_dt = datetime(end_dt.year, end_dt.month, end_dt.day, tzinfo=end_dt.tzinfo) # to day

        all_data: list[pd.DataFrame] = []

        for date in generate_daily_dates(start_dt, end_dt):
            date_str = to_date_str(date)
            url = self.balancing_settlement_url + f"system-prices/{date_str}"
            response = self._get(url)

            all_data.append(pd.DataFrame(response.json()["data"]))
            time.sleep(self.request_delay)

        return pd.concat(all_data).reset_index(drop=True)

    def get_lolp_drm_forecasts(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:

        start_dt, end_dt = self._sanitise_dt_params(start_dt, end_dt)

        all_data: list[pd.DataFrame] = []

        for week_start_dt, week_end_dt in generate_weekly_dates(start_dt, end_dt): # Elexon endpoint allows max 1 week at a time

            start_formatted = to_iso_z_minutes(week_start_dt)
            end_formatted = to_iso_z_minutes(week_end_dt)
            sp_start = settlement_period(week_start_dt)
            sp_end = settlement_period(week_end_dt)

            request_params = {
                "from": start_formatted,
                "to": end_formatted,
                "settlementPeriodFrom": sp_start,
                "settlementPeriodTo": sp_end,
                "format": "json",
            }

            url = self.system_forecast_url + "loss-of-load?" + parse.urlencode(request_params)
            response = self._get(url)

            all_data.append(pd.DataFrame(response.json()["data"]))

        return pd.concat(all_data).reset_index(drop=True)
