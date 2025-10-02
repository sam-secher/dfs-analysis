from datetime import datetime
from pathlib import Path
from typing import cast

from api_clients.elexon import ElexonClient
from api_clients.neso import NESOClient
import pandas as pd

from utils.datetime_helpers import datetime_to_sp_series, sp_to_timedelta_series

class DataDownloader:
    def __init__(self, data_dir: str, rate_limit: int = 2) -> None:
        self._neso_client = NESOClient(rate_limit=rate_limit)
        self._elexon_client = ElexonClient(rate_limit=rate_limit)

        self._data_dir = Path(data_dir)
        self._neso_dir = self._data_dir / "neso"
        self._dfs_dir = self._neso_dir / "dfs"
        self._elexon_dir = self._data_dir / "elexon"

        self._interconnector_results_path = self._neso_dir / "interconnector_results.csv"
        self._lolp_drm_path = self._elexon_dir / "lolp_drm.csv"
        self._settlement_data_path = self._elexon_dir / "settlement_data.csv"

        self._dfs_utilisation_report_summary_path = self._dfs_dir / "utilisation_report_summary.csv"
        self._dfs_utilisation_report_path = self._dfs_dir / "utilisation_report.csv"
        self._dfs_service_requirement_path = self._dfs_dir / "service_requirement.csv"
        self._dfs_industry_notification_path = self._dfs_dir / "industry_notification.csv"

        self.dfs_data = cast(pd.DataFrame, None)
        self.lolp_drm_data = cast(pd.DataFrame, None)
        self.interconnector_data = cast(pd.DataFrame, None)
        self.settlement_data = cast(pd.DataFrame, None)

    def run(self, start_dt: datetime, end_dt: datetime, download_interconnector_results: bool, download_dfs_resources: bool,
        download_lolp_drm: bool, download_bm_settlement_data: bool) -> None:

        self._data_dir.mkdir(exist_ok=True)

        if download_interconnector_results:

            self._neso_dir.mkdir(exist_ok=True)

            data = self._neso_client.get_interconnector_requirements(start_dt, end_dt, limit=10000)
            data.to_csv(self._interconnector_results_path, index=False)

        if download_dfs_resources:

            self._dfs_dir.mkdir(parents=True, exist_ok=True)

            dfs_resources = self._neso_client.get_dfs_resources()
            dfs_resources.utilisation_report_summary.to_csv(self._dfs_utilisation_report_summary_path, index=False)
            dfs_resources.utilisation_report.to_csv(self._dfs_utilisation_report_path, index=False)
            dfs_resources.service_requirement.to_csv(self._dfs_service_requirement_path, index=False)
            dfs_resources.industry_notification.to_csv(self._dfs_industry_notification_path, index=False)

        if download_lolp_drm:

            self._elexon_dir.mkdir(exist_ok=True)

            data = self._elexon_client.get_lolp_drm_forecasts(start_dt, end_dt)
            data.to_csv(self._lolp_drm_path, index=False)

        if download_bm_settlement_data:

            self._elexon_dir.mkdir(exist_ok=True)

            data = self._elexon_client.get_balancing_settlement_data(start_dt, end_dt)
            data.to_csv(self._settlement_data_path, index=False)

    def load_and_clean(self) -> None:
        self._load_dfs_data()
        self._load_lolp_drm_data()
        self._load_interconnector_data()
        self._load_settlement_data()

    def _load_settlement_data(self) -> None:
        settlement_df = pd.read_csv(self._settlement_data_path)
        datetimes = pd.to_datetime(settlement_df["startTime"], utc=True)
        sps = settlement_df["settlementPeriod"]
        system_price = settlement_df["systemSellPrice"]
        niv = settlement_df["netImbalanceVolume"]

        columns = [
            "datetime",
            "settlement_period",
            "system_price",
            "niv",
        ]

        settlement_data = pd.DataFrame(data=zip(datetimes, sps, system_price, niv), columns=columns)
        self.settlement_data = settlement_data.sort_values("datetime").reset_index(drop=True)

    def _load_interconnector_data(self) -> None:
        interconnector_df = pd.read_csv(self._interconnector_results_path)
        published_datetime = pd.to_datetime(interconnector_df["Published DateTime"], utc=True)
        dates = pd.to_datetime(interconnector_df["Start Time"], utc=True)
        buy_sell = interconnector_df["Buy Sell"].map({"Buy": 1, "Sell": -1})
        volume_ifa1 = interconnector_df["IFA1 Volume"] * buy_sell
        volume_ifa2 = interconnector_df["IFA2 Volume"] * buy_sell
        volume_vkl = interconnector_df["VKL Volume"] * buy_sell
        volume_el = interconnector_df["EL Volume"] * buy_sell
        volume_nemo = interconnector_df["NEMO Volume"] * buy_sell
        volume_bn = interconnector_df["BN Volume"] * buy_sell
        clearing_price = interconnector_df["Clearing Price"]

        columns = [
            "datetime",
            "volume_ifa1",
            "volume_ifa2",
            "volume_vkl",
            "volume_el",
            "volume_nemo",
            "volume_bn",
            "clearing_price",
            "published_datetime",
        ]

        interconnector_data = pd.DataFrame(data=zip(dates, volume_ifa1, volume_ifa2, volume_vkl, volume_el, volume_nemo, volume_bn, clearing_price, published_datetime), columns=columns)
        interconnector_data = interconnector_data.groupby("datetime", as_index=False).agg(
            volume_ifa1=("volume_ifa1", "sum"),
            volume_ifa2=("volume_ifa2", "sum"),
            volume_vkl=("volume_vkl", "sum"),
            volume_el=("volume_el", "sum"),
            volume_nemo=("volume_nemo", "sum"),
            volume_bn=("volume_bn", "sum"),
            clearing_price=("clearing_price", "max"),
        ) # need to do this for duplicated datetimes, not sure if this is correct, may be better to take VWA for clearing price
        interconnector_data_shifted = interconnector_data.copy()
        interconnector_data_shifted["datetime"] = interconnector_data_shifted["datetime"] + pd.Timedelta(minutes=30)
        interconnector_data = pd.concat([interconnector_data, interconnector_data_shifted]).sort_values("datetime")
        self.interconnector_data = interconnector_data.reset_index(drop=True)

    def _load_lolp_drm_data(self) -> None:
        lolp_drm_df = pd.read_csv(self._lolp_drm_path)
        dates = pd.to_datetime(lolp_drm_df["settlementDate"], format="%Y-%m-%d", utc=True)
        sps = lolp_drm_df["settlementPeriod"]
        times = sp_to_timedelta_series(sps)
        datetimes = dates + times
        forecast_horizon = lolp_drm_df["forecastHorizon"]
        lolp = lolp_drm_df["lossOfLoadProbability"]
        drm = lolp_drm_df["deratedMargin"]

        columns = [
            "datetime",
            "settlement_period",
            "forecast_horizon",
            "lolp",
            "drm",
        ]

        lolp_drm_data = pd.DataFrame(data=zip(datetimes, sps, forecast_horizon, lolp, drm), columns=columns)
        self.lolp_drm_data = lolp_drm_data.sort_values("datetime").reset_index(drop=True)

    def _load_dfs_data(self) -> None:
        dfs_requirement_df = pd.read_csv(self._dfs_service_requirement_path)
        datetimes = dfs_requirement_df["Delivery Date"] + " " + dfs_requirement_df["From"] # SPs
        datetimes = pd.to_datetime(datetimes, format="%d/%m/%Y %H:%M", utc=True)
        settlement_periods = datetime_to_sp_series(datetimes)
        dfs_unit_id = dfs_requirement_df["DFS Unit ID"]
        dfs_participant = dfs_requirement_df["Registered DFS Participant"]
        offered_volume_mw = dfs_requirement_df["DFS Volume MW"]
        offered_price = dfs_requirement_df["Utilisation Price GBP per MWh"] # GBP/MWh
        offer_status = dfs_requirement_df["Status"] # Accepted / Rejected

        columns = [
            "datetime",
            "settlement_period",
            "dfs_unit_id",
            "dfs_participant",
            "offered_volume_mw",
            "offered_price",
            "offer_status",
        ]

        dfs_data = pd.DataFrame(data=zip(datetimes, settlement_periods, dfs_unit_id, dfs_participant, offered_volume_mw, offered_price, offer_status), columns=columns)
        self.dfs_data = dfs_data.sort_values("datetime").reset_index(drop=True)


