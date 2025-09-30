from datetime import datetime
from pathlib import Path

from api_clients.elexon import ElexonClient
from api_clients.neso import NESOClient


class DataDownloader:
    def __init__(self, data_dir: str, rate_limit: int = 2) -> None:
        self.neso_client = NESOClient()
        self.elexon_client = ElexonClient()
        self.data_dir = data_dir

        self.request_delay = 1 / rate_limit # delay between requests in seconds

    def run(self, start_dt: datetime, end_dt: datetime, download_interconnector_results: bool, download_dfs_resources: bool,
        download_lolp_drm: bool, download_bm_settlement_data: bool) -> None:
        data_dir = Path(self.data_dir)
        data_dir.mkdir(exist_ok=True)

        neso_dir = data_dir / "neso"
        elexon_dir = data_dir / "elexon"

        if download_interconnector_results:

            neso_dir.mkdir(exist_ok=True)

            data = self.neso_client.get_interconnector_requirements(start_dt, end_dt)
            data.to_csv(neso_dir / "interconnector_results.csv", index=False)

        if download_dfs_resources:

            dfs_dir = neso_dir / "dfs"
            dfs_dir.mkdir(parents=True, exist_ok=True)

            dfs_resources = self.neso_client.get_dfs_resources()
            dfs_resources.utilisation_report_summary.to_csv(dfs_dir / "utilisation_report_summary.csv", index=False)
            dfs_resources.utilisation_report.to_csv(dfs_dir / "utilisation_report.csv", index=False)
            dfs_resources.service_requirement.to_csv(dfs_dir / "service_requirement.csv", index=False)
            dfs_resources.industry_notification.to_csv(dfs_dir / "industry_notification.csv", index=False)

        if download_lolp_drm:

            elexon_dir.mkdir(exist_ok=True)

            data = self.elexon_client.get_lolp_drm_forecasts(start_dt, end_dt)
            data.to_csv(elexon_dir / "lolp_drm.csv", index=False)

        if download_bm_settlement_data:

            elexon_dir.mkdir(exist_ok=True)

            data = self.elexon_client.get_balancing_settlement_data(start_dt, end_dt)
            data.to_csv(elexon_dir / "settlement_data.csv", index=False)
