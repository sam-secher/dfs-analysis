from datetime import UTC, datetime, time

from data.downloader import DataDownloader
from model.forecasting import DFSForecastingModel
from utils.cmd_helpers import parse_args
from utils.logging_config import setup_logging
from views.app import App


def main() -> None:

    args = parse_args()
    setup_logging(args)

    if args.interface == "cmd":
        data_downloader = DataDownloader(data_dir="data", rate_limit=10) # Make sure this is disabled before running the app
        data_downloader.run(
            start_dt=datetime(2024, 11, 1, tzinfo=UTC),
            end_dt=datetime(2025, 9, 30, tzinfo=UTC),
            download_interconnector_results=False,
            download_dfs_resources=False,
            download_lolp_drm=False,
            download_bm_settlement_data=False,
        )

    data_downloader.load_and_clean()
    forecasting_model = DFSForecastingModel(data_downloader, issue_time=time(10, 0))
    forecasting_model.train()
    forecasting_model.predict()

    if args.interface == "streamlit":
        App()

if __name__ == "__main__":
    main()
