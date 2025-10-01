from datetime import UTC, datetime

from data.downloader import DataDownloader
from utils.cmd_helpers import parse_args
from utils.logging_config import setup_logging
from views.app import App


def main() -> None:

    args = parse_args()
    setup_logging(args)

    if args.interface == "cmd":
        data_downloader = DataDownloader(data_dir="data") # Make sure this is disabled before running the app
        data_downloader.run(
            start_dt=datetime(2024, 11, 1, tzinfo=UTC),
            end_dt=datetime(2025, 9, 30, tzinfo=UTC),
            download_interconnector_results=False,
            download_dfs_resources=False,
            download_lolp_drm=False,
            download_bm_settlement_data=False,
        )

    if args.interface == "streamlit":
        App()

if __name__ == "__main__":
    main()
