from datetime import UTC, datetime

from data.downloader import DataDownloader
from utils.logging_config import setup_logging


def main() -> None:

    setup_logging()

    data_downloader = DataDownloader(data_dir="data")
    data_downloader.run(
        start_dt=datetime(2024, 11, 1, tzinfo=UTC),
        end_dt=datetime(2025, 9, 30, tzinfo=UTC),
        download_interconnector_results=False,
        download_dfs_resources=False,
        download_lolp_drm=False,
        download_bm_settlement_data=True,
    )


if __name__ == "__main__":
    main()
