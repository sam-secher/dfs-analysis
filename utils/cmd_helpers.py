import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--interface",
        default="cmd",
        type=str,
        choices=["cmd", "streamlit"],
        help="Set the interface for the app",
    )
    return parser.parse_args()
