import logging
from argparse import Namespace


def setup_logging(args: Namespace) -> None:
    logging.basicConfig(level=args.log_level)
