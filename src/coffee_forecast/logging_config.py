import logging

from dotenv import load_dotenv


def configure_logging(level: int = logging.INFO) -> None:
    load_dotenv()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s — %(message)s",
    )
