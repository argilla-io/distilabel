import logging
import os

DISTILABEL_LOG_LEVEL = os.environ.get("DISTILABEL_LOG_LEVEL", "INFO").upper()


def _get_root_logger() -> logging.Logger:
    return logging.getLogger("distilabel")


def _configure_root_logger():
    root_logger = _get_root_logger()
    root_logger.setLevel(DISTILABEL_LOG_LEVEL)


_configure_root_logger()


def get_logger() -> logging.Logger:
    return logging.getLogger("distilabel")
