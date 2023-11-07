import logging


def _get_root_logger() -> logging.Logger:
    return logging.getLogger("distilabel")


def _configure_root_logger():
    root_logger = _get_root_logger()
    root_logger.setLevel(logging.INFO)


_configure_root_logger()


def get_logger() -> logging.Logger:
    return logging.getLogger("distilabel")
