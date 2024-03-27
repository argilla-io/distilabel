# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import multiprocessing as mp
import os
import warnings
from logging.handlers import QueueHandler, QueueListener

from rich.logging import RichHandler

log_queue = mp.Queue()


def setup_logging() -> None:
    """Sets up logging to use a queue across all processes."""
    handlers = [RichHandler()]

    # Create a listener (background thread) to listen for logs from the queue. The listener
    # will be stopped when the main process exits or when `queue_listener.stop()` is called.
    queue_listener = QueueListener(log_queue, *handlers)
    queue_listener.start()

    # Handler for subprocesses to use
    queue_handler = QueueHandler(log_queue)
    logging.basicConfig(handlers=[queue_handler], level=logging.INFO)


def get_logger(suffix: str) -> logging.Logger:
    """Returns a logger with the specified suffix. It will return the right logger depending
    on whether it is called in the main process or a child process.

    Args:
        suffix: The suffix to append to the logger name.

    Returns:
        The logger with the specified suffix.
    """
    logger_name = f"distilabel.{suffix}"

    logging.getLogger("argilla.client.feedback.dataset.local.mixins").disabled = True
    logging.getLogger("datasets").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)

    log_level = os.environ.get("DISTILABEL_LOG_LEVEL", "INFO").upper()
    if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        warnings.warn(
            f"Invalid log level {log_level}. Defaulting to INFO.",
            UserWarning,
            stacklevel=2,
        )
        log_level = "INFO"

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    return logger
