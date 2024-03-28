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
from typing import TYPE_CHECKING, Any

from rich.logging import RichHandler

if TYPE_CHECKING:
    from queue import Queue


def setup_logging(log_queue: "Queue[Any]") -> None:
    """Sets up logging to use a queue across all processes."""

    # Disable overly verbose loggers
    logging.getLogger("argilla.client.feedback.dataset.local.mixins").disabled = True
    logging.getLogger("datasets").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)

    # If the current process is the main process, set up a `QueueListener`
    # to handle logs from all subprocesses
    if mp.current_process().name == "MainProcess":
        # Only in the main process, set up a listener to handle logs from the queue
        handlers = [RichHandler(rich_tracebacks=True)]
        queue_listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
        queue_listener.start()

    log_level = os.environ.get("DISTILABEL_LOG_LEVEL", "INFO").upper()
    if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        warnings.warn(
            f"Invalid log level '{log_level}', using default 'INFO' instead.",
            stacklevel=2,
        )
        log_level = "INFO"

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(QueueHandler(log_queue))
