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
from logging import FileHandler
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from rich.logging import RichHandler

if TYPE_CHECKING:
    from queue import Queue


_SILENT_LOGGERS = [
    "datasets",
    "httpx",
    "openai._base_client",
    "httpcore.http11",
    "httpcore.connection",
    "urllib3.connectionpool",
    "filelock",
    "fsspec",
    "asyncio",
]

queue_listener: Union[QueueListener, None] = None


def setup_logging(log_queue: "Queue[Any]", filename: Optional[str] = None) -> None:
    """Sets up logging to use a queue across all processes."""
    global queue_listener

    # Disable overly verbose loggers
    logging.getLogger("argilla.client.feedback.dataset.local.mixins").disabled = True
    for logger in _SILENT_LOGGERS:
        logging.getLogger(logger).setLevel(logging.CRITICAL)

    # If the current process is the main process, set up a `QueueListener`
    # to handle logs from all subprocesses
    if mp.current_process().name == "MainProcess":
        formatter = logging.Formatter("['%(name)s'] %(message)s")
        handler = RichHandler(rich_tracebacks=True)
        handler.setFormatter(formatter)
        if not Path(filename).parent.exists():
            Path(filename).parent.mkdir(parents=True, exist_ok=True)

        file_handler = FileHandler(filename, delay=True)
        file_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)

        queue_listener = QueueListener(
            log_queue, handler, file_handler, respect_handler_level=True
        )
        queue_listener.start()

    log_level = os.environ.get("DISTILABEL_LOG_LEVEL", "INFO").upper()
    if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        warnings.warn(
            f"Invalid log level '{log_level}', using default 'INFO' instead.",
            stacklevel=2,
        )
        log_level = "INFO"

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)
    root_logger.addHandler(QueueHandler(log_queue))


def stop_logging() -> None:
    """Stops the `QueueListener` if it's running."""
    global queue_listener
    if queue_listener is not None:
        queue_listener.stop()
        queue_listener = None
