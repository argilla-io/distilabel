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
import os
import warnings

from rich.logging import RichHandler


def get_logger(suffix: str) -> logging.Logger:
    """Gets the `logging.Logger` for the `distilabel` package with a custom
    configuration. Also uses `rich` for better formatting.
    """
    # Disable logging for argilla.client.feedback.dataset.local.mixins
    # as it's too verbose, and there's no way to disable all the `argilla` logs
    logging.getLogger("argilla.client.feedback.dataset.local.mixins").disabled = True

    # Remove `datasets` logger to only log on `critical` mode
    # as it produces `PyTorch` messages to update on `info`
    logging.getLogger("datasets").setLevel(logging.CRITICAL)

    logging.getLogger("httpx").setLevel(logging.CRITICAL)

    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )

    log_level = os.environ.get("DISTILABEL_LOG_LEVEL", "INFO").upper()
    if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        warnings.warn(
            f"Invalid log level '{log_level}', using default 'INFO' instead.",
            stacklevel=2,
        )
        log_level = "INFO"

    logger = logging.getLogger(f"distilabel.{suffix}")
    logger.setLevel(log_level)
    return logger
