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

from rich.logging import RichHandler

DISTILABEL_LOG_LEVEL = os.environ.get("DISTILABEL_LOG_LEVEL", "INFO").upper()


logging.basicConfig(
    level=logging.INFO,
    format="[PID: %(process)d] %(message)s",
    datefmt="%X",
    handlers=[RichHandler(rich_tracebacks=True, tracebacks_show_locals=True)],
)

# Silence httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)


def _get_root_logger() -> logging.Logger:
    return logging.getLogger("distilabel")


def _configure_root_logger():
    root_logger = _get_root_logger()
    root_logger.setLevel(DISTILABEL_LOG_LEVEL)


_configure_root_logger()


def get_logger() -> logging.Logger:
    return logging.getLogger("distilabel")
