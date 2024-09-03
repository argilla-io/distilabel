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

# Idea from: https://github.com/vllm-project/vllm/blob/main/vllm/envs.py

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from distilabel import constants

if TYPE_CHECKING:
    DISTILABEL_LOG_LEVEL: str = "INFO"
    DISTILABEL_PIPELINE_NAME: Optional[str] = None
    DISTILABEL_PIPELINE_CACHE_ID: Optional[str] = None
    DISTILABEL_PIPELINE_CACHE_DIR: Optional[str] = None
    DISTILABEL_CACHE_DIR: Optional[str] = None

ENVIRONMENT_VARIABLES: Dict[str, Callable[[], Any]] = {
    # `distilabel` logging level.
    "DISTILABEL_LOG_LEVEL": lambda: os.getenv("DISTILABEL_LOG_LEVEL", "INFO").upper(),
    # The name of the `distilabel` pipeline currently running.
    constants.PIPELINE_NAME_ENV_NAME: lambda: os.getenv(
        constants.PIPELINE_NAME_ENV_NAME, None
    ),
    # The cache ID of the `distilabel` pipeline currently running.
    constants.PIPELINE_CACHE_ID_ENV_NAME: lambda: os.getenv(
        constants.PIPELINE_CACHE_ID_ENV_NAME, None
    ),
    # The cache path of the `distilabel` pipeline currently running.
    constants.PIPELINE_CACHE_DIR_ENV_NAME: lambda: os.getenv(
        constants.PIPELINE_CACHE_DIR_ENV_NAME, None
    ),
    # The cache ID of the `distilabel` pipeline currently running.
    "DISTILABEL_CACHE_DIR": lambda: os.getenv("DISTILABEL_CACHE_DIR", None),
}


def __getattr__(name: str) -> Any:
    # lazy evaluation of environment variables
    if name in ENVIRONMENT_VARIABLES:
        return ENVIRONMENT_VARIABLES[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return list(ENVIRONMENT_VARIABLES.keys())
