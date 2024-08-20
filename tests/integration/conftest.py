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

import os
import tempfile
from typing import TYPE_CHECKING, Generator

import pytest
from distilabel.telemetry import TelemetryClient

if TYPE_CHECKING:
    pass


@pytest.fixture(autouse=True)
def temp_cache_dir() -> Generator[None, None, None]:
    """Set the cache directory to a temporary directory for all tests."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.environ["DISTILABEL_CACHE_DIR"] = tmpdirname
        yield


@pytest.fixture(autouse=True)
def mock_telemetry(mocker) -> "TelemetryClient":
    # Create a real instance TelemetryClient
    real_telemetry = TelemetryClient()

    # Create a wrapper to track calls to other methods
    for attr_name in dir(real_telemetry):
        attr = getattr(real_telemetry, attr_name)
        if callable(attr) and not attr_name.startswith("__"):
            wrapped = mocker.Mock(wraps=attr)
            setattr(real_telemetry, attr_name, wrapped)

    # Patch the _TELEMETRY_CLIENT to use the real_telemetry
    mocker.patch("argilla_server.telemetry._TELEMETRY_CLIENT", new=real_telemetry)

    return real_telemetry
