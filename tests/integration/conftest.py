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
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from distilabel.telemetry._client import TelemetryClient


@pytest.fixture(autouse=True)
def temp_cache_dir() -> Generator[None, None, None]:
    """Set the cache directory to a temporary directory for all tests."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.environ["DISTILABEL_CACHE_DIR"] = tmpdirname
        yield


@pytest.fixture(autouse=True)
def mock_telemetry():
    # Patch the entire TelemetryClient class
    with patch(
        "distilabel.telemetry._TELEMETRY_CLIENT", autospec=True
    ) as MockTelemetryClient:
        # Mock individual methods with wraps to execute the original methods
        instance = TelemetryClient()
        mock_instance = MockTelemetryClient
        mock_instance.track_add_step_data = MagicMock(
            wraps=instance.track_add_step_data
        )
        mock_instance.track_process_batch_data = MagicMock(
            wraps=instance.track_process_batch_data
        )
        mock_instance.track_run_data = MagicMock(wraps=instance.track_run_data)
        mock_instance.track_exception = MagicMock(wraps=instance.track_exception)
        yield mock_instance
