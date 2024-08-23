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

import sys
from typing import TYPE_CHECKING, List

import pytest

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.nodes import Item


def pytest_configure(config: "Config") -> None:
    config.addinivalue_line(
        "markers",
        "skip_python_versions(versions): mark test to be skipped on specified Python versions",
    )


def pytest_collection_modifyitems(config: "Config", items: List["Item"]) -> None:
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    for item in items:
        skip_versions_marker = item.get_closest_marker("skip_python_versions")
        if skip_versions_marker:
            versions_to_skip = skip_versions_marker.args[0]
            if current_version in versions_to_skip:
                skip_reason = f"Test not supported on Python {current_version}"
                item.add_marker(pytest.mark.skip(reason=skip_reason))
