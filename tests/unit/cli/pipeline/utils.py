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

from unittest import mock

import pytest

from distilabel.cli.pipeline.utils import (
    get_config_from_url,
    parse_runtime_parameters,
    valid_http_url,
)


def test_parse_runtime_parameters() -> None:
    assert parse_runtime_parameters(
        [
            (["step_1", "key1"], "value1"),
            (["step_1", "key2"], "value2"),
            (["step_2", "key1"], "value1"),
            (["step_2", "key2"], "value2"),
            (["step_2", "key3", "subkey1"], "subvalue1"),
            (["step_2", "key3", "subkey2"], "subvalue2"),
        ]
    ) == {
        "step_1": {"key1": "value1", "key2": "value2"},
        "step_2": {
            "key1": "value1",
            "key2": "value2",
            "key3": {
                "subkey1": "subvalue1",
                "subkey2": "subvalue2",
            },
        },
    }


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://argilla.io", True),
        ("http://argilla.io", True),
        ("argilla.io", False),
        ("argilla", False),
    ],
)
def test_valid_http_url(url: str, expected: bool) -> None:
    assert valid_http_url(url) == expected


def test_get_config_from_url_json() -> None:
    response = mock.MagicMock()
    response.json = mock.MagicMock(return_value={"unit": "test"})

    with mock.patch("requests.get") as get_mock:
        get_mock.return_value = response
        assert get_config_from_url("https://argilla.io/pipeline.json") == {
            "unit": "test"
        }


def test_get_config_from_url_yaml() -> None:
    response = mock.MagicMock()
    response.content = b"unit: test"

    with mock.patch("requests.get") as get_mock:
        get_mock.return_value = response
        assert get_config_from_url("https://argilla.io/pipeline.yaml") == {
            "unit": "test"
        }


def test_get_config_from_url_raise_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported file format"):
        get_config_from_url("https://argilla.io")
