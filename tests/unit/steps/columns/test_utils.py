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

from distilabel.constants import DISTILABEL_METADATA_KEY
from distilabel.steps.columns.utils import merge_distilabel_metadata


def test_merge_distilabel_metadata() -> None:
    rows = [
        {DISTILABEL_METADATA_KEY: {"a": 1, "b": 1}},
        {DISTILABEL_METADATA_KEY: {"a": 2, "b": 2}},
    ]
    result = merge_distilabel_metadata(*rows)
    assert result == {"a": [1, 2], "b": [1, 2]}


def test_merge_distilabel_metadata_list() -> None:
    rows = [
        {
            DISTILABEL_METADATA_KEY: [
                {"a": 1.0, "b": 1.0},
                {"a": 1.1, "b": 1.1},
                {"a": 1.2, "b": 1.2},
            ]
        },
        {
            DISTILABEL_METADATA_KEY: [
                {"a": 2.0, "b": 2.0},
                {"a": 2.1, "b": 2.1},
                {"a": 2.2, "b": 2.2},
            ]
        },
    ]
    result = merge_distilabel_metadata(*rows)
    assert result == [
        {"a": 1.0, "b": 1.0},
        {"a": 1.1, "b": 1.1},
        {"a": 1.2, "b": 1.2},
        {"a": 2.0, "b": 2.0},
        {"a": 2.1, "b": 2.1},
        {"a": 2.2, "b": 2.2},
    ]
