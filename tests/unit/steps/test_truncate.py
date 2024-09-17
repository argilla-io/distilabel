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

from typing import Optional

import pytest

from distilabel.steps.truncate import TruncateTextColumn


@pytest.mark.parametrize(
    "max_length, text, tokenizer, expected",
    [
        (
            10,
            "This is a sample text that is longer than 10 characters",
            None,
            "This is a ",
        ),
        (
            4,
            "This is a sample text that is longer than 10 characters",
            "teknium/OpenHermes-2.5-Mistral-7B",
            "This is a sample",
        ),
    ],
)
def test_truncate_row(
    max_length: int, text: str, tokenizer: Optional[str], expected: str
) -> None:
    trunc = TruncateTextColumn(
        column="text", max_length=max_length, tokenizer=tokenizer
    )
    trunc.load()

    assert next(trunc.process([{"text": text}])) == [{"text": expected}]
