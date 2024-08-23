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

from typing import Any

import pytest

from distilabel.utils.chat import is_openai_format


@pytest.mark.parametrize(
    "input, expected",
    [
        (None, False),
        (1, False),
        ("Hello", False),
        (
            [
                {"role": "user", "content": "Hello!"},
            ],
            True,
        ),
        (
            [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi! How can I help you?"},
            ],
            True,
        ),
        (
            [
                {"role": "system", "content": "You're a helpful assistant"},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi! How can I help you?"},
            ],
            True,
        ),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What’s in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                            },
                        },
                    ],
                }
            ],
            True,
        ),
    ],
)
def test_is_openai_format(input: Any, expected: bool) -> None:
    assert is_openai_format(input) == expected
