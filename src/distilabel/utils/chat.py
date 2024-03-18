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


def is_openai_format(input: Any) -> bool:
    """Checks if the input is in OpenAI chat-like format:

    ```python
    [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help you?"},
    ]
    ```

    Args:
        input: The input to check.

    Returns:
        A boolean indicating if the input is in OpenAI chat-like format.
    """
    if not isinstance(input, list):
        return False
    return all(
        isinstance(x, dict) and "role" in x.keys() and "content" in x.keys()
        for x in input
    )
