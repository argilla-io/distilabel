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

from typing import Callable, List, Union

from distilabel.steps.tasks.typing import ChatType


def compute_tokens(
    text_or_messages: Union[str, ChatType], tokenizer: Callable[[str], List[int]]
) -> int:
    """Helper function to count the number of tokens in a text or list of messages.

    Args:
        text_or_messages: Either a string response or a list of messages.
        tokenizer: A callable function that take str and returns the tokenized version of the text.

    Returns:
        int: _description_
    """
    if isinstance(text_or_messages, str):
        text = text_or_messages
    else:
        # If it's a list of messages, concatenate the content of each message
        text = " ".join([message["content"] for message in text_or_messages])

    return len(tokenizer(text)) if text else 0
