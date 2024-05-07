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

from typing import List, Literal, Optional, Union

from pydantic import BaseModel
from typing_extensions import TypedDict


class ChatItem(TypedDict):
    role: str
    content: str


ChatType = List[ChatItem]
"""ChatType is a type alias for a `list` of `dict`s following the OpenAI conversational format."""


class OutlinesStructuredOutputDict(TypedDict):
    """Type alias for the arguments that can be passed to generate structured outputs using `outlines`."""

    format: Literal["text", "json", "regex", "cfg"]
    structure: Union[str, BaseModel]
    whitespace_pattern: Optional[str] = None
