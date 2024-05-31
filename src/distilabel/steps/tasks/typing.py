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

from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import BaseModel
from typing_extensions import TypedDict


class ChatItem(TypedDict):
    role: str
    content: str


ChatType = List[ChatItem]
"""ChatType is a type alias for a `list` of `dict`s following the OpenAI conversational format."""


class StructuredOutputType(TypedDict, total=False):
    """TypedDict to represent the structured output configuration from outlines."""

    format: Literal["json", "regex"]
    """One of "json" or "regex"."""
    schema: Union[str, Type[BaseModel], Dict[str, Any]]
    """The schema to use for the structured output. If "json", it
    can be a pydantic.BaseModel class, or the schema as a string,
    as obtained from `model_to_schema(BaseModel)`, if "regex", it
    should be a regex pattern as a string.
    """
    whitespace_pattern: Optional[Union[str, List[str]]]
    """If "json" corresponds to a string or a list of
    strings with a pattern (doesn't impact string literals).
    For example, to allow only a single space or newline with
    `whitespace_pattern=r"[\n ]?"`
    """


DefaultInput = ChatType
StructuredInput = Tuple[ChatType, Union[StructuredOutputType, None]]
FormattedInput = Union[DefaultInput, StructuredInput]
