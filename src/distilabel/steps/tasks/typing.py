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
from typing_extensions import Required, TypedDict


class ChatItem(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


ChatType = List[ChatItem]
"""ChatType is a type alias for a `list` of `dict`s following the OpenAI conversational format."""


class ChatCompletionTextPart(TypedDict, total=False):
    type: Required[Literal["text"]]
    text: Required[str]


class ImageUrl(TypedDict):
    url: Required[str]
    """Either a URL of the image or the base64 encoded image data."""


class ChatCompletionImagePart(TypedDict, total=False):
    type: Required[Literal["image_url"]]
    image_url: Required[ImageUrl]


VisionMessage = Union[ChatCompletionTextPart, ChatCompletionImagePart]
"""Type alias for the user's message in a conversation that can include text or an image.
It's the standard type for vision language models:
https://platform.openai.com/docs/guides/vision
"""


# Note: Follows the definition from vLLM: https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/chat_utils.py#L141
# TODO: We should merge this with the ChatType to have a single type for the new conversations
# that may include multimodal images/audio/video... or tool calls for example.
class ConversationMessage(TypedDict, total=False):
    role: Required[Literal["system", "user", "assistant"]]
    """The role of the message's author."""

    content: Union[Optional[str], list[VisionMessage]]
    """The contents of the message for a model with vision capabilities,
    meaning the models can take in images and answer questions about them."""


ConversationType = list[ConversationMessage]
"""ConversationType is a type alias for a `list` of `dict`s following the OpenAI conversational format for
vision models."""


class OutlinesStructuredOutputType(TypedDict, total=False):
    """TypedDict to represent the structured output configuration from `outlines`."""

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


class InstructorStructuredOutputType(TypedDict, total=False):
    """TypedDict to represent the structured output configuration from `instructor`."""

    format: Optional[Literal["json"]]
    """One of "json"."""
    schema: Union[Type[BaseModel], Dict[str, Any]]
    """The schema to use for the structured output, a `pydantic.BaseModel` class. """
    mode: Optional[str]
    """Generation mode. Take a look at `instructor.Mode` for more information, if not informed it will
    be determined automatically. """
    max_retries: int
    """Number of times to reask the model in case of error, if not set will default to the model's default. """


StructuredOutputType = Union[
    OutlinesStructuredOutputType, InstructorStructuredOutputType
]
"""StructuredOutputType is an alias for the union of `OutlinesStructuredOutputType` and `InstructorStructuredOutputType`."""

StandardInput = ChatType
"""StandardInput is an alias for ChatType that defines the default / standard input produced by `format_input`."""
StructuredInput = Tuple[StandardInput, Union[StructuredOutputType, None]]
"""StructuredInput defines a type produced by `format_input` when using either `StructuredGeneration` or a subclass of it."""
FormattedInput = Union[StandardInput, StructuredInput, ConversationType]
"""FormattedInput is an alias for the union of `StandardInput` and `StructuredInput` as generated
by `format_input` and expected by the `LLM`s, as well as `ConversationType` for the vision language models."""
