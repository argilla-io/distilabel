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

from typing import List, Literal, Union

from typing_extensions import Required, TypedDict


class TextContent(TypedDict, total=False):
    type: Required[Literal["text"]]
    text: Required[str]


class ImageUrl(TypedDict):
    url: Required[str]
    """Either a URL of the image or the base64 encoded image data."""


class ImageContent(TypedDict, total=False):
    """Type alias for the user's message in a conversation that can include text or an image.
    It's the standard type for vision language models:
    https://platform.openai.com/docs/guides/vision
    """

    type: Required[Literal["image_url"]]
    image_url: Required[ImageUrl]


class ChatItem(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: Union[str, list[Union[TextContent, ImageContent]]]


ChatType = List[ChatItem]
"""ChatType is a type alias for a `list` of `dict`s following the OpenAI conversational format."""
