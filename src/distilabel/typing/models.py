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

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel
from typing_extensions import NotRequired, TypedDict

from distilabel.typing.base import ChatType

LLMOutput = List[Union[str, None]]


class Logprob(TypedDict):
    token: str
    logprob: float


LLMLogprobs = List[List[List[Logprob]]]
"""A type alias representing the probability distributions output by an `LLM`.

Structure:
    - Outermost list: contains multiple generation choices when sampling (`n` sequences)
    - Middle list: represents each position in the generated sequence
    - Innermost list: contains the log probabilities for each token in the vocabulary at that position
"""


class TokenCount(TypedDict):
    input_tokens: List[int]
    output_tokens: List[int]


LLMStatistics = Union[TokenCount, Dict[str, Any]]
"""Initially the LLMStatistics will contain the token count, but can have more variables.
They can be added once we have them defined for every LLM.
"""


class GenerateOutput(TypedDict):
    generations: LLMOutput
    statistics: LLMStatistics
    logprobs: NotRequired[LLMLogprobs]


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
FormattedInput = Union[StandardInput, StructuredInput, str]
"""FormattedInput is an alias for the union of `StandardInput` and `StructuredInput` as generated
by `format_input` and expected by the `LLM`s, as well as `ConversationType` for the vision language models."""


if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray

    GenericFloat = TypeVar("GenericFloat", bound=floating[Any])

    HiddenState = NDArray[GenericFloat]
else:
    HiddenState = Any
