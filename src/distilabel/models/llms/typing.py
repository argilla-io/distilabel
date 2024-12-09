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

from typing import TYPE_CHECKING, Any, Dict, List, TypedDict, TypeVar, Union

from typing_extensions import NotRequired

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


if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray

    GenericFloat = TypeVar("GenericFloat", bound=floating[Any])

    HiddenState = NDArray[GenericFloat]
else:
    HiddenState = Any
