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

from typing import TYPE_CHECKING, Callable, List, Optional, Union

from distilabel.steps.tasks.typing import ChatType

if TYPE_CHECKING:
    from distilabel.models.llms.typing import GenerateOutput, LLMLogprobs, LLMOutput


def compute_tokens(
    text_or_messages: Union[str, ChatType], tokenizer: Callable[..., List[int]]
) -> int:
    """Helper function to count the number of tokens in a text or list of messages.

    Args:
        text_or_messages: Either a string response or a list of messages.
        tokenizer: A callable function that take str and returns the tokenized version of the text.

    Returns:
        The number of tokens.
    """
    if isinstance(text_or_messages, list):
        return sum([len(tokenizer(message["content"])) for message in text_or_messages])
    else:
        return len(tokenizer(text_or_messages))


def prepare_output(
    generations: "LLMOutput",
    input_tokens: Optional[List[int]] = None,
    output_tokens: Optional[List[int]] = None,
    logprobs: Optional["LLMLogprobs"] = None,
) -> "GenerateOutput":
    """Helper function to prepare the output of the LLM.

    Args:
        generations: The outputs from an LLM.
        input_tokens: The number of tokens of the inputs. Defaults to `None`.
        output_tokens: The number of tokens of the LLM response. Defaults to `None`.

    Returns:
        Output generation from an LLM.
    """
    output: "GenerateOutput" = {
        "generations": generations,
        "statistics": {
            "input_tokens": input_tokens or [],
            "output_tokens": output_tokens or [],
        },
    }
    if logprobs is not None:
        output["logprobs"] = logprobs
    return output
