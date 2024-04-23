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

import warnings
from typing import Any, Dict, List, Union

from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.typing import ChatType
from distilabel.utils.chat import is_openai_format


class TextGeneration(Task):
    """TextGeneration is a pre-defined task that defines the `instruction` as the input
    and `generation` as the output. This task is used to generate text based on the input
    instruction. The model_name is also returned as part of the output in order to enhance it.

    Input columns:
        - instruction (`str`): The instruction to generate text from.

    Output columns:
        - generation (`str`): The generated text.
        - model_name (`str`): The model name used to generate the text.
    """

    @property
    def inputs(self) -> List[str]:
        """The input for the task is the `instruction`."""
        return ["instruction"]

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""

        if is_openai_format(input["instruction"]):
            warnings.warn(
                "Providing `instruction` formatted as an OpenAI chat / conversation is"
                " about to be deprecated in `distilabel v1.2.0`, please make sure to use"
                " `ChatTextGeneration` with `messages` as input instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return input["instruction"]

        if not isinstance(input["instruction"], str):
            raise ValueError(
                f"Input `instruction` must be a string. Got: {input['instruction']}."
            )

        return [{"role": "user", "content": input["instruction"]}]

    @property
    def outputs(self) -> List[str]:
        """The output for the task is the `generation` and the `model_name`."""
        return ["generation", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """The output is formatted as a dictionary with the `generation`. The `model_name`
        will be automatically included within the `process` method of `Task`."""
        return {"generation": output}


class ChatGeneration(TextGeneration):
    """ChatGeneration is a pre-defined task that defines the `messages` as the input
    and `generation` as the output. This task is used to generate text based on a conversation.
    The `model_name` is also returned as part of the output in order to enhance it.

    Input columns:
        - messages (`List[Dict[Literal["role", "content"], str]]`): The messages to generate the
            follow up completion from.

    Output columns:
        - generation (`str`): The generated text from the assistant.
        - model_name (`str`): The model name used to generate the text.
    """

    @property
    def inputs(self) -> List[str]:
        """The input for the task are the `messages`."""
        return ["messages"]

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        """The input is formatted as a `ChatType` assuming that the messages provided
        are already formatted that way i.e. following the OpenAI chat format."""

        if not is_openai_format(input["messages"]):
            raise ValueError(
                "Input `instruction` must be a string or an OpenAI chat-like format. "
                f"Got: {input['messages']}. Please check: 'https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models'."
            )

        if input["messages"][-1]["role"] != "user":
            raise ValueError(
                "The last message must be from the user. Please check: "
                "'https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models'."
            )

        return input["messages"]
