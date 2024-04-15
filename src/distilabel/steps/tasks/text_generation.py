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

        instruction = input["instruction"]

        if isinstance(instruction, str):
            return [{"role": "user", "content": input["instruction"]}]

        if not is_openai_format(instruction):
            raise ValueError(
                f"Input `instruction` must be a string or an OpenAI chat-like format. "
                f"Got: {instruction}. Please check: 'https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models'."
            )

        return instruction

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
