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

from typing import TYPE_CHECKING, Any, Dict, List

from distilabel.llm.base import LLM
from distilabel.steps.base import Step

if TYPE_CHECKING:
    from distilabel.steps.base import StepInput
    from distilabel.steps.task.typing import ChatType
    from distilabel.steps.typing import StepOutput


class GenerateEmbeddings(Step):
    """Generate embeddings for a text input using the last hidden state of an `LLM`, as
    described in the paper 'What Makes Good Data for Alignment? A Comprehensive Study of
    Automatic Data Selection in Instruction Tuning'.

    Input columns:
        text (`str`, `List[Dict[str, str]]`): The input text or conversation to generate
            embeddings for.

    Output columns:
        embedding (`List[float]`): The embedding of the input text or conversation.

    Reference:
        - https://arxiv.org/abs/2312.15685
    """

    @property
    def inputs(self) -> List[str]:
        return ["text"]

    @property
    def outputs(self) -> List[str]:
        return ["embedding"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        text = input["text"]

        # input is in `ChatType` format
        if isinstance(text, list):
            return text

        return [{"role": "user", "content": text}]

    def process(self, inputs: "StepInput") -> "StepOutput":  # type: ignore
        formatted_inputs = [self.format_input(input) for input in inputs]
        last_hidden_states = self.llm.get_last_hidden_states(formatted_inputs)
        for input, hidden_state in zip(inputs, last_hidden_states):
            input["embedding"] = hidden_state[-1].tolist()
        yield inputs
