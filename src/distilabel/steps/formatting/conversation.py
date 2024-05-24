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

from typing import TYPE_CHECKING, List

from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class ConversationTemplate(Step):
    """Generate a conversation template from an instruction and a response.

    Input columns:
        - instruction (`str`): The instruction to be used in the conversation.
        - response (`str`): The response to be used in the conversation.

    Output columns:
        - conversation (`ChatType`): The conversation template.

    Categories:
        - format
        - chat
        - template
    """

    @property
    def inputs(self) -> List[str]:
        """The instruction and response."""
        return ["instruction", "response"]

    @property
    def outputs(self) -> List[str]:
        """The conversation template."""
        return ["conversation"]

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        """Generate a conversation template from an instruction and a response.

        Args:
            inputs: The input data.

        Yields:
            The input data with the conversation template.
        """
        for input in inputs:
            input["conversation"] = [
                {"role": "user", "content": input["instruction"]},
                {"role": "assistant", "content": input["response"]},
            ]
        yield inputs
