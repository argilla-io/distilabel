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

from typing import TYPE_CHECKING, Any, Dict, List, Union

from pydantic import Field, PositiveInt

from distilabel.llms.mixins.magpie import MagpieChatTemplateMixin
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.base import StepInput
from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType, FormattedInput
    from distilabel.steps.typing import StepOutput


class Magpie(Task):
    n_turns: RuntimeParameter[PositiveInt] = Field(
        default=1,
        description="The number of turns to generate for the conversation.",
    )

    def model_post_init(self, _: Any) -> None:
        """Checks that the provided `LLM` uses the `MagpieChatTemplateMixin`."""
        super().load()

        if not isinstance(self.llm, MagpieChatTemplateMixin):
            raise ValueError(
                f"`Magpie` task can only be used with an `LLM` that uses the `MagpieChatTemplateMixin`."
                f"`{self.llm.__class__.__name__}` doesn't use the aforementioned mixin."
            )

        self.llm.use_magpie_template = True

    @property
    def inputs(self) -> List[str]:
        return []

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        return []

    @property
    def outputs(self) -> List[str]:
        return ["conversation"]

    def format_output(
        self,
        output: Union[str, None],
        input: Union[Dict[str, Any], None] = None,
    ) -> Dict[str, Any]:
        return {}

    def _prepare_inputs_for_instruction_generation(
        self, inputs: List[Dict[str, Any]]
    ) -> List["FormattedInput"]:
        return [
            [{"role": "system", "content": input["system_prompt"]}]
            if "system_prompt" in input
            else []
            for input in inputs
        ]

    def _append_messages_to_conversations(
        self, role: str, messages: List[str], conversations: List["ChatType"]
    ) -> List["ChatType"]:
        for instruction, conversation in zip(messages, conversations):
            conversation.append({"role": role, "content": instruction})
        return conversations

    def process(self, inputs: StepInput) -> "StepOutput":
        conversations = self._prepare_inputs_for_instruction_generation(inputs=inputs)

        for _ in range(self.n_turns):  # type: ignore
            # Generate instruction or user message
            outputs = self.llm.generate(
                inputs=conversations,
                num_generations=1,
                **self.llm.generation_kwargs,  # type: ignore
            )

            conversations = self._append_messages_to_conversations(
                role="user",
                messages=[output[0] for output in outputs],
                conversations=conversations,  # type: ignore
            )

            # Generate response
            outputs = self.llm.generate(
                inputs=conversations,
                num_generations=1,
                **self.llm.generation_kwargs,  # type: ignore
            )

            conversations = self._append_messages_to_conversations(
                role="assistant",
                messages=[output[0] for output in outputs],
                conversations=conversations,  # type: ignore
            )

        yield [{"conversation": conversation} for conversation in conversations]
