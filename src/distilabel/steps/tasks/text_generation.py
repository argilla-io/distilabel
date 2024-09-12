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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from jinja2 import Template
from pydantic import Field, PrivateAttr

from distilabel.errors import DistilabelUserError
from distilabel.steps.tasks.base import Task
from distilabel.utils.chat import is_openai_format

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import StepColumns


class TextGeneration(Task):
    """Simple text generation with an `LLM` given an instruction.

    `TextGeneration` is a pre-defined task that defines the `instruction` as the input
    and `generation` as the output. This task is used to generate text based on the input
    instruction. The model_name is also returned as part of the output in order to enhance it.

    Attributes:
        system_prompt: The system prompt to use in the generation. If not provided, then
            it will check if the input row has a column named `system_prompt` and use it.
            If not, then no system prompt will be used. Defaults to `None`.
        use_system_prompt: DEPRECATED. To be removed in 1.5.0. Whether to use the system
            prompt in the generation. Defaults to `True`, which means that if the column
            `system_prompt` is defined within the input batch, then the `system_prompt`
            will be used, otherwise, it will be ignored.

    Input columns:
        - instruction (`str`): The instruction to generate text from.

    Output columns:
        - generation (`str`): The generated text.
        - model_name (`str`): The name of the model used to generate the text.

    Categories:
        - text-generation

    Examples:
        Generate text from an instruction:

        ```python
        from distilabel.steps.tasks import TextGeneration
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        text_gen = TextGeneration(
            llm=InferenceEndpointsLLM(
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
            )
        )

        text_gen.load()

        result = next(
            text_gen.process(
                [{"instruction": "your instruction"}]
            )
        )
        # result
        # [
        #     {
        #         'instruction': 'your instruction',
        #         'model_name': 'mistralai/Mistral-7B-Instruct-v0.2',
        #         'generation': 'generation',
        #     }
        # ]
        ```
    """

    system_prompt: Union[str, None] = None
    use_system_prompt: bool = Field(default=True, deprecated=True)
    template: Optional[str] = Field(
        default=None,
        description=(
            "This is a template or prompt to use for the generation. "
            "If not provided the instruction will be used as is."
        ),
    )
    extra_columns: Optional[List[str]] = Field(
        default=None,
        description=(
            "Extra columns to include in the input. If a `template` is provided which needs "
            "additional columns, then they should be provided here."
        ),
    )

    _can_be_used_with_offline_batch_generation = True
    _template: Optional[Template] = PrivateAttr(default=None)

    def load(self) -> None:
        super().load()
        self._template = Template(self.template or "{{ instruction }}")
        # TODO: Make a quick check for the extra columns here

    @property
    def inputs(self) -> "StepColumns":
        """The input for the task is the `instruction`."""
        columns = {"instruction": True, "system_prompt": False}
        if self.extra_columns:
            columns.update({column: True for column in self.extra_columns})
        return columns

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""

        if is_openai_format(input["instruction"]):
            raise DistilabelUserError(
                "Providing `instruction` formatted as an OpenAI chat / conversation is"
                " deprecated, you should use `ChatGeneration` with `messages` as input instead.",
                page="components-gallery/tasks/textgeneration/",
            )

        if not isinstance(input["instruction"], str):
            raise DistilabelUserError(
                f"Input `instruction` must be a string. Got: {input['instruction']}.",
                page="components-gallery/tasks/textgeneration/",
            )

        fields = {"instruction": input["instruction"]}
        fields.update({column: input[column] for column in self.extra_columns or []})
        messages = [{"role": "user", "content": self._template.render(**fields)}]

        row_system_prompt = input.get("system_prompt")
        if row_system_prompt:
            messages.insert(0, {"role": "system", "content": row_system_prompt})

        if self.system_prompt and not row_system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        return messages  # type: ignore

    @property
    def outputs(self) -> List[str]:
        """The output for the task is the `generation` and the `model_name`."""
        return ["generation", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        """The output is formatted as a dictionary with the `generation`. The `model_name`
        will be automatically included within the `process` method of `Task`."""
        return {"generation": output}


class ChatGeneration(Task):
    """Generates text based on a conversation.

    `ChatGeneration` is a pre-defined task that defines the `messages` as the input
    and `generation` as the output. This task is used to generate text based on a conversation.
    The `model_name` is also returned as part of the output in order to enhance it.

    Input columns:
        - messages (`List[Dict[Literal["role", "content"], str]]`): The messages to generate the
            follow up completion from.

    Output columns:
        - generation (`str`): The generated text from the assistant.
        - model_name (`str`): The model name used to generate the text.

    Categories:
        - chat-generation

    Icon:
        `:material-chat:`

    Examples:
        Generate text from a conversation in OpenAI chat format:

        ```python
        from distilabel.steps.tasks import ChatGeneration
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        chat = ChatGeneration(
            llm=InferenceEndpointsLLM(
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
            )
        )

        chat.load()

        result = next(
            chat.process(
                [
                    {
                        "messages": [
                            {"role": "user", "content": "How much is 2+2?"},
                        ]
                    }
                ]
            )
        )
        # result
        # [
        #     {
        #         'messages': [{'role': 'user', 'content': 'How much is 2+2?'}],
        #         'model_name': 'mistralai/Mistral-7B-Instruct-v0.2',
        #         'generation': '4',
        #     }
        # ]
        ```
    """

    @property
    def inputs(self) -> List[str]:
        """The input for the task are the `messages`."""
        return ["messages"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """The input is formatted as a `ChatType` assuming that the messages provided
        are already formatted that way i.e. following the OpenAI chat format."""

        if not is_openai_format(input["messages"]):
            raise DistilabelUserError(
                "Input `messages` must be an OpenAI chat-like format conversation. "
                f"Got: {input['messages']}. Please check: 'https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models'.",
                page="components-gallery/tasks/chatgeneration/",
            )

        if input["messages"][-1]["role"] != "user":
            raise DistilabelUserError(
                "The last message must be from the user. Please check: "
                "'https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models'.",
                page="components-gallery/tasks/chatgeneration/",
            )

        return input["messages"]

    @property
    def outputs(self) -> List[str]:
        """The output for the task is the `generation` and the `model_name`."""
        return ["generation", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        """The output is formatted as a dictionary with the `generation`. The `model_name`
        will be automatically included within the `process` method of `Task`."""
        return {"generation": output}
