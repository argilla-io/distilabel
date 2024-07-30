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
    """Simple text generation with an `LLM` given an instruction.

    `TextGeneration` is a pre-defined task that defines the `instruction` as the input
    and `generation` as the output. This task is used to generate text based on the input
    instruction. The model_name is also returned as part of the output in order to enhance it.

    Attributes:
        use_system_prompt: Whether to use the system prompt in the generation. Defaults to `True`,
            which means that if the column `system_prompt` is defined within the input batch, then
            the `system_prompt` will be used, otherwise, it will be ignored.

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

    use_system_prompt: bool = True

    @property
    def inputs(self) -> List[str]:
        """The input for the task is the `instruction`."""
        return ["instruction"]

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""

        if is_openai_format(input["instruction"]):
            raise ValueError(
                "Providing `instruction` formatted as an OpenAI chat / conversation is"
                " deprecated, you should use `ChatGeneration` with `messages` as input instead.",
            )

        if not isinstance(input["instruction"], str):
            raise ValueError(
                f"Input `instruction` must be a string. Got: {input['instruction']}."
            )

        messages = [{"role": "user", "content": input["instruction"]}]
        if self.use_system_prompt:
            if "system_prompt" in input:
                messages.insert(
                    0, {"role": "system", "content": input["system_prompt"]}
                )
            else:
                warnings.warn(
                    "`use_system_prompt` is set to `True`, but no `system_prompt` in input batch, so it will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )
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

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        """The input is formatted as a `ChatType` assuming that the messages provided
        are already formatted that way i.e. following the OpenAI chat format."""

        if not is_openai_format(input["messages"]):
            raise ValueError(
                "Input `messages` must be an OpenAI chat-like format conversation. "
                f"Got: {input['messages']}. Please check: 'https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models'."
            )

        if input["messages"][-1]["role"] != "user":
            raise ValueError(
                "The last message must be from the user. Please check: "
                "'https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models'."
            )

        return input["messages"]

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
