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

from distilabel.llms.base import LLM
from distilabel.steps.base import Step, StepInput
from distilabel.utils.chat import is_openai_format

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import StepOutput


class GenerateEmbeddings(Step):
    """Generate embeddings using the last hidden state of an `LLM`.

    Generate embeddings for a text input using the last hidden state of an `LLM`, as
    described in the paper 'What Makes Good Data for Alignment? A Comprehensive Study of
    Automatic Data Selection in Instruction Tuning'.

    Attributes:
        llm: The `LLM` to use to generate the embeddings.

    Input columns:
        - text (`str`, `List[Dict[str, str]]`): The input text or conversation to generate
            embeddings for.

    Output columns:
        - embedding (`List[float]`): The embedding of the input text or conversation.
        - model_name (`str`): The model name used to generate the embeddings.

    Categories:
        - embedding
        - llm

    References:
        - [What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning](https://arxiv.org/abs/2312.15685)

    Examples:

        Rank LLM candidates:

        ```python
        from distilabel.steps.tasks import GenerateEmbeddings
        from distilabel.llms.huggingface import TransformersLLM

        # Consider this as a placeholder for your actual LLM.
        embedder = GenerateEmbeddings(
            llm=TransformersLLM(
                model="TaylorAI/bge-micro-v2",
                model_kwargs={"is_decoder": True},
                cuda_devices=[],
            )
        )
        embedder.load()

        result = next(
            embedder.process(
                [
                    {"text": "Hello, how are you?"},
                ]
            )
        )
        ```

    Citations:

        ```
        @misc{liu2024makesgooddataalignment,
            title={What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning},
            author={Wei Liu and Weihao Zeng and Keqing He and Yong Jiang and Junxian He},
            year={2024},
            eprint={2312.15685},
            archivePrefix={arXiv},
            primaryClass={cs.CL},
            url={https://arxiv.org/abs/2312.15685},
        }
        ```
    """

    llm: LLM

    def load(self) -> None:
        """Loads the `LLM` used to generate the embeddings."""
        super().load()

        self.llm.load()

    @property
    def inputs(self) -> List[str]:
        """The inputs for the task is a `text` column containing either a string or a
        list of dictionaries in OpenAI chat-like format."""
        return ["text"]

    @property
    def outputs(self) -> List[str]:
        """The outputs for the task is an `embedding` column containing the embedding of
        the `text` input."""
        return ["embedding", "model_name"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """Formats the input to be used by the LLM to generate the embeddings. The input
        can be in `ChatType` format or a string. If a string, it will be converted to a
        list of dictionaries in OpenAI chat-like format.

        Args:
            input: The input to format.

        Returns:
            The OpenAI chat-like format of the input.
        """
        text = input["text"] = input["text"]

        # input is in `ChatType` format
        if isinstance(text, str):
            return [{"role": "user", "content": text}]

        if is_openai_format(text):
            return text

        raise ValueError(
            f"Couldn't format input for step {self.name}. The `text` input column has to"
            " be a string or a list of dictionaries in OpenAI chat-like format."
        )

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        """Generates an embedding for each input using the last hidden state of the `LLM`.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Yields:
            A list of Python dictionaries with the outputs of the task.
        """
        formatted_inputs = [self.format_input(input) for input in inputs]
        last_hidden_states = self.llm.get_last_hidden_states(formatted_inputs)
        for input, hidden_state in zip(inputs, last_hidden_states):
            input["embedding"] = hidden_state[-1].tolist()
            input["model_name"] = self.llm.model_name
        yield inputs
