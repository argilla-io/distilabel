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

from typing import TYPE_CHECKING, Literal

from jinja2 import Template
from pydantic import Field

from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.text_generation import (
    TextGeneration,
    check_column_in_template,
)

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ConversationType
    from distilabel.steps.typing import StepColumns


class VisionGeneration(TextGeneration):
    """Vision generation with an `LLM` given a prompt.

    `VisionGeneration` is a pre-defined task that allows passing a custom prompt using the
    Jinja2 syntax. By default, a `instruction` is expected in the inputs, but the using
    `template` and `columns` attributes one can define a custom prompt and columns expected
    from the text. Additionally, an `image` column is expected containing one of the
    url, base64 encoded image or PIL image.

    Attributes:
        system_prompt: The system prompt to use in the generation.
            If not, then no system prompt will be used. Defaults to `None`.
        template: The template to use for the generation. It must follow the Jinja2 template
            syntax. If not provided, it will assume the text passed is an instruction and
            construct the appropriate template.
        columns: A string with the column, or a list with columns expected in the template.
            Take a look at the examples for more information. Defaults to `instruction`.
        image_type: The type of the image provided, this will be used to preprocess if necessary.
            Must be one of "url", "base64" or "PIL".

    Input columns:
        - dynamic (determined by `columns` attribute): By default will be set to `instruction`.
            The columns can point both to a `str` or a `list[str]` to be used in the template.
        - image: The column containing the image URL, base64 encoded image or PIL image.

    Output columns:
        - generation (`str`): The generated text.
        - model_name (`str`): The name of the model used to generate the text.

    Categories:
        - vision-generation

    References:
        - [Jinja2 Template Designer Documentation](https://jinja.palletsprojects.com/en/3.1.x/templates/)

    Examples:
        Generate text from an instruction:

        ```python
        from distilabel.steps.tasks import TextGeneration
        from distilabel.models import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        text_gen = TextGeneration(
            llm=InferenceEndpointsLLM(
                model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
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
        #         'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
        #         'generation': 'generation',
        #     }
        # ]
        ```

    """

    image_type: Literal["url", "base64", "PIL"] = Field(
        default=...,
        description="The type of the image provided, this will be used to preprocess if necessary.",
    )

    @property
    def inputs(self) -> "StepColumns":
        columns = super().inputs
        columns["image"] = True
        return columns

    def load(self) -> None:
        Task.load(self)

        for column in self.columns:
            check_column_in_template(
                column, self.template, page="components-gallery/tasks/visiongeneration/"
            )

        self._template = Template(self.template)

    def _prepare_message_content(self, input: dict[str, any]) -> "ConversationType":
        """Prepares the content for the template and returns the formatted messages."""
        fields = {column: input[column] for column in self.columns}
        # TODO: Any transformation should be done here.
        img_url = input["image"]
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self._template.render(**fields),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url,
                        },
                    },
                ],
            }
        ]

    def format_input(self, input: dict[str, any]) -> "ConversationType":
        """The input is formatted as a `ConversationType` assuming that the instruction
        is the first interaction from the user within a conversation."""
        messages = self._prepare_message_content(input)

        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        return messages  # type: ignore
