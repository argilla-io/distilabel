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
        - [Image-Text-to-Text](https://huggingface.co/tasks/image-text-to-text)
        - [OpenAI Vision](https://platform.openai.com/docs/guides/vision)

    Examples:
        Generate text from an instruction:

        ```python
        from distilabel.steps.tasks import VisionGeneration
        from distilabel.models.llms import InferenceEndpointsLLM

        vision = VisionGeneration(
            name="vision_gen",
            llm=InferenceEndpointsLLM(
                model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
            ),
            image_type="url"
        )

        vision.load()

        result = next(
            vision.process(
                [
                    {
                        "instruction": "What’s in this image?",
                        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    }
                ]
            )
        )
        # result
        # [
        #     {
        #         "instruction": "What\u2019s in this image?",
        #         "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        #         "generation": "Based on the visual cues in the image...",
        #         "model_name": "meta-llama/Llama-3.2-11B-Vision-Instruct"
        #         ... # distilabel_metadata would be here
        #     }
        # ]
        # result[0]["generation"]
        # "Based on the visual cues in the image, here are some possible story points:\n\n* The image features a wooden boardwalk leading through a lush grass field, possibly in a park or nature reserve.\n\nAnalysis and Ideas:\n* The abundance of green grass and trees suggests a healthy ecosystem or habitat.\n* The presence of wildlife, such as birds or deer, is possible based on the surroundings.\n* A footbridge or a pathway might be a common feature in this area, providing access to nearby attractions or points of interest.\n\nAdditional Questions to Ask:\n* Why is a footbridge present in this area?\n* What kind of wildlife inhabits this region"
        ```

    """

    image_type: Literal["url", "base64", "PIL"] = Field(
        default="url",
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
