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

import importlib.resources as importlib_resources
from typing import TYPE_CHECKING, Any, Dict, Union

from jinja2 import Template

from distilabel.steps.tasks import Task

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import StepColumns


class URIAL(Task):
    """Generates a response using a non-instruct fine-tuned model.

    `URIAL` is a pre-defined task that generates a response using a non-instruct fine-tuned
    model. This task is used to generate a response based on the conversation provided as
    input.

    Input columns:
        - instruction (`str`, optional): The instruction to generate a response from.
        - conversation (`List[Dict[str, str]]`, optional): The conversation to generate
            a response from (the last message must be from the user).

    Output columns:
        - generation (`str`): The generated response.
        - model_name (`str`): The name of the model used to generate the response.

    Categories:
        - text-generation

    References:
        - [The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning](https://arxiv.org/abs/2312.01552)

    Examples:
        Generate text from an instruction:

        ```python
        from distilabel.models import vLLM
        from distilabel.steps.tasks import URIAL

        step = URIAL(
            llm=vLLM(
                model="meta-llama/Meta-Llama-3.1-8B",
                generation_kwargs={"temperature": 0.7},
            ),
        )

        step.load()

        results = next(
            step.process(inputs=[{"instruction": "What's the most most common type of cloud?"}])
        )
        # [
        #     {
        #         'instruction': "What's the most most common type of cloud?",
        #         'generation': 'Clouds are classified into three main types, high, middle, and low. The most common type of cloud is the middle cloud.',
        #         'distilabel_metadata': {...},
        #         'model_name': 'meta-llama/Meta-Llama-3.1-8B'
        #     }
        # ]
        ```
    """

    def load(self) -> None:
        """Loads the Jinja2 template for the given `aspect`."""
        super().load()

        _path = str(
            importlib_resources.files("distilabel")
            / "steps"
            / "tasks"
            / "templates"
            / "urial.jinja2"
        )

        self._template = Template(open(_path).read())

    @property
    def inputs(self) -> "StepColumns":
        return {"instruction": False, "conversation": False}

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        messages = (
            [{"role": "user", "content": input["instruction"]}]
            if "instruction" in input
            else input["conversation"]
        )

        if messages[-1]["role"] != "user":
            raise ValueError("The last message must be from the user.")

        return [{"role": "user", "content": self._template.render(messages=messages)}]

    @property
    def outputs(self) -> "StepColumns":
        return ["generation", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        if output is None:
            return {"generation": None}

        response = output.split("\n\n# User")[0]
        if response.startswith("\n\n"):
            response = response[2:]
        response = response.strip()

        return {"generation": response}
