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

import sys

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from jinja2 import Template
from pydantic import PrivateAttr

from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType


class SelfInstruct(Task):
    """Generate instructions based on a given input using an `LLM`.

    `SelfInstruct` is a pre-defined task that, given a number of instructions, a
    certain criteria for query generations, an application description, and an input,
    generates a number of instruction related to the given input and following what
    is stated in the criteria for query generation and the application description.
    It is based in the SelfInstruct framework from the paper "Self-Instruct: Aligning
    Language Models with Self-Generated Instructions".

    Attributes:
        num_instructions: The number of instructions to be generated. Defaults to 5.
        criteria_for_query_generation: The criteria for the query generation. Defaults
            to the criteria defined within the paper.
        application_description: The description of the AI application that one want
            to build with these instructions. Defaults to `AI assistant`.

    Input columns:
        - input (`str`): The input to generate the instructions. It's also called seed in
            the paper.

    Output columns:
        - instructions (`List[str]`): The generated instructions.
        - model_name (`str`): The model name used to generate the instructions.

    Categories:
        - text-generation

    Reference:
        - [`Self-Instruct: Aligning Language Models with Self-Generated Instructions`](https://arxiv.org/abs/2212.10560)

    Examples:
        Generate instructions based on a given input:

        ```python
        from distilabel.steps.tasks import SelfInstruct
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        self_instruct = SelfInstruct(
            llm=InferenceEndpointsLLM(
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
            ),
            num_instructions=5,  # This is the default value
        )

        self_instruct.load()

        result = next(self_instruct.process([{"input": "instruction"}]))
        # result
        # [
        #     {
        #         'input': 'instruction',
        #         'model_name': 'mistralai/Mistral-7B-Instruct-v0.2',
        #         'instructions': ["instruction 1", "instruction 2", "instruction 3", "instruction 4", "instruction 5"],
        #     }
        # ]
        ```

    Citations:
        ```
        @misc{wang2023selfinstructaligninglanguagemodels,
            title={Self-Instruct: Aligning Language Models with Self-Generated Instructions},
            author={Yizhong Wang and Yeganeh Kordi and Swaroop Mishra and Alisa Liu and Noah A. Smith and Daniel Khashabi and Hannaneh Hajishirzi},
            year={2023},
            eprint={2212.10560},
            archivePrefix={arXiv},
            primaryClass={cs.CL},
            url={https://arxiv.org/abs/2212.10560},
        }
        ```
    """

    num_instructions: int = 5
    criteria_for_query_generation: str = (
        "Incorporate a diverse range of verbs, avoiding repetition.\n"
        "Ensure queries are compatible with AI model's text generation functions and are limited to 1-2 sentences.\n"
        "Design queries to be self-contained and standalone.\n"
        'Blend interrogative (e.g., "What is the significance of x?") and imperative (e.g., "Detail the process of x.") styles.'
    )
    application_description: str = "AI assistant"

    _template: Union[Template, None] = PrivateAttr(...)

    def load(self) -> None:
        """Loads the Jinja2 template."""
        super().load()

        _path = str(
            importlib_resources.files("distilabel")
            / "steps"
            / "tasks"
            / "templates"
            / "self-instruct.jinja2"
        )

        self._template = Template(open(_path).read())

    @property
    def inputs(self) -> List[str]:
        """The input for the task is the `input` i.e. seed text."""
        return ["input"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""
        return [
            {
                "role": "user",
                "content": self._template.render(
                    input=input["input"],
                    application_description=self.application_description,
                    criteria_for_query_generation=self.criteria_for_query_generation,
                    num_instructions=self.num_instructions,
                ),
            }
        ]

    @property
    def outputs(self):
        """The output for the task is a list of `instructions` containing the generated instructions."""
        return ["instructions", "model_name"]

    def format_output(
        self,
        output: Union[str, None],
        input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """The output is formatted as a list with the generated instructions.

        Args:
            output: the raw output of the LLM.
            input: the input to the task. Used for obtaining the number of responses.

        Returns:
            A dict with containing the generated instructions.
        """
        if output is None:
            return {"instructions": []}
        return {"instructions": [line for line in output.split("\n") if line != ""]}
