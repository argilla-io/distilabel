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

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, get_args

from distilabel.tasks.base import get_template
from distilabel.tasks.prompt import Prompt
from distilabel.tasks.text_generation.base import TextGenerationTask

_EVOL_INSTRUCT_TEMPLATE = get_template("evol-instruct.jinja2")


EvolutionMethod = Literal[
    "breadth", "constraints", "deepen", "concretizing", "reasoning"
]


@dataclass
class EvolInstructTask(TextGenerationTask):
    """A `TextGenerationTask` following the `EvolInstruct` specification for building the prompts.

    Args:
        system_prompt (str, optional): the system prompt to be used. Not defined for this task.

    Notes:
        - The paper mentions a parameter *M* for the number of evolutions. This behavior can be achieved dealing
        with the dataset directly, generating copies of the inputs. Take a look at
        [examples/pipeline-evol-instruct-alpaca.py](https://github.com/argilla-io/distilabel/tree/main/examples/pipeline-evol-instruct-alpaca.py)
        script for an example.
        - The original implementation includes an elimination step, which is not implemented currently.

    References:
        - [`WizardLM: Empowering Large Language Models to Follow Complex Instructions`](https://arxiv.org/pdf/2304.12244.pdf)
    """

    system_prompt: str = ""

    __jinja2_template__: str = _EVOL_INSTRUCT_TEMPLATE

    def generate_prompt(
        self, input: str, evolution_method: Optional[EvolutionMethod] = None, **_: Any
    ) -> Prompt:
        """Generates a prompt following the Evol-Instruct specification.

        Args:
            input (str): the input to be used for the prompt.
            evolution_method (str, optional): The evolution method to be used. If not provided (the default), a random one is chosen
                like the original paper. Available ones are "breadth", "constraints", "deepen", "concretizing" and "reasoning".

        Returns:
            Prompt: the generated prompt.

        Examples:
            >>> from distilabel.tasks.text_generation import EvolInstructTask
            >>> task = EvolInstructTask()
            >>> task.generate_prompt("Give three tips for staying healthy.")
            Prompt(
                system_prompt="",
                formatted_prompt="I want you to act as a Prompt ...",
            )
        """
        if not evolution_method:
            evolution_method = random.choice(get_args(EvolutionMethod))

        render_kwargs = {
            "evol_method": evolution_method,
            "instruction": input,
        }
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        )

    @property
    def output_args_names(self) -> List[str]:
        return ["instruction"]

    def parse_output(self, output: str) -> Dict[str, List[str]]:
        """Parses the output of the model into the desired format."""
        return {"instruction": output}
