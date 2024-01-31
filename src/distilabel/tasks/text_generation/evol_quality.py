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


from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from distilabel.tasks.base import get_template
from distilabel.tasks.prompt import Prompt
from distilabel.tasks.text_generation.evol_instruct import EvolInstructTask

_EVOL_QUALITY_TEMPLATE = get_template("evol-quality.jinja2")


EvolutionMethod = Literal[
    "helpfulness",
    "relevance",
    "depth",
    "creativity",
    "details",
]


@dataclass
class EvolQualityTask(EvolInstructTask):
    """A `TextGenerationTask` following the `Deita` specification for improving the responses.

    From the reference repository: *DEITA (short for Data-Efficient Instruction Tuning for Alignment),
    a series of models fine-tuned from LLaMA and Mistral models using data samples automatically selected with our proposed approach*

    The task is defined as follows:
    Starting from an initial (simpler) instruction response, select an evolving-method to upgrade the simple response
    to a more complex one or create..
    The Evolving methods includes the following operations: add "helpfulness", "relevance", "deepen", "creativity" and "details".

    Given the evolved reposnes are generated from LLMs, sometimes the evolving will fail. We adopt an responses eliminator
    to filter the failed instructions, called Elimination Evolving, but we don't apply the step of asking again to the LLM it the
    answer is a copy from the same used prompt. Note that we slightly modify the elimination evolving step, from the original paper,
    to allow for filtering of the responses.

    This evolutionary process can be repeated for several rounds to obtain instruction data containing various complexities.
    Currently the task is implemented as a single step, so to generate multiple evolutions you can "repeat" the instructions
    in the original dataset. An example of a similar implementation with EvolInstruct can be seen at the following script:
    [examples/pipeline-evol-instruct-alpaca.py](https://github.com/argilla-io/distilabel/tree/main/examples/pipeline-evol-instruct-alpaca.py)

    Args:
        system_prompt (str, optional): the system prompt to be used. Not defined for this task.

    References:
        - [`What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning`](https://arxiv.org/abs/2312.15685)
        - [`WizardLM: Empowering Large Language Models to Follow Complex Instructions`](https://arxiv.org/abs/2304.12244)
    """

    system_prompt: str = ""

    __jinja2_template__: str = _EVOL_QUALITY_TEMPLATE

    def generate_prompt(
        self,
        input: str,
        generation: str,
        evolution_method: Optional[EvolutionMethod] = None,
        **_: Any,
    ) -> Prompt:
        """Generates a prompt following the Evol-Instruct specification.

        Args:
            input (str): the input to be used for the prompt.
            evolution_method (str, optional): The evolution method to be used. If not provided (the default), a random one is chosen
                like the original paper. Available ones are "helpfulness", "relevance", "deepen", "creativity" and "details".

        Returns:
            Prompt: the generated prompt.

        Examples:
            >>> from distilabel.tasks.text_generation import EvolQualityGeneratorTask
            >>> task = EvolQualityGeneratorTask()
            >>> task.generate_prompt("Give three tips for staying healthy.", "1. Eat healthy food. 2. Exercise. 3. Sleep well.")
            Prompt(
                system_prompt="",
                formatted_prompt="I want you to act as a Prompt ...",
            )
        """
        evolution_method = self._get_evolution_method(evolution_method, EvolutionMethod)

        render_kwargs = {
            "evol_method": evolution_method,
            "instruction": input,
            "generation": generation,
        }
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        )

    @property
    def input_args_names(self) -> List[str]:
        return ["input", "generation"]

    @property
    def output_args_names(self) -> List[str]:
        return ["generations"]

    def parse_output(self, output: str) -> Dict[str, List[str]]:
        """Parses the output of the model into the desired format, applying the elimination step for bad generations.

        Args:
            output (str): the output of the model.

        Note:
            The elimination step is applied to the output, but only steps 2-4 in the paper are implemented.
            Refer to point 3.2, Elimination Evolving section in [`WizardLM: Empowering Large Language Models to Follow Complex Instructions`](https://arxiv.org/abs/2304.12244)
            for more information on the elimination evolving step, and take a look at the `_elimination_evolving`
            method for more information of the implementation.
        """
        response_words = {
            "#Given Response#",
            "#Created Response#",
            "given response",
            "created response",
            "#The Given Response#",
            "#Rewritten Response#",
            "rewritten response",
        }
        output = self._elimination_evolving(output, response_words=response_words)
        return {self.output_args_names[0]: output}
