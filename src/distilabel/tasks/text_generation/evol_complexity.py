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
from typing import Any, Literal, Optional

from distilabel.logger import get_logger
from distilabel.tasks.base import get_template
from distilabel.tasks.prompt import Prompt
from distilabel.tasks.text_generation.evol_instruct import EvolInstructTask

logger = get_logger()


# inherits from evol-instruct with limited evolution methods
_EVOL_COMPLEXITY_TEMPLATE = get_template("evol-instruct.jinja2")


EvolutionMethod = Literal["constraints", "deepen", "concretizing", "reasoning"]


@dataclass
class EvolComplexityTask(EvolInstructTask):
    """A `TextGenerationTask` following the `EvolComplexity` specification for building prompts. This is a special case
    of the original EvolInstructTask, where the evolution method is fixed to "constraints", "deepen", "concretizing" or "reasoning".
    Additionally, an additional elimation step should be executed to screen out instructions that are not useful.

    From the reference repository: *Evol-Instruct is a novel method using LLMs instead of humans to automatically mass-produce
    open-domain instructions of various difficulty levels and skills range, to improve the performance of LLMs.*

    The task is defined as follows:
    Starting from an initial (simpler) instruction, select in-depth or in-breadth evolving to upgrade the simple instruction
    to a more complex one or create a new one (to increase diversity).
    The In-depth Evolving includes the following operations: "constraints", "deepen", "concretizing" or "reasoning".
    The In-breadth Evolving is mutation, i.e., generating a completely new instruction based on the given instruction.

    Given the evolved instructions are generated from LLMs, sometimes the evolving will fail. We adopt an instruction eliminator
    to filter the failed instructions, called Elimination Evolving, but we don't apply the step of asking again to the LLM it the
    answer is a copy from the same used prompt.

    This evolutionary process can be repeated for several rounds to obtain instruction data containing various complexities.
    Currently the task is implemented as a single step, so to generate multiple evolutions you can "repeat" the instructions
    in the original dataset. An example can be seen at the following script:
    [examples/pipeline-evol-instruct-alpaca.py](https://github.com/argilla-io/distilabel/tree/main/examples/pipeline-evol-instruct-alpaca.py)

    Args:
        system_prompt (str, optional): the system prompt to be used. Not defined for this task.

    References:
        - [`WizardLM: Empowering Large Language Models to Follow Complex Instructions`](https://arxiv.org/abs/2304.12244)
        - [`What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning`](https://arxiv.org/abs/2312.15685)
    """

    system_prompt: str = ""

    __jinja2_template__: str = _EVOL_COMPLEXITY_TEMPLATE

    def generate_prompt(
        self, input: str, evolution_method: Optional[EvolutionMethod] = None, **_: Any
    ) -> Prompt:
        """Generates a prompt following the Evol-Complexity specification of the Deita Paper.

        Args:
            input (str): the input to be used for the prompt.
            evolution_method (str, optional): The evolution method to be used. If not provided (the default), a random one is chosen
                like the original paper. Available ones are "constraints", "deepen", "concretizing" or "reasoning".

        Returns:
            Prompt: the generated prompt.

        Examples:
            >>> from distilabel.tasks.text_generation import EvolComplexityGeneratorTask
            >>> task = EvolComplexityGeneratorTask()
            >>> task.generate_prompt("Give three tips for staying healthy.")
            Prompt(
                system_prompt="",
                formatted_prompt="I want you to act as a Prompt ...",
            )
        """
        evolution_method = self._get_evolution_method(evolution_method, EvolutionMethod)

        return super().generate_prompt(input, evolution_method=evolution_method, **_)
