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
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, get_args

from distilabel.logger import get_logger
from distilabel.tasks.base import get_template
from distilabel.tasks.prompt import Prompt
from distilabel.tasks.text_generation.base import TextGenerationTask

logger = get_logger()


_EVOL_INSTRUCT_TEMPLATE = get_template("evol-instruct.jinja2")


EvolutionMethod = Literal[
    "breadth", "constraints", "deepen", "concretizing", "reasoning"
]


@dataclass
class EvolInstructTask(TextGenerationTask):
    """A `TextGenerationTask` following the `EvolInstruct` specification for building the prompts.

    From the reference repository: *Evol-Instruct is a novel method using LLMs instead of humans to automatically mass-produce
    open-domain instructions of various difficulty levels and skills range, to improve the performance of LLMs.*

    The task is defined as follows:
    Starting from an initial (simpler) instruction, select in-depth or in-breadth evolving to upgrade the simple instruction
    to a more complex one or create a new one (to increase diversity).
    The In-depth Evolving includes the following operations: add constraints, deepening, concretizing and increase reasoning.
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

    def _elimination_evolving(self, output: str) -> Optional[str]:
        """Performs the elimination step of the Evol-Instruct task, steps 2-4 in the paper:

        2. The evolved instruction makes it difficult for the LLM to generate a response. We found that
        when the generated response contains “sorry” and is relatively short in length (i.e., less than
        80 words), it often indicates that the LLM struggles to respond to the evolved instruction.
        So we can use this rule to make a judgment.
        3. The response generated by the LLM only contains punctuation and stop words.
        4. The evolved instruction obviously copies some words from the evolving prompt, such as
        “given prompt”, “rewritten prompt”, “#Rewritten Prompt#”, etc.
        """
        if output == "":
            return

        # 2) The evolved instruction makes it difficult for the LLM to generate a response.
        if "sorry" in output.lower() and len(output.split(" ")) < 80:
            return

        # 3) The output only contains punctuation and stop words
        if (
            len(set(output).intersection(string.punctuation)) / len(set(output))
        ) == 1.0:
            logger.info(
                f"Evolution step removed the output, it only contains punctuation and stop words: {output}"
            )
            return

        # 4) Remove copied words from the prompt
        prompt_words = {
            "#Given Prompt#",
            "#Created Prompt#",
            "given prompt",
            "created prompt",
            "#The Given Prompt#",
            "#Rewritten Prompt#",
            "rewritten prompt",
        }
        if any(word in output for word in prompt_words):
            logger.info(
                f"Evolution step removed the output due to word repetition from the prompt: {output}"
            )
            return

        return output

    def parse_output(self, output: str) -> Dict[str, List[str]]:
        """Parses the output of the model into the desired format, applying the elimination step for bad generations."""
        output = self._elimination_evolving(output)
        return {"instruction": output}
