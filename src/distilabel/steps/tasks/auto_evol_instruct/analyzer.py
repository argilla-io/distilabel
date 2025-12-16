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

from typing import TYPE_CHECKING, Optional, Union

from jinja2 import Template
from pydantic import PrivateAttr
from typing_extensions import override

from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import StepColumns


SYSTEM_PROMPT: str = """\
You are an expert at analyzing the evolution of a given instruction. You will look at the trajectory of the evolution from an initial instruction and make feedbacks based on how the complexity is being increased in each stage.

Please strictly output using the following format, do not add anything else to the response:

***FORMAT INSTRUCTION***
Choose one of the two options:
Option 1 - If all cases are evolving correctly, please strictly output:
### PASSED

Option 2 - If you identify cases that did not evolve correctly, please strictly output:
### FAILED - Reason: [reason_of_fail]
and so on...
***END OF FORMAT INSTRUCTION***"""


USER_PROMPT: str = """\
The following list shows cases where an Instruction evolves into a more complex version of an Instruction.
For each case, stage 0 represents the Instruction in its initial state, and stage 1 requires an increase in complexity based on the previous stage.

Please identify cases that failed to evolve, and provide the reason why it fails.

Evolution Trajectory:
{{ evol_trajectory }}
"""


class AutoEvolTrajectoryAnalizer(Task):
    """_summary_

    Attributes:
        system_prompt: The system prompt to be used in the completions.
        user_prompt: ...

    Input columns:
        - instruction (`str`): The original instruction.
        - evolved_instruction (`str`): The evolved instruction from using AutoEvolver task.

    Output columns:
        - feedback (`str`): Feedback for the optimization.
        - model_name (`str`): The name of the model used to generate the feedback.

    Categories:
        - text-generation

    References:
        - [`Automatic Instruction Evolving for Large Language Models`](https://arxiv.org/abs/2406.00770)

    Examples:
        Annotate your steps with the Math Shepherd Completer:

        ```python
        from distilabel.steps.tasks import AutoEvolver
        from distilabel.models import InferenceEndpointsLLM

        model_id = "Qwen/Qwen2.5-72B-Instruct"

        llm = InferenceEndpointsLLM(
            model_id=model_id,
            tokenizer_id=model_id,
            generation_kwargs={
                "max_new_tokens": 2048, "temperature": 0.2,
            },
        )
        evolver = AutoEvolTrajectoryAnalizer(
            llm=llm  # evol_llm
        )
        evolver.load()

        result_analyzer = next(
            analyzer.process(
                [
                    {
                        "instruction": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
                        "evolved_instruction": "Natalia sold hair clips to 48 of her friends in April, and then she sold half as many clips, but no less than 20, in May. How many hair clips did Natalia sell altogether in April and May?"
                    }
                ]
            )
        )
        print(result[0]["feedback"])
        # '### PASSED'
        ```

    Citations:

        ```
        @misc{zeng2024automaticinstructionevolvinglarge,
            title={Automatic Instruction Evolving for Large Language Models},
            author={Weihao Zeng and Can Xu and Yingxiu Zhao and Jian-Guang Lou and Weizhu Chen},
            year={2024},
            eprint={2406.00770},
            archivePrefix={arXiv},
            primaryClass={cs.CL},
            url={https://arxiv.org/abs/2406.00770},
        }
        ```
    """

    system_prompt: Optional[str] = SYSTEM_PROMPT
    user_prompt: str = USER_PROMPT

    _template: Union[Template, None] = PrivateAttr(...)

    def load(self) -> None:
        super().load()
        self._template = Template(self.user_prompt)

    @property
    def inputs(self) -> "StepColumns":
        return ["instruction", "evolved_instruction"]

    @property
    def outputs(self) -> "StepColumns":
        return ["feedback", "model_name"]

    def format_input(self, input: dict[str, any]) -> "ChatType":
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""
        evol_trajectory = (
            f"Stage 0: {input['instruction']}\nStage 1: {input['evolved_instruction']}"
        )

        messages = [
            {
                "role": "user",
                "content": self._template.render(evol_trajectory=evol_trajectory),
            },
        ]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        return messages

    @override
    def format_output(
        self, output: Union[str, None], input: dict[str, any]
    ) -> dict[str, any]:
        return {"feedback": output, "model_name": self.llm.model_name}
