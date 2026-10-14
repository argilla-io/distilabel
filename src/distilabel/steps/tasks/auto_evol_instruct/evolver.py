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

import json
from typing import TYPE_CHECKING, Optional, Union

from jinja2 import Template
from pydantic import PrivateAttr

from distilabel.steps.tasks.auto_evol_instruct.utils import parse_steps
from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import StepColumns


USER_PROMPT: str = """\
Please follow the steps below to rewrite the given "#Instruction#" into a more complex version.

Step 1: Please read the "#Instruction#" below carefully and list all the possible methods to make this instruction more complex (to make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle). Please do not provide methods to change the language of the instruction!

Step 2: Please create a comprehensive plan based on the #Methods List# generated in Step 1 to make the #Instruction# more complex. The plan should include several methods from the #Methods List#.

Step 3: Please execute the plan step by step and provide the #Rewritten Instruction#. #Rewritten Instruction# can only add 10 to 20 words into the "#Instruction#".

Step 4: Please carefully review the #Rewritten Instruction# and identify any unreasonable parts. Ensure that the #Rewritten Instruction# is only a more complex version of the #Instruction#, make sure that it only adds 10 to 20 words into the "#Instruction#". Just provide the #Finally Rewritten Instruction# without any explanation.

**Output Instructions**
Please generate the optimized instruction strictly using ONLY the given below format, do not add anything else:

```Optimized Instruction
Step 1:
#Methods List#

Step 2:
#Plan#

Step 3:
#Rewritten Instruction#

Step 4:
#Finally Rewritten Instruction#
```

REMEMBER that you are generating a more complex version of the instruction (or question), NOT answering #Instruction#. The #Finally Rewritten Instruction# should only add 10 to 20 words the #Instruction# below.

#Instruction#: {{ instruction }}
"""

NEXT_OPTIMIZATION_METHOD: str = """\
Please follow the steps below to rewrite the given "#Instruction#" into a more complex version.

{{step_details}}

**Output Instructions**
Please generate the optimized instruction strictly using ONLY the given below format, do not add anything else:

```Optimized Instruction
{{format_steps}}
```
"""


ADD_INSTRUCTION: str = """REMEMBER that you are generating a more complex version of the instruction (or question), NOT answering #Instruction#. The #Finally Rewritten Instruction# should only add 10 to 20 words the #Instruction# below.

#Instruction#: {{ instruction }}
"""


class AutoEvolver(Task):
    """_summary_

    Attributes:
        system_prompt: The system prompt to be used in the completions.
        user_prompt: ...

    Input columns:
        - instruction (`str`): The original instruction.

    Output columns:
        - evolved_instruction (`str`): The evolved instruction.
        - model_name (`str`): The name of the model used to generate the revision.

    Categories:
        - text-generation

    References:
        - [`Automatic Instruction Evolving for Large Language Models`](https://arxiv.org/abs/2406.00770)

    Examples:
        Evolve instructions:

        ```python
        from distilabel.steps.tasks import AutoEvolver
        from distilabel.models import InferenceEndpointsLLM

        model_id = "Qwen/Qwen2.5-72B-Instruct"

        llm = InferenceEndpointsLLM(
            model_id=model_id,
            tokenizer_id=model_id,
            generation_kwargs={
                "max_new_tokens": 2048, "temperature": 0.5,
            },
        )
        evolver = AutoEvolver(
            input_batch_size=4,
            llm=llm  # evol_llm
        )
        evolver.load()

        result = next(evolver.process([{"instruction": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"}]))
        print(result[0]["evolved_instruction"])
        # 'Natalia sold hair clips to 48 of her friends in April, and then she sold half as many clips, but no less than 20, in May. How many hair clips did Natalia sell altogether in April and May?'
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

    system_prompt: Optional[str] = (
        "You are an Instruction Rewriter that rewrites the given #Instruction# into a more complex version."
    )
    user_prompt: str = USER_PROMPT

    _template: Union[Template, None] = PrivateAttr(...)

    def load(self) -> None:
        super().load()
        self._template = Template(self.user_prompt)

    @property
    def inputs(self) -> "StepColumns":
        return ["instruction"]

    @property
    def outputs(self) -> "StepColumns":
        return ["evolved_instruction", "evolution_metadata", "model_name"]

    def format_input(self, input: dict[str, any]) -> "ChatType":
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""

        messages = [
            {
                "role": "user",
                "content": self._template.render(instruction=input["instruction"]),
            },
        ]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        return messages

    def format_output(
        self, output: Union[str, None], input: dict[str, any]
    ) -> dict[str, any]:
        """The output is formatted as a list with the score of each instruction-response pair.

        Args:
            output: the raw output of the LLM.
            input: the input to the task. Used for obtaining the number of responses.

        Returns:
            A dict with the key `scores` containing the scores for each instruction-response pair.
        """
        if output is None:
            input.update(
                **{"evolved_instruction": None, "model_name": self.llm.model_name}
            )
            return input

        steps = parse_steps(output)
        evolution_metadata = {
            step["step_name"]: step["step_instruction"] for step in steps[:-1]
        }
        input.update(
            **{
                "evolved_instruction": steps[-1]["step_instruction"],
                "evolution_metadata": json.dumps(evolution_metadata),
                "model_name": self.llm.model_name,
            }
        )
        return input

    @property
    def optimization_method(self) -> str:
        messages = self.format_input({"instruction": ""})
        if len(messages) == 1:
            return messages[0]["content"]
        return messages[1]["content"]

    @staticmethod
    def build_new_optimization_method(optimized_instruction: str) -> str:
        """Builds a new optimized prompt.
        SHOULD BE REUSED HERE PASSING IT A THE NEW user_prompt

        # TODO: MAKES SENSE TO MAKE A CLASSMETHOD AND RETURN A NEW AutoEvolver TASK
        INSTANCE??
        """
        # optimized_instruction is the output from EvolOptimizer.process
        steps = parse_steps(optimized_instruction)

        step_details = ""
        format_steps = ""

        for i, step in enumerate(steps, start=1):
            step_name = step["step_name"]
            step_instruction = step["step_instruction"]

            step_details += f"Step {i}: {step_instruction}\n\n"
            format_steps += f"Step {i}:\n#{step_name}#\n\n"

        optimization_method = NEXT_OPTIMIZATION_METHOD.replace(
            "{{step_details}}", step_details
        )
        optimization_method = NEXT_OPTIMIZATION_METHOD.replace(
            "{{format_steps}}", format_steps
        )
        optimization_method += "\n" + ADD_INSTRUCTION
        return optimization_method
