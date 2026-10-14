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

from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import StepColumns


SYSTEM_PROMPT: str = """You are an Instruction Method Optimizer.

**Output Instructions**
Add more steps to achieve the most refined method if needed, however, REMEMBER that the final step in your output has to be "#Finally Rewritten Instruction#" no matter how many steps are added.
Please generate the optimized method strictly using ONLY the given below format, do not add anything else."""


USER_PROMPT: str = """\
Feedback: {{ feedback }}
Based on the feedback from the evolution failure case, optimize the method below to create a more effective instruction rewriting process without negatively impacting performance on other cases. Ensure that the complexity of the optimized method is not lower than the previous method.
If the feedback is "### PASSED", then come up with a better method than the current one to create a more complex and effective instruction rewriting process. Remember that the new method should not be very similar to the current method, be creative with new steps for the new method.

Current Method:
{{ current_method }}

```Optimized Method
Step 1:
#Methods List#
Describe how to generate a list of methods to make instructions more complex, incorporating the feedback

Step 2:
#Plan#
Explain how to create a comprehensive plan based on the Methods List

[Note]Add more steps here as you want to achieve the best method. The steps should align with the instruction domain/topic, and should not involve any tools or visualization, it should be text-only methods. The last step should always be #Finally Rewritten Instruction#.

Step N-1:
#Rewritten Instruction#
Do not generate new Instruction here, but please provide a detailed the process of executing the plan to rewrite the instruction. You are generating a guide to write a better instruction, NOT THE INSTRUCTION ITSELF.

Step N:
#Finally Rewritten Instruction#
Do not generate new Instruction here, but please provide the process to write the final rewritten instruction. You are generating a guide to write a better instruction, NOT THE INSTRUCTION ITSELF.
```"""


class AutoEvolOptimizer(Task):
    system_prompt: Optional[str] = SYSTEM_PROMPT
    user_prompt: Optional[str] = USER_PROMPT

    _template: Union[Template, None] = PrivateAttr(...)

    def load(self) -> None:
        super().load()
        self._template = Template(self.user_prompt)

    @property
    def inputs(self) -> "StepColumns":
        return ["optimization_method", "feedback"]

    @property
    def outputs(self) -> "StepColumns":
        return ["optimized_method", "model_name"]

    def format_input(self, input: dict[str, any]) -> "ChatType":
        messages = [
            {
                "role": "user",
                "content": self._template.render(
                    feedback=input["feedback"],
                    current_method=input["optimization_method"],
                ),
            },
        ]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        return messages

    def format_output(
        self, output: Union[str, None], input: dict[str, any]
    ) -> dict[str, any]:
        if output is None:
            input.update(
                **{"optimized_method": None, "model_name": self.llm.model_name}
            )
            return input

        # steps = parse_steps(output)
        input.update(**{"optimized_method": output, "model_name": self.llm.model_name})
        return input
