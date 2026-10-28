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

import re
import sys

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

from typing import Any, Dict, List, Optional, Union

from jinja2 import Template
from pydantic import PrivateAttr

from distilabel.steps.tasks.base import Task
from distilabel.typing import ChatType


class InstructionBacktranslation(Task):
    """Self-Alignment with Instruction Backtranslation.

    Attributes:
        system_prompt: The system prompt for the instruction backtranslation task.
        _template: the Jinja2 template to use for the Instruction Backtranslation task.

    Input columns:
        - instruction (`str`): The reference instruction to evaluate the text output.
        - generation (`str`): The text output to evaluate for the given instruction.
        - system_prompt (`Optional[str]`): The system prompt for the instruction backtranslation task.

    Output columns:
        - score (`str`): The score for the generation based on the given instruction.
        - reason (`str`): The reason for the provided score.
        - model_name (`str`): The model name used to score the generation.

    Categories:
        - critique

    References:
        - [`Self-Alignment with Instruction Backtranslation`](https://arxiv.org/abs/2308.06259)

    Examples:
        Generate a score and reason for a given instruction and generation:

        ```python
        from distilabel.steps.tasks import InstructionBacktranslation

        instruction_backtranslation = InstructionBacktranslation(
                name="instruction_backtranslation",
                llm=llm,
                input_batch_size=10,
                output_mappings={"model_name": "scoring_model"},
            )
        instruction_backtranslation.load()

        result = next(
            instruction_backtranslation.process(
                [
                    {
                        "instruction": "How much is 2+2?",
                        "generation": "4",
                    }
                ]
            )
        )
        # result
        # [
        #     {
        #         "instruction": "How much is 2+2?",
        #         "generation": "4",
        #         "score": 3,
        #         "reason": "Reason for the generation.",
        #         "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        #     }
        # ]
        ```

    Citations:
        ```
        @misc{li2024selfalignmentinstructionbacktranslation,
            title={Self-Alignment with Instruction Backtranslation},
            author={Xian Li and Ping Yu and Chunting Zhou and Timo Schick and Omer Levy and Luke Zettlemoyer and Jason Weston and Mike Lewis},
            year={2024},
            eprint={2308.06259},
            archivePrefix={arXiv},
            primaryClass={cs.CL},
            url={https://arxiv.org/abs/2308.06259},
        }
        ```
    """

    _template: Optional["Template"] = PrivateAttr(default=...)
    _can_be_used_with_offline_batch_generation = True

    def load(self) -> None:
        """Loads the Jinja2 template."""
        super().load()

        _path = str(
            importlib_resources.files("distilabel")
            / "steps"
            / "tasks"
            / "templates"
            / "instruction-backtranslation.jinja2"
        )

        self._template = Template(open(_path).read())

    @property
    def inputs(self) -> List[str]:
        """The input for the task is the `instruction`, and the `generation` for it."""
        return {"instruction": True, "generation": True, "system_prompt": False}

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""
        messages = []
        if "system_prompt" in input:
            messages.append({"role": "system", "content": input["system_prompt"]})
        elif self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": self._template.render(  # type: ignore
                    instruction=input["instruction"], generation=input["generation"]
                ),
            },
        )
        return messages

    @property
    def outputs(self) -> List[str]:
        """The output for the task is the `score`, `reason` and the `model_name`."""
        return ["score", "reason", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """The output is formatted as a dictionary with the `score` and `reason`. The
        `model_name` will be automatically included within the `process` method of `Task`.

        Args:
            output: a string representing the output of the LLM via the `process` method.
            input: the input to the task, as required by some tasks to format the output.

        Returns:
            A dictionary containing the `score` and the `reason` for the provided `score`.
        """
        pattern = r"(.+?)Score: (\d)"

        matches = None
        if output is not None:
            matches = re.findall(pattern, output, re.DOTALL)
        if matches is None:
            return {"score": None, "reason": None}

        return {
            "score": int(matches[0][1]),
            "reason": matches[0][0].strip(),
        }
