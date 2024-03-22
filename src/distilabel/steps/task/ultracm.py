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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from jinja2 import Template
from pydantic import PrivateAttr

from distilabel.llm.base import LLM
from distilabel.llm.vllm import vLLM
from distilabel.steps.task.base import Task

if TYPE_CHECKING:
    from distilabel.steps.task.typing import ChatType


_ULTRACM_TEMPLATE = """Given my answer to an instruction, your role is to provide specific and constructive feedback for me. You should find the best way for me to learn from your feedback and improve my performance.

You should consider multiple aspects of my answer, including helpfulness, truthfulness, honesty, and to what extent the answer follows instructions.
---

### Instruction
{{ instruction }}

### Answer
{{ completion }}
---

Please act as a teacher and provide specific and constructive feedback. Besides describing the weaknesses of the answer, you should also provide specific suggestions to guide me toward understanding how to improve. Please note, however, that your suggestions should help me better complete the instructions, but you should not introduce new requirements that are not mentioned in the instructions. Your feedback should focus on enhancing my ability to think critically and respond accurately. However, never explicitly provide the reference answer, nor do polite phrases be required. Only respond with concise feedback in chat style. Finally, score the overall quality of the answer from 1 to 10, where 1 is the worst and 10 is the best.

*Format*
### Feedback
[Your feedback]
Overall Score: [1-10]
—--

### Feedback
"""


class UltraCM(Task):
    """A critique specialized `Task` following the prompt templated used by UltraCM (from UltraFeedback).

    By default will be initialized with a specific `LLM` model from Hugging Face Hub using `vLLM`, which
    corresponds to the `openbmb/UltraCM-13b` model, the model that was trained to be used with this task.
    Take into account that independent of the engine used for the model, it's prepared to work with this specific one.

    Input columns:
        - `instruction` (`str`): The instruction that was used to generate the `completion`.
        - `completion` (`str`): The instruction that was used to generate the `completion`.

    Output columns:
        - `score` (`float`): The overall score of the answer from 1 to 10.
        - `critique` (`str`): The feedback given from the model to improve the answer.
        - `raw_output` (`str`): The raw output from the model, in case it couldn't be parsed.

    Notes:
        Since the UltraCM model has been trained with OpenAI API generated data, the prompting
        strategy may just be consistent / compliant with either GPT-3.5 or GPT-4 from OpenAI API, or
        with their own model. Any other model may fail on the generation of a structured output, as
        well as providing an incorrect / inaccurate critique.

    References:
        - [`UltraFeedback: Boosting Language Models with High-quality Feedback`](https://arxiv.org/abs/2310.01377)
        - [`UltraFeedback - GitHub Repository`](https://github.com/OpenBMB/UltraFeedback)
        - [`openbmb/UltraCM-13b`](https://huggingface.co/openbmb/UltraCM-13b)
    """

    llm: LLM = vLLM(model="openbmb/UltraCM-13b")
    system_prompt: str = (
        "User: A one-turn chat between a curious user and an artificial intelligence"
        " assistant. The assistant gives helpful, very detailed, and polite answers to"
        " the user's questions.</s>"
    )
    _template: Union[Template, None] = PrivateAttr(...)

    def load(self) -> None:
        super().load()
        self._template = Template(_ULTRACM_TEMPLATE)

    @property
    def inputs(self) -> List[str]:
        """The input for the task are `instruction` and `completion`."""
        return ["instruction", "completion"]

    @property
    def outputs(self) -> List[str]:
        """The output for the task are `score` and `critique`."""
        return ["score", "critique", "raw_output"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """The input is formatted as a `ChatType` with a default system prompt as defined in the
        model."""
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"User: {self._template.render(**input)}</s>\nAssistant: ",
            },
        ]  # type: ignore

    def format_output(
        self, output: Union[str, None], _: Dict[str, Any] = None
    ) -> Dict[str, Optional[Union[float, str]]]:
        """The output is formatted as a list with the score as a float and the critique for the response.

        If the output is `None` or the result couldn't be parsed, the result will be a dictionary with the score and critique as `None`.

        Args:
            output: the raw output of the LLM.

        Returns:
            A dict with containing the the score and critique for the response.
        """
        result = {output: None for output in self.outputs}
        if output is None:
            return result

        output = output.strip("\n").strip()
        if "Overall Score:" in output:
            critique, score = output.split("Overall Score:")
            critique = critique.strip("\n").strip()
            score = float(score.strip("\n").strip())
            result[self.outputs[0]] = score
            result[self.outputs[1]] = critique
        else:
            result[self.outputs[2]] = output

        return result
