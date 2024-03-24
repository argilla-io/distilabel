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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

from jinja2 import Template
from pydantic import PrivateAttr

from distilabel.llm.base import LLM
from distilabel.llm.vllm import vLLM
from distilabel.steps.task.base import Task

if TYPE_CHECKING:
    from distilabel.steps.task.typing import ChatType


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
        " the user's questions."
    )
    _template: Optional["Template"] = PrivateAttr(default=...)

    def load(self) -> None:
        super().load()
        _path = str(
            importlib_resources.files("distilabel")
            / "steps"
            / "task"
            / "templates"
            / "ultracm.jinja2"
        )
        with open(_path, "r") as f:
            self._template = Template(f.read())

    @property
    def inputs(self) -> List[str]:
        """The input for the task are `instruction` and `completion`."""
        return ["instruction", "completion"]

    @property
    def outputs(self) -> List[str]:
        """The output for the task are `score` and `critique`."""
        return ["score", "critique", "raw_output", "model_name"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """The input is formatted as a `ChatType` with a default system prompt as defined in the
        model."""
        return [
            {"role": "system", "content": self.system_prompt + "</s>"},
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
        if output is None:
            return {output: None for output in self.outputs}

        result = {"score": None, "critique": None, "raw_output": None}
        output = output.strip("\n").strip()
        if "Overall Score:" in output:
            critique, score = output.split("Overall Score:")
            critique = critique.strip("\n").strip()
            score = float(score.strip("\n").strip())
            result["score"] = score
            result["critique"] = critique
        else:
            result["raw_output"] = output

        return result
