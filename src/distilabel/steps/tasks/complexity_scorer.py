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

from typing import TYPE_CHECKING, Any, Dict, List, Union

import orjson
from jinja2 import Template
from pydantic import PrivateAttr
from typing_extensions import override

from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType


_PARSE_SCORE_LINE_REGEX = re.compile(r"\[\d+\] score: (\d+)", re.IGNORECASE)


class ComplexityScorer(Task):
    """Score instructions based on their complexity using an `LLM`.

    `ComplexityScorer` is a pre-defined task used to rank a list of instructions based in
    their complexity. It's an implementation of the complexity score task from the paper
    'What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection
    in Instruction Tuning'.

    Attributes:
        _template: a Jinja2 template used to format the input for the LLM.

    Input columns:
        - instructions (`List[str]`): The list of instructions to be scored.

    Output columns:
        - scores (`List[float]`): The score for each instruction.
        - model_name (`str`): The model name used to generate the scores.

    Categories:
        - scorer
        - complexity
        - instruction

    References:
        - [`What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning`](https://arxiv.org/abs/2312.15685)

    Examples:
        Evaluate the complexity of your instructions:

        ```python
        from distilabel.steps.tasks import ComplexityScorer
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        scorer = ComplexityScorer(
            llm=InferenceEndpointsLLM(
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
            )
        )

        scorer.load()

        result = next(
            scorer.process(
                [{"instructions": ["plain instruction", "highly complex instruction"]}]
            )
        )
        # result
        # [{'instructions': ['plain instruction', 'highly complex instruction'], 'model_name': 'test', 'scores': [1, 5], 'distilabel_metadata': {'raw_output_complexity_scorer_0': 'output'}}]
        ```

        Generate structured output with default schema:

        ```python
        from distilabel.steps.tasks import ComplexityScorer
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        scorer = ComplexityScorer(
            llm=InferenceEndpointsLLM(
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
            ),
            use_default_structured_output=use_default_structured_output
        )

        scorer.load()

        result = next(
            scorer.process(
                [{"instructions": ["plain instruction", "highly complex instruction"]}]
            )
        )
        # result
        # [{'instructions': ['plain instruction', 'highly complex instruction'], 'model_name': 'test', 'scores': [1, 2], 'distilabel_metadata': {'raw_output_complexity_scorer_0': '{ \\n  "scores": [\\n    1, \\n    2\\n  ]\\n}'}}]
        ```

    Citations:
        ```
        @misc{liu2024makesgooddataalignment,
            title={What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning},
            author={Wei Liu and Weihao Zeng and Keqing He and Yong Jiang and Junxian He},
            year={2024},
            eprint={2312.15685},
            archivePrefix={arXiv},
            primaryClass={cs.CL},
            url={https://arxiv.org/abs/2312.15685},
        }
        ```
    """

    _template: Union[Template, None] = PrivateAttr(...)

    def load(self) -> None:
        """Loads the Jinja2 template."""
        super().load()

        _path = str(
            importlib_resources.files("distilabel")
            / "steps"
            / "tasks"
            / "templates"
            / "complexity-scorer.jinja2"
        )

        self._template = Template(open(_path).read())

    @property
    def inputs(self) -> List[str]:
        """The inputs for the task are the `instructions`."""
        return ["instructions"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""
        return [
            {
                "role": "user",
                "content": self._template.render(instructions=input["instructions"]),  # type: ignore
            }
        ]

    @property
    def outputs(self) -> List[str]:
        """The output for the task are: a list of `scores` containing the complexity score for each
        instruction in `instructions`, and the `model_name`."""
        return ["scores", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """The output is formatted as a list with the score of each instruction.

        Args:
            output: the raw output of the LLM.
            input: the input to the task. Used for obtaining the number of responses.

        Returns:
            A dict with the key `scores` containing the scores for each instruction.
        """
        if output is None:
            return {"scores": [None] * len(input["instructions"])}

        if self.use_default_structured_output:
            return self._format_structured_output(output, input)

        scores = []
        score_lines = output.split("\n")
        for i, line in enumerate(score_lines):
            match = _PARSE_SCORE_LINE_REGEX.match(line)
            score = float(match.group(1)) if match else None
            scores.append(score)
            if i == len(input["instructions"]) - 1:
                break
        return {"scores": scores}

    @override
    def get_structured_output(self) -> Dict[str, Any]:
        """Creates the json schema to be passed to the LLM, to enforce generating
        a dictionary with the output which can be directly parsed as a python dictionary.

        The schema corresponds to the following:

        ```python
        from pydantic import BaseModel
        from typing import List

        class SchemaComplexityScorer(BaseModel):
            scores: List[int]
        ```

        Returns:
            JSON Schema of the response to enforce.
        """
        return {
            "properties": {
                "scores": {
                    "items": {"type": "integer"},
                    "title": "Scores",
                    "type": "array",
                }
            },
            "required": ["scores"],
            "title": "SchemaComplexityScorer",
            "type": "object",
        }

    def _format_structured_output(
        self, output: str, input: Dict[str, Any]
    ) -> Dict[str, str]:
        """Parses the structured response, which should correspond to a dictionary
        with either `positive`, or `positive` and `negative` keys.

        Args:
            output: The output from the `LLM`.

        Returns:
            Formatted output.
        """
        try:
            return orjson.loads(output)
        except orjson.JSONDecodeError:
            return {"scores": [None] * len(input["instructions"])}
