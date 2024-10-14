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

from typing import Any, Dict, List, Union

import orjson
from jinja2 import Template
from pydantic import PrivateAttr
from typing_extensions import override

from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.typing import ChatType

_PARSE_SCORE_LINE_REGEX = re.compile(r"\[\d+\] score: (\d+)", re.IGNORECASE)


class QualityScorer(Task):
    """Score responses based on their quality using an `LLM`.

    `QualityScorer` is a pre-defined task that defines the `instruction` as the input
    and `score` as the output. This task is used to rate the quality of instructions and responses.
    It's an implementation of the quality score task from the paper 'What Makes Good Data
    for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning'.
    The task follows the same scheme as the Complexity Scorer, but the instruction-response pairs
    are scored in terms of quality, obtaining a quality score for each instruction.

    Attributes:
        _template: a Jinja2 template used to format the input for the LLM.

    Input columns:
        - instruction (`str`): The instruction that was used to generate the `responses`.
        - responses (`List[str]`): The responses to be scored. Each response forms a pair with the instruction.

    Output columns:
        - scores (`List[float]`): The score for each instruction.
        - model_name (`str`): The model name used to generate the scores.

    Categories:
        - scorer
        - quality
        - response

    References:
        - [`What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning`](https://arxiv.org/abs/2312.15685)

    Examples:
        Evaluate the quality of your instructions:

        ```python
        from distilabel.steps.tasks import QualityScorer
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        scorer = QualityScorer(
            llm=InferenceEndpointsLLM(
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
            )
        )

        scorer.load()

        result = next(
            scorer.process(
                [
                    {
                        "instruction": "instruction",
                        "responses": ["good response", "weird response", "bad response"]
                    }
                ]
            )
        )
        # result
        [
            {
                'instructions': 'instruction',
                'model_name': 'test',
                'scores': [5, 3, 1],
            }
        ]
        ```

        Generate structured output with default schema:

        ```python
        from distilabel.steps.tasks import QualityScorer
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        scorer = QualityScorer(
            llm=InferenceEndpointsLLM(
                model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            ),
            use_default_structured_output=True
        )

        scorer.load()

        result = next(
            scorer.process(
                [
                    {
                        "instruction": "instruction",
                        "responses": ["good response", "weird response", "bad response"]
                    }
                ]
            )
        )

        # result
        [{'instruction': 'instruction',
        'responses': ['good response', 'weird response', 'bad response'],
        'scores': [1, 2, 3],
        'distilabel_metadata': {'raw_output_quality_scorer_0': '{  "scores": [1, 2, 3] }'},
        'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct'}]
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
    _can_be_used_with_offline_batch_generation = True

    def load(self) -> None:
        """Loads the Jinja2 template."""
        super().load()

        _path = str(
            importlib_resources.files("distilabel")
            / "steps"
            / "tasks"
            / "templates"
            / "quality-scorer.jinja2"
        )

        self._template = Template(open(_path).read())

    @property
    def inputs(self) -> List[str]:
        """The inputs for the task are `instruction` and `responses`."""
        return ["instruction", "responses"]

    def format_input(self, input: Dict[str, Any]) -> ChatType:  # type: ignore
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""
        return [
            {
                "role": "user",
                "content": self._template.render(  # type: ignore
                    instruction=input["instruction"], responses=input["responses"]
                ),
            }
        ]

    @property
    def outputs(self):
        """The output for the task is a list of `scores` containing the quality score for each
        response in `responses`."""
        return ["scores", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """The output is formatted as a list with the score of each instruction-response pair.

        Args:
            output: the raw output of the LLM.
            input: the input to the task. Used for obtaining the number of responses.

        Returns:
            A dict with the key `scores` containing the scores for each instruction-response pair.
        """
        if output is None:
            return {"scores": [None] * len(input["responses"])}

        if self.use_default_structured_output:
            return self._format_structured_output(output, input)

        scores = []
        score_lines = output.split("\n")

        for i, line in enumerate(score_lines):
            match = _PARSE_SCORE_LINE_REGEX.match(line)
            score = float(match.group(1)) if match else None
            scores.append(score)
            if i == len(input["responses"]) - 1:
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

        class SchemaQualityScorer(BaseModel):
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
            "title": "SchemaQualityScorer",
            "type": "object",
        }

    def _format_structured_output(
        self, output: str, input: Dict[str, Any]
    ) -> Dict[str, str]:
        """Parses the structured response, which should correspond to a dictionary
        with the scores, and a list with them.

        Args:
            output: The output from the `LLM`.

        Returns:
            Formatted output.
        """
        try:
            return orjson.loads(output)
        except orjson.JSONDecodeError:
            return {"scores": [None] * len(input["responses"])}

    @override
    def _sample_input(self) -> ChatType:
        return self.format_input(
            {
                "instruction": f"<PLACEHOLDER_{'instruction'.upper()}>",
                "responses": [
                    f"<PLACEHOLDER_{f'RESPONSE_{i}'.upper()}>" for i in range(2)
                ],
            }
        )
