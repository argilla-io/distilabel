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
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List

from distilabel.tasks.base import get_template
from distilabel.tasks.critique.base import CritiqueTask, CritiqueTaskOutput
from distilabel.tasks.prompt import Prompt

_PROMETHEUS_TEMPLATE = get_template("prometheus.jinja2")


@dataclass
class PrometheusTask(CritiqueTask):
    """A `CritiqueTask` following the prompt templated used by Prometheus.

    Args:
        system_prompt (str, optional): the system prompt to be used for generation. Defaults to `None`.
        scoring_criteria (str): the scoring criteria to be used for the task, that defines
            the scores below, provided via `score_descriptions`.
        score_descriptions (Dict[int, str]): the descriptions of the scores, where
            the key is the rating value (ideally those should be consecutive), and the
            value is the description of each rating.

    Disclaimer:
        Since the Prometheus model has been trained with OpenAI API generated data, the prompting
        strategy may just be consistent / compliant with either GPT-3.5 or GPT-4 from OpenAI API, or
        with their own model. Any other model may fail on the generation of a structured output, as
        well as providing an incorrect / inaccurate critique.

    References:
        - [`Prometheus: Inducing Fine-grained Evaluation Capability in Language Models`](https://arxiv.org/abs/2310.08491)
        - [`kaist-ai/prometheus-13b-v1.0`](https://huggingface.co/kaist-ai/prometheus-7b-v1.0)
        - [`kaist-ai/prometheus-13b-v1.0`](https://huggingface.co/kaist-ai/prometheus-13b-v1.0)
    """

    scoring_criteria: str
    score_descriptions: Dict[int, str]

    system_prompt: str = "You are a fair evaluator language model."

    __jinja2_template__: ClassVar[str] = _PROMETHEUS_TEMPLATE

    @property
    def input_args_names(self) -> List[str]:
        return super().input_args_names + ["ref_completion"]

    def generate_prompt(
        self, input: str, generations: str, ref_completion: str, **_: Any
    ) -> Prompt:
        """Generates a prompt following the Prometheus specification.

        Args:
            input (str): the input to be used for the prompt.
            generations (List[str]): the generations to be used for the prompt, in
                this case, the ones to be critiqued.
            ref_completion (str): the reference completion to be used for the prompt,
                which is the reference one, assuming the one with the highest score.

        Returns:
            Prompt: the generated prompt.

        Examples:
            >>> from distilabel.tasks.critique import PrometheusTask
            >>> task = PrometheusTask(
            ...     scoring_criteria="Overall quality of the responses provided.",
            ...     score_descriptions={0: "false", 1: "partially false", 2: "average", 3: "partially true", 4: "true"},
            ... )
            >>> task.generate_prompt(
            ...     input="What are the first 5 Fibonacci numbers?",
            ...     generations=["0 1 1 2 3", "0 1 1 2 3"],
            ...     ref_completion="0 1 1 2 3",
            ... )
            Prompt(
                system_prompt="You are a fair evaluator language model.",
                formatted_prompt=""###Task Description:...",
            )
        """
        render_kwargs = {
            "instruction": input,
            "completion": generations,
            "ref_completion": ref_completion,
            "scoring_criteria": self.scoring_criteria,
            "score_descriptions": self.score_descriptions,
        }
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        )

    def parse_output(self, output: str) -> CritiqueTaskOutput:  # type: ignore
        """Parses the output of the model into the desired format."""
        # We use a regex instead of splitting by the delimiter because the
        # critique may contain the delimiter, and using the regex is safer.
        pattern = r"(.+?)\. \[RESULT\] (\d+)"
        match = re.search(pattern, output)
        if match:
            return CritiqueTaskOutput(
                score=float(match.group(2)),
                critique=match.group(1).strip(),
            )
