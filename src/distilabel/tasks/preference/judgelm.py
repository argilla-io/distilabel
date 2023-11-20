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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

from typing_extensions import TypedDict

from distilabel.tasks.base import get_template
from distilabel.tasks.preference.base import PreferenceTask
from distilabel.tasks.prompt import Prompt

_JUDGELM_TEMPLATE = get_template("judgelm.jinja2")


class JudgeLMOutput(TypedDict):
    rating: List[int]
    rationale: str


@dataclass
class JudgeLMTask(PreferenceTask):
    __jinja2_template__: str = _JUDGELM_TEMPLATE

    task_description: str = (
        "We would like to request your feedback on the performance of {num_responses} AI assistants in response to the"
        " user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details"
        " of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher"
        " score indicates better overall performance.\nPlease first output a single line containing only {num_responses}"
        " values indicating the scores for Assistants 1 to {num_responses}, respectively. The {num_responses} scores are separated by"
        " a space. In the subsequent line, please provide a comprehensive explanation of your evaluation,"
        " avoiding any potential bias and ensuring that the order in which the responses were presented does"
        " not affect your judgment."
    )
    system_prompt: str = "You are a helpful and precise assistant for checking the quality of the answer."

    def generate_prompt(self, input: str, generations: List[str]) -> Prompt:
        render_kwargs = {
            "input": input,
            "responses": generations,
            "task_description": self.task_description.format(
                num_responses=len(generations)
            ),
        }
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        )

    def parse_output(self, output: str) -> JudgeLMOutput:
        split_output = output.split("\n")
        # `rating` here could be parsed to float as in some scenarios we
        # find the model is producing scores as 8.5, but that will break
        # the `argilla` integration as it expects an integer for the `RatingQuestion`
        # so we can either do the parsing there or leave it as is.
        rating = [float(rating) for rating in split_output[0].split(" ")]
        rationale = "\n".join(split_output[1:])
        return JudgeLMOutput(rating=rating, rationale=rationale)