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
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, TypedDict

from distilabel.tasks.base import Prompt, get_template
from distilabel.tasks.preference.base import PreferenceTask

_ULTRAJUDGE_TEMPLATE = get_template("ultrajudge.jinja2")


class Area(TypedDict):
    """A `TypedDict` representing an area of evaluation."""

    rating: float
    rationale: str


class UltraJudgeOutput(TypedDict):
    """A `TypedDict` representing the output of the UltraJudge task."""

    rating: float
    areas: Dict[str, Area]


@dataclass
class UltraJudgeTask(PreferenceTask):
    """A `PreferenceTask` for the UltraJudge task. The `UltraJudge` task has been defined
    at Argilla specically for a better evaluation using AI Feedback. The task is defined
    based on both UltraFeedback and JudgeLM, but with several improvements / modifications.

    Args:
        system_prompt (str, optional): the system prompt to be used for generation. Defaults to `None`.
        task_description (Union[str, None], optional): the description of the task. Defaults to `None`.
        areas (List[str], optional): the areas to be used for the task. Defaults to a list of four areas:
            "Practical Accuracy", "Clarity & Transparency", "Authenticity & Reliability", and "Compliance with Intent".
    """

    system_prompt: str = (
        "You are an evaluator tasked with assessing AI assistants' responses from the perspective of typical user preferences."
        " Your critical analysis should focus on human-like engagement, solution effectiveness, accuracy, clarity, and"
        " creativity. Approach each response as if you were the user, considering how well the response meets your needs"
        " and expectations in a real-world scenario. Provide detailed feedback that highlights strengths and areas for"
        " improvement in each response, keeping in mind the goal of simulating a human's preferred choice. "
        "Your evaluation should be impartial and thorough, reflecting a human's perspective in preferring responses that are practical,"
        " clear, authentic, and aligned with their intent. Avoid bias, and focus on the content and quality of the responses."
    )

    task_description: str = (
        "Your task is to rigorously evaluate the performance of {num_responses} AI assistants, simulating a human's perspective."
        " You will assess each response based on four key domains, reflecting aspects that are typically valued by humans:"
        " {areas}."
        " First provide a score between 0 and 10 and write a detailed feedback for each area and assistant."
        " Finally, provide a list of {num_responses} scores, each separated by a space, to reflect the performance of Assistants 1 to {num_responses}."
    )

    areas: List[str] = field(
        default_factory=lambda: [
            "Practical Accuracy",
            "Clarity & Transparency",
            "Authenticity & Reliability",
            "Compliance with Intent",
        ]
    )

    __jinja2_template__: ClassVar[str] = field(
        default=_ULTRAJUDGE_TEMPLATE, init=False, repr=False
    )

    @property
    def output_args_names(self) -> List[str]:
        """Returns the names of the output arguments of the task."""
        return ["rating", "areas"]

    @property
    def areas_str(self) -> str:
        """Returns a string representation of the areas."""
        return ", ".join(self.areas[:-1]) + ", and " + self.areas[-1]

    @property
    def extract_area_score_and_rationale_regex(self) -> str:
        """Returns a regex to extract the area, score, and rationale from the output."""
        return rf"({'|'.join(self.areas)})\s*-\s*(\d+(?:\.\d+)?)\n(.*?)(?=\n\n|\Z)"

    @property
    def extract_final_scores_regex(self) -> str:
        """Returns a regex to extract the final scores from the output."""
        return r"Final scores:\s*((?:\d+(?:\.\d+)?\s*)+)"

    def generate_prompt(self, input: str, generations: List[str]) -> Prompt:
        """Generates a prompt following the UltraJudge specification.

        Args:
            input (str): the input to be used for the prompt.
            generations (List[str]): the generations to be used for the prompt.

        Returns:
            Prompt: the generated prompt.

        Examples:
            >>> from distilabel.tasks.preference import UltraJudgeTask
            >>> task = UltraJudgeTask(system_prompt="You are a helpful assistant.")
            >>> task.generate_prompt("What are the first 5 Fibonacci numbers?", ["0 1 1 2 3", "0 1 1 2 3"])
            Prompt(
                system_prompt="You are a helpful assistant.",
                formatted_prompt="Your task is to rigorously evaluate the performance of ...",
            )
        """
        render_kwargs = {
            "task_description": self.task_description.format(
                num_responses=len(generations), areas=self.areas_str
            ),
            "instruction": input,
            "responses": generations,
        }

        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        )

    def parse_output(self, output: str) -> List[UltraJudgeOutput]:
        """Parses the output of the model into the desired format."""
        num_areas = len(self.areas)
        # `areas_results` includes num_generations * num_areas tuples
        areas_results = re.findall(self.extract_area_score_and_rationale_regex, output)
        final_scores = [
            float(str_score)
            for str_score in re.findall(self.extract_final_scores_regex, output)[
                0
            ].split(" ")
        ]

        outputs = []
        for i, rating in enumerate(final_scores):
            areas = {}
            # Get the areas for the i-th generation
            for area in areas_results[i * num_areas : i * num_areas + num_areas]:
                name, area_rating, rationale = area
                areas[name] = Area(rating=area_rating, rationale=rationale)
            outputs.append(UltraJudgeOutput(rating=rating, areas=areas))

        return outputs

    def _to_argilla_rationale(
        self,
        dataset_row: Dict[str, Any],
    ) -> str:
        """Gets the `rationale` column from a `datasets.Dataset` row and formats it
        as expected by Argilla.
        """

        def format_area(area):
            sections = []
            for title, ratings in area.items():
                sections.append(title)
                for k, v in ratings.items():
                    sections.append(f"{k}:{v}")
            return "\n".join(sections)

        rationales = []
        for idx, area in enumerate(dataset_row["areas"], start=1):
            formatted_area = format_area(area)
            rationales.append(f"Rationale for generation-{idx}:\n{formatted_area}\n")
        return "\n".join(rationales)
