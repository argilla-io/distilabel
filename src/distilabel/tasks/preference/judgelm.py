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

from distilabel.tasks.base import Task, get_template
from distilabel.tasks.prompt import Prompt

try:
    import argilla as rg

    _argilla_installed = True
except ImportError:
    _argilla_installed = False

if TYPE_CHECKING:
    from argilla.client.feedback.schemas.records import FeedbackRecord
    from argilla.client.feedback.schemas.types import (
        AllowedFieldTypes,
        AllowedQuestionTypes,
    )

_JUDGELM_TEMPLATE = get_template("judgelm.jinja2")


class JudgeLMOutput(TypedDict):
    ratings: List[int]
    rationale: str


@dataclass
class JudgeLMTask(Task):
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

    @property
    def input_args_names(self) -> List[str]:
        return ["input", "generations"]

    @property
    def output_args_names(self) -> List[str]:
        return ["ratings", "rationale"]

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
        # `ratings` here could be parsed to float as in some scenarios we
        # find the model is producing scores as 8.5, but that will break
        # the `argilla` integration as it expects an integer for the `RatingQuestion`
        # so we can either do the parsing there or leave it as is.
        ratings = [int(float(rating)) for rating in split_output[0].split(" ")]
        rationale = "\n".join(split_output[1:])
        return JudgeLMOutput(ratings=ratings, rationale=rationale)

    def to_argilla_fields(
        self,
        dataset_row: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> List["AllowedFieldTypes"]:
        if not _argilla_installed:
            raise ImportError("The argilla library is not installed.")
        argilla_fields = []
        for arg_name in self.input_args_names:
            if arg_name not in dataset_row:
                raise ValueError(
                    f"Dataset row does not contain the required field '{arg_name}'."
                )
            if isinstance(dataset_row[arg_name], list):
                for idx in range(1, len(dataset_row[arg_name]) + 1):
                    argilla_fields.append(rg.TextField(name=f"{arg_name}-{idx}"))
            elif isinstance(dataset_row[arg_name], str):
                argilla_fields.append(rg.TextField(name=arg_name))
            else:
                raise ValueError(
                    f"Type {type(dataset_row[arg_name])} is not supported."
                )
        return argilla_fields

    def to_argilla_questions(
        self,
        dataset_row: Dict[str, Any],
        group_ratings_as_ranking: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> List["AllowedQuestionTypes"]:
        # TODO: move argilla_installed check to the `Argilla` abstract class
        if not _argilla_installed:
            raise ImportError("The argilla library is not installed.")

        arg_name = "generations"
        if arg_name not in dataset_row:
            raise ValueError(
                f"Dataset row does not contain the required field '{arg_name}'."
            )
        argilla_questions = []
        if isinstance(dataset_row[arg_name], list):
            # If `group_ratings_as_ranking` is True, then we group all the ratings into a ranking
            if group_ratings_as_ranking:
                argilla_questions.append(
                    rg.RankingQuestion(
                        name=f"{arg_name}-ranking",
                        title=f"Rank the {arg_name} from best to worst.",
                        values=[
                            f"{arg_name}-{idx}"
                            for idx in range(1, len(dataset_row[arg_name]) + 1)
                        ],
                    )
                )
            # Otherwise, we ask for each rating individually, but we still add the rationale
            for idx in range(1, len(dataset_row[arg_name]) + 1):
                if not group_ratings_as_ranking:
                    argilla_questions.append(
                        rg.RatingQuestion(
                            name=f"{arg_name}-{idx}-rating",
                            title=f"What's the rating for {arg_name}-{idx}?",
                            values=list(range(1, 11)),
                        ),
                    )
            argilla_questions.append(
                rg.TextQuestion(
                    name=f"{arg_name}-rationale",
                    title=f"What's the rationale behind the {arg_name} ratings?",
                ),
            )
        return argilla_questions

    def to_argilla_record(  # noqa: C901
        self,
        dataset_row: Dict[str, Any],
        group_ratings_as_ranking: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> "FeedbackRecord":
        if not _argilla_installed:
            raise ImportError("The argilla library is not installed.")
        fields = {}
        for input_arg_name in self.input_args_names:
            if isinstance(dataset_row[input_arg_name], list):
                for idx in range(1, len(dataset_row[input_arg_name]) + 1):
                    fields.update(
                        {
                            f"{input_arg_name}-{idx}": dataset_row[input_arg_name][
                                idx - 1
                            ].strip()
                        }
                    )
            else:
                fields.update({input_arg_name: dataset_row[input_arg_name]})
        suggestions = []
        for output_arg_name in self.output_args_names:
            if output_arg_name == "rationale":
                suggestions.append(
                    {
                        "question_name": f"generations-{output_arg_name}",
                        "value": dataset_row[output_arg_name],
                    }
                )
            elif output_arg_name == "ratings":
                if group_ratings_as_ranking:

                    def ratings_as_ranking_value(ratings: List[int]):
                        indexed_ratings = list(enumerate(ratings, start=1))
                        sorted_ratings = sorted(
                            indexed_ratings, key=lambda x: x[1], reverse=True
                        )

                        ranked_fields = []
                        current_rank = 1
                        for i, (index, rating) in enumerate(sorted_ratings):
                            if i > 0 and rating < sorted_ratings[i - 1][1]:
                                current_rank = i + 1
                            ranked_fields.append(
                                {"rank": current_rank, "value": f"generations-{index}"}
                            )

                        return ranked_fields

                    suggestions.append(
                        {
                            "question_name": "generations-ranking",
                            "value": ratings_as_ranking_value(
                                dataset_row[output_arg_name]
                            ),
                        }
                    )
                else:
                    for idx, value in enumerate(dataset_row[output_arg_name], start=1):
                        suggestions.append(
                            {
                                "question_name": f"generations-{idx}-rating",
                                "value": value,
                            }
                        )
        return rg.FeedbackRecord(fields=fields, suggestions=suggestions)
