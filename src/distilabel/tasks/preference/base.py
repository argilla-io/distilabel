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
from distilabel.tasks.base import Task

try:
    import argilla as rg
except ImportError:
    pass

if TYPE_CHECKING:
    from argilla.client.feedback.schemas.records import FeedbackRecord
    from argilla.client.feedback.schemas.types import (
        AllowedFieldTypes,
        AllowedQuestionTypes,
        AllowedMetadataPropertyTypes,
    )

@dataclass
class PreferenceTask(Task):
    def to_argilla_fields(
        self,
        dataset_row: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> List["AllowedFieldTypes"]:
        argilla_fields = []
        for arg_name in self.input_args_names:
            self._check_argument_exists(dataset_row, arg_name)
            if isinstance(dataset_row[arg_name], list):
                for idx in range(1, len(dataset_row[arg_name]) + 1):
                    argilla_fields.append(rg.TextField(name=f"{arg_name}-{idx}"))
            elif isinstance(dataset_row[arg_name], str):
                argilla_fields.append(rg.TextField(name=arg_name))
        return argilla_fields

    def to_argilla_questions(
        self,
        dataset_row: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> List["AllowedQuestionTypes"]:
        arg_name = "generations"
        self._check_argument_exists(dataset_row, arg_name)
        argilla_questions = []
        if isinstance(dataset_row[arg_name], list):
            # add ratings for generation
            for idx in range(1, len(dataset_row[arg_name]) + 1):
                argilla_questions.append(
                    rg.RatingQuestion(
                        name=f"{arg_name}-{idx}-rating",
                        title=f"What's the rating for {arg_name}-{idx}?",
                        values=list(range(1, 11)),
                    ),
                )
        argilla_questions.append(
            rg.TextQuestion(
                name=f"ratings-rationale",
                title=f"What's the rationale behind the ratings?",
            ),
        )
        return argilla_questions

    def to_argilla_metadata_properties(
        self,
        dataset_row: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> List["AllowedMetadataPropertyTypes"]:
        metadata_properties = []
        for arg_name in self.input_args_names:
            self._check_argument_exists(dataset_row, arg_name)
            if isinstance(dataset_row[arg_name], list):
                for idx in range(1, len(dataset_row[arg_name]) + 1):
                    metadata_properties.append(
                        rg.IntegerMetadataProperty(name=f"length-{arg_name}-{idx}")
                    )
                    metadata_properties.append(
                        rg.FloatMetadataProperty(name=f"rating-{arg_name}-{idx}")
                    )
            elif isinstance(dataset_row[arg_name], str):
                metadata_properties.append(
                    rg.IntegerMetadataProperty(name=f"length-{arg_name}")
                )
            else:
                raise ValueError(
                    f"Type {type(dataset_row[arg_name])} is not supported."
                )
        # additionally, we want the distance between best score and the second best
        if isinstance(dataset_row[arg_name], list):
            metadata_properties.append(
                rg.FloatMetadataProperty(name=f"distance-best-rated")
            )
        return metadata_properties

    def _check_argument_exists(self, dataset_row, arg_name):
        if arg_name not in dataset_row:
            raise ValueError(
                    f"Dataset row does not contain the required field '{arg_name}'."
                )

    def to_argilla_record(  # noqa: C901
        self,
        dataset_row: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> "FeedbackRecord":
        fields = {}
        metadata = {}
        for input_arg_name in self.input_args_names:
            if isinstance(dataset_row[input_arg_name], list):
                for idx in range(1, len(dataset_row[input_arg_name]) + 1):
                    input_value = dataset_row[input_arg_name][idx - 1].strip()
                    # update field
                    fields.update({f"{input_arg_name}-{idx}": input_value})
                    # update field-related metadata
                    metadata.update(
                        {f"length-{input_arg_name}-{idx}": len(input_value)}
                    )
            else:
                input_value = dataset_row[input_arg_name].strip()
                fields.update({input_arg_name: input_value})
                metadata.update({f"length-{input_arg_name}": len(input_value)})

        suggestions = []

        # add rationale
        suggestions.append(
            {
                "question_name": f"ratings-rationale",
                "value": self._to_argilla_rationale(dataset_row),
            }
        )
        for output_arg_name in self.output_args_names:
            if output_arg_name == "rating":
                ratings = []
                for idx, value in enumerate(dataset_row[output_arg_name], start=1):
                    ratings.append(value)
                    # add suggestions
                    suggestions.append(
                        {
                            "question_name": f"generations-{idx}-rating",
                            "value": int(value),
                        }
                    )
                    # update rating metadata
                    metadata.update({f"rating-{input_arg_name}-{idx}": value})
                if len(ratings) >= 2:
                    sorted_ratings = sorted(ratings, reverse=True)
                    # update rating distance from best to second
                    metadata.update(
                        {f"distance-best-rated": sorted_ratings[0] - sorted_ratings[1]}
                    )

        return rg.FeedbackRecord(
            fields=fields, suggestions=suggestions, metadata=metadata
        )

    def _to_argilla_rationale(self, dataset_row: Dict[str, Any]) -> str:
        return dataset_row["rationale"]