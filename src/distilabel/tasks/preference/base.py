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
from distilabel.tasks.prompt import Prompt

if TYPE_CHECKING:
    from argilla.client.feedback.schemas.records import FeedbackRecord
    from argilla.client.feedback.schemas.types import (
        AllowedFieldTypes,
        AllowedQuestionTypes,
        AllowedMetadataPropertyTypes,
    )


@dataclass
class PreferenceTask(Task):
    @property
    def input_args_names(self) -> List[str]:
        return ["input", "generations"]

    @property
    def output_args_names(self) -> List[str]:
        return ["rating", "rationale"]
    
    def to_argilla_fields(
        self, dataset_row: Dict[str, Any], *args: Any, **kwargs: Any
    ) -> List["AllowedFieldTypes"]:
        return self._process_dataset_row(dataset_row, self._create_text_field)

    def to_argilla_questions(
        self, dataset_row: Dict[str, Any], *args: Any, **kwargs: Any
    ) -> List["AllowedQuestionTypes"]:
        questions = []
        arg_name = "generations"
        self._check_argument_exists(dataset_row, arg_name)
        if isinstance(dataset_row[arg_name], list):
            for idx in range(1, len(dataset_row[arg_name]) + 1):
                question_name = f"{arg_name}-{idx}-rating"
                title = f"What's the rating for {arg_name}-{idx}?"
                questions.append(
                    self._create_rating_question(
                        question_name, title, list(range(1, 11))
                    )
                )
        questions.append(
            self._create_text_question(
                "ratings-rationale", "What's the rationale behind the ratings?"
            )
        )
        return questions

    def to_argilla_metadata_properties(
        self, dataset_row: Dict[str, Any], *args: Any, **kwargs: Any
    ) -> List["AllowedMetadataPropertyTypes"]:
        metadata_properties = []
        for arg_name in self.input_args_names:
            self._check_argument_exists(dataset_row, arg_name)
            if isinstance(dataset_row[arg_name], list):
                for idx in range(1, len(dataset_row[arg_name]) + 1):
                    metadata_properties.append(
                        self._create_metadata_property(
                            f"length-{arg_name}-{idx}", "integer"
                        )
                    )
                    metadata_properties.append(
                        self._create_metadata_property(
                            f"rating-{arg_name}-{idx}", "float"
                        )
                    )
            elif isinstance(dataset_row[arg_name], str):
                metadata_properties.append(
                    self._create_metadata_property(f"length-{arg_name}", "integer")
                )
            else:
                raise ValueError(
                    f"Type {type(dataset_row[arg_name])} is not supported."
                )
        # add distance between best rating and the second best
        if isinstance(dataset_row[arg_name], list):
            metadata_properties.append(
                self._create_metadata_property("distance-best-rated", "float")
            )
        return metadata_properties

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
                for idx, value in enumerate(dataset_row[input_arg_name], start=1):
                    self._update_fields_and_metadata(
                        fields, metadata, input_arg_name, value, idx
                    )
            else:
                self._update_fields_and_metadata(
                    fields, metadata, input_arg_name, dataset_row[input_arg_name]
                )

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
                output_data = dataset_row.get(output_arg_name)
                if output_data is not None:
                    for idx, value in enumerate(output_data, start=1):
                            ratings.append(value)
                            # add suggestions
                            suggestions.append(
                                {
                                    "question_name": f"generations-{idx}-rating",
                                    "value": int(value),
                                }
                            )
                            # update rating metadata
                            metadata.update({f"rating-generations-{idx}": value})
                if len(ratings) >= 2:
                    sorted_ratings = sorted(ratings, reverse=True)
                    # update rating distance from best to second
                    metadata.update(
                        {f"distance-best-rated": sorted_ratings[0] - sorted_ratings[1]}
                    )
        return self._create_argilla_record(
            fields=fields, suggestions=suggestions, metadata=metadata
        )

    def _to_argilla_rationale(self, dataset_row: Dict[str, Any]) -> str:
        return dataset_row["rationale"]