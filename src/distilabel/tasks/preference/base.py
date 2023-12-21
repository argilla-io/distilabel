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

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from distilabel.tasks.base import Task
from distilabel.utils.argilla import (
    infer_fields_from_dataset_row,
    model_metadata_from_dataset_row,
)
from distilabel.utils.imports import _ARGILLA_AVAILABLE

if _ARGILLA_AVAILABLE:
    import argilla as rg

if TYPE_CHECKING:
    from argilla import FeedbackDataset
    from argilla.client.feedback.schemas.records import FeedbackRecord


@dataclass
class PreferenceTask(Task):
    """A `Task` for preference rating tasks.

    Args:
        system_prompt (str): the system prompt to be used for generation.
        task_description (Union[str, None], optional): the description of the task. Defaults to `None`.
    """

    @property
    def input_args_names(self) -> List[str]:
        """Returns the names of the input arguments of the task."""
        return ["input", "generations"]

    @property
    def output_args_names(self) -> List[str]:
        """Returns the names of the output arguments of the task."""
        return ["rating", "rationale"]

    def to_argilla_dataset(
        self,
        dataset_row: Dict[str, Any],
        generations_column: Optional[str] = "generations",
        ratings_column: Optional[str] = "rating",
        ratings_values: Optional[List[int]] = None,
        rationale_column: Optional[str] = "rationale",
    ) -> "FeedbackDataset":
        # First we infer the fields from the input_args_names, but we could also
        # create those manually instead using `rg.TextField(...)`
        fields = infer_fields_from_dataset_row(
            field_names=self.input_args_names, dataset_row=dataset_row
        )
        # Then we add the questions, which cannot be easily inferred in this case,
        # because those depend neither on the outputs nor on the inputs, but in a combination
        # of both, since the questions will be formulated using the inputs, but assigned to the
        # outputs.
        if generations_column is None or generations_column not in dataset_row:
            raise ValueError(
                f"The `generations_column='{generations_column}'` is not present in the dataset"
                f" row. Please provide any of {list(dataset_row.keys())}.",
            )
        if ratings_column is None or ratings_column not in dataset_row:
            raise ValueError(
                f"The `ratings_column='{ratings_column}'` is not present in the dataset row. Please"
                f" provide any of {list(dataset_row.keys())}.",
            )
        if rationale_column is None or rationale_column not in dataset_row:
            raise ValueError(
                f"The `rationale_column='{rationale_column}'` is not present in the dataset row. Please"
                f" provide any of {list(dataset_row.keys())}.",
            )
        questions = []
        for idx in range(1, len(dataset_row[generations_column]) + 1):
            questions.append(
                rg.RatingQuestion(  # type: ignore
                    name=f"{generations_column}-{idx}-{ratings_column}",
                    title=f"What's the {ratings_column} for {generations_column}-{idx}?",
                    values=ratings_values or [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                )
            )
        questions.append(
            rg.TextQuestion(  # type: ignore
                name=f"{ratings_column}-{rationale_column}",
                title=f"What's the {rationale_column} behind each {ratings_column}?",
            )
        )
        # Finally, we define some metadata properties that can be potentially used
        # while exploring the dataset within Argilla to get more insights on the data.
        metadata_properties = []
        for arg_name in self.input_args_names:
            if isinstance(dataset_row[arg_name], list):
                for idx in range(1, len(dataset_row[arg_name]) + 1):
                    metadata_properties.append(
                        rg.IntegerMetadataProperty(name=f"length-{arg_name}-{idx}")  # type: ignore
                    )
                    if arg_name == generations_column:
                        metadata_properties.append(
                            rg.FloatMetadataProperty(
                                name=f"{ratings_column}-{arg_name}-{idx}"
                            )  # type: ignore
                        )
            elif isinstance(dataset_row[arg_name], str):
                metadata_properties.append(
                    rg.IntegerMetadataProperty(name=f"length-{arg_name}")  # type: ignore
                )
            else:
                warnings.warn(
                    f"Unsupported input type ({type(dataset_row[arg_name])}), skipping...",
                    UserWarning,
                    stacklevel=2,
                )
        metadata_properties.append(
            rg.FloatMetadataProperty(name=f"distance-best-{ratings_column}")  # type: ignore
        )
        # Then we just return the `FeedbackDataset` with the fields, questions, and metadata properties
        # defined above.
        return rg.FeedbackDataset(
            fields=fields,
            questions=questions,
            metadata_properties=metadata_properties,  # Note that these are always optional
        )

    def _merge_rationales(
        self, rationales: List[str], generations_column: str = "generations"
    ) -> str:
        return "\n".join(
            f"{generations_column}-{idx}:\n{rationale}\n"
            for idx, rationale in enumerate(rationales, start=1)
        )

    def to_argilla_record(  # noqa: C901
        self,
        dataset_row: Dict[str, Any],
        generations_column: Optional[str] = "generations",
        ratings_column: Optional[str] = "rating",
        rationale_column: Optional[str] = "rationale",
    ) -> "FeedbackRecord":
        """Converts a dataset row to an Argilla `FeedbackRecord`."""
        # We start off with the fields, which are the inputs of the LLM, but also
        # build the metadata from them, as previously specified within the
        fields, metadata = {}, {}
        for arg_name in self.input_args_names:
            arg_value = dataset_row[arg_name]
            if isinstance(arg_value, list):
                for idx, value in enumerate(arg_value, start=1):
                    fields[f"{arg_name}-{idx}"] = value.strip() if value else ""
                    if value is not None:
                        metadata[f"length-{arg_name}-{idx}"] = len(value.strip())
            elif isinstance(arg_value, str):
                fields[arg_name] = arg_value.strip() if arg_value else ""
                if arg_value is not None:
                    metadata[f"length-{arg_name}"] = len(arg_value.strip())
            else:
                warnings.warn(
                    f"Unsupported input type ({type(arg_value)}), skipping...",
                    UserWarning,
                    stacklevel=2,
                )
        # Then we include the suggestions, which are generated from the outputs
        # of the LLM instead.
        suggestions = []
        if rationale_column is None or rationale_column not in dataset_row:
            raise ValueError(
                f"The rationale column {rationale_column} is not present in the dataset row."
            )
        if dataset_row.get(rationale_column) is not None:
            rationales = dataset_row.get(rationale_column)
            suggestions.append(
                {
                    "question_name": f"{ratings_column}-{rationale_column}",
                    "value": self._merge_rationales(rationales=rationales)
                    if isinstance(rationales, list)
                    else rationales,
                }
            )
        if ratings_column is None or ratings_column not in dataset_row:
            raise ValueError(
                f"The ratings column {ratings_column} is not present in the dataset row."
            )
        if dataset_row.get(ratings_column) is not None:
            ratings = dataset_row.get(ratings_column)
            for idx, value in enumerate(ratings, start=1):  # type: ignore
                suggestions.append(
                    {
                        "question_name": f"{generations_column}-{idx}-{ratings_column}",
                        "value": 1 if value < 1 else int(value) if value < 10 else None,
                    }
                )
                metadata[f"{ratings_column}-{generations_column}-{idx}"] = value
            if len(ratings) >= 2:  # type: ignore
                sorted_ratings = sorted(ratings, reverse=True)  # type: ignore
                metadata[f"distance-best-{ratings_column}"] = (
                    sorted_ratings[0] - sorted_ratings[1]
                )
        # Then we add the model metadata from the `generation_model` and `labelling_model`
        # columns of the dataset, if they exist.
        metadata.update(model_metadata_from_dataset_row(dataset_row=dataset_row))
        # Finally, we return the `FeedbackRecord` with the fields and the metadata
        return rg.FeedbackRecord(
            fields=fields, suggestions=suggestions, metadata=metadata
        )
