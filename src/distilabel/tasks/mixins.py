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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

from distilabel.utils.argilla import (
    infer_fields_from_dataset_row,
    model_metadata_from_dataset_row,
)
from distilabel.utils.imports import _ARGILLA_AVAILABLE

if _ARGILLA_AVAILABLE:
    import argilla as rg

if TYPE_CHECKING:
    from argilla import FeedbackDataset, FeedbackRecord


class TaskProtocol(Protocol):
    @property
    def input_args_names(self) -> List[str]:
        ...


class RatingToArgillaMixin:
    """Mixin that adds the `to_argilla_dataset` and `to_argilla_record` methods for tasks
    that generate both ratings and rationales i.e. `PreferenceTask` or `CritiqueTask`.
    """

    def _check_column_is_present(
        self,
        column_name: str,
        dataset_row: Dict[str, Any],
        column_type: str = "generations",
    ) -> None:
        """Helper function to check if a column is present in the dataset row.

        Args:
            column_name (str): Name of the column to check.
            dataset_row (Dict[str, Any]): Row from the dataset.
            column_type (str, optional): Type of column expected in the dataset. Defaults to "generations".

        Raises:
            ValueError: If the column is not present in the dataset row when it should be.
        """
        # The function is added mainly to simplify the code in the `to_argilla_dataset` to pass the mccabe complexity check.
        if column_name is None or column_name not in dataset_row:
            raise ValueError(
                f"The `{column_type}_column='{column_name}'` is not present in the"
                f" dataset row. Please provide any of {list(dataset_row.keys())}.",
            )

    def to_argilla_dataset(
        self: TaskProtocol,
        dataset_row: Dict[str, Any],
        generations_column: str = "generations",
        ratings_column: str = "rating",
        rationale_column: str = "rationale",
        ratings_values: Optional[List[int]] = None,
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
        self._check_column_is_present(
            generations_column, dataset_row, column_type="generations"
        )
        self._check_column_is_present(
            ratings_column, dataset_row, column_type="ratings"
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
        if rationale_column:
            self._check_column_is_present(
                rationale_column, dataset_row, column_type="rationale"
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
        self: TaskProtocol,
        dataset_row: Dict[str, Any],
        generations_column: str = "generations",
        ratings_column: str = "rating",
        rationale_column: str = "rationale",
        ratings_values: Optional[List[int]] = None,
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
        if dataset_row.get(rationale_column) is not None:
            rationales = dataset_row.get(rationale_column)
            suggestions.append(
                {
                    "question_name": f"{ratings_column}-{rationale_column}",
                    "value": self._merge_rationales(rationales=rationales)  # type: ignore
                    if isinstance(rationales, list)
                    else rationales,
                }
            )
        if ratings_column and ratings_column not in dataset_row:
            raise ValueError(
                f"The ratings column {ratings_column} is not present in the dataset row."
            )
        if dataset_row.get(ratings_column) is not None:
            ratings = dataset_row.get(ratings_column)
            if isinstance(ratings, list):
                for idx, value in enumerate(ratings, start=1):  # type: ignore
                    suggestions.append(
                        {
                            "question_name": f"{generations_column}-{idx}-{ratings_column}",
                            "value": 1
                            if value < 1
                            else int(value)
                            if value
                            <= (
                                max(ratings_values)
                                if ratings_values is not None
                                else 10
                            )
                            else None,
                        }
                    )
                    metadata[f"{ratings_column}-{generations_column}-{idx}"] = value
                if len(ratings) >= 2:  # type: ignore
                    sorted_ratings = sorted(ratings, reverse=True)  # type: ignore
                    metadata[f"distance-best-{ratings_column}"] = (
                        sorted_ratings[0] - sorted_ratings[1]
                    )
            elif isinstance(ratings, (str, float, int)):
                suggestions.append(
                    {
                        "question_name": f"{generations_column}-1-{ratings_column}",
                        "value": int(ratings),
                    }
                )
        # Then we add the model metadata from the `generation_model` and `labelling_model`
        # columns of the dataset, if they exist.
        metadata.update(model_metadata_from_dataset_row(dataset_row=dataset_row))
        # Finally, we return the `FeedbackRecord` with the fields and the metadata
        return rg.FeedbackRecord(
            fields=fields, suggestions=suggestions, metadata=metadata
        )
