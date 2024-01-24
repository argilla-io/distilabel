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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from distilabel.utils.argilla import (
    infer_fields_from_dataset_row,
    model_metadata_from_dataset_row,
)
from distilabel.utils.imports import _ARGILLA_AVAILABLE

if TYPE_CHECKING:
    from argilla import FeedbackDataset, FeedbackRecord


if _ARGILLA_AVAILABLE:
    import argilla as rg


class InstructTaskMixin:
    """Mixin that adds the `to_argilla_dataset` and `to_argilla_record` methods for tasks
    that generate/modify instructions `SelfInstructTask` or `EvolInstructTask`.
    """

    def to_argilla_dataset(self, dataset_row: Dict[str, Any]) -> "FeedbackDataset":
        # First we infer the fields from the input_args_names, but we could also
        # create those manually instead using `rg.TextField(...)`
        fields = infer_fields_from_dataset_row(
            field_names=self.input_args_names,
            dataset_row=dataset_row,
        )
        # Once the input fields have been defined, then we also include the instruction
        # field which will be fulfilled with each of the instructions generated.
        for arg_name in self.output_args_names:
            fields.append(rg.TextField(name=arg_name, title=arg_name))  # type: ignore
        # Then we add a default `RatingQuestion` which asks the users to provide a
        # rating for each of the generations, differing from the scenario where the inputs
        # are the fields and the outputs the ones used to formulate the quesstions. So on,
        # in this scenario we won't have suggestions, as the questions will be related to the
        # combination of inputs and outputs.
        questions = [
            rg.RatingQuestion(  # type: ignore
                name="instruction-rating",
                title="How would you rate the generated instruction?",
                values=list(range(1, 11)),
            )
        ]
        # Finally, we define some metadata properties that can be potentially used
        # while exploring the dataset within Argilla to get more insights on the data.
        metadata_properties = []
        for arg_name in self.input_args_names:
            if isinstance(dataset_row[arg_name], list):
                for idx in range(1, len(dataset_row[arg_name]) + 1):
                    metadata_properties.append(
                        rg.IntegerMetadataProperty(name=f"length-{arg_name}-{idx}")  # type: ignore
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
            rg.IntegerMetadataProperty(name="length-instruction")  # type: ignore
        )  # type: ignore
        # Then we just return the `FeedbackDataset` with the fields, questions, and metadata properties
        # defined above.
        return rg.FeedbackDataset(
            fields=fields,
            questions=questions,  # type: ignore
            metadata_properties=metadata_properties,  # Note that these are always optional
        )

    def to_argilla_record(
        self,
        dataset_row: Dict[str, Any],
        instructions_column: Optional[str] = None,
    ) -> List["FeedbackRecord"]:
        """Converts a dataset row to a list of Argilla `FeedbackRecord`s."""
        records = []
        if instructions_column is None:
            instructions_column = self.output_args_names[0]

        for instruction in dataset_row[instructions_column]:  # type: ignore
            fields, metadata = {}, {}
            for arg_name in self.input_args_names:
                arg_value = dataset_row[arg_name]
                if isinstance(arg_value, list):
                    for idx, value in enumerate(arg_value, start=1):
                        value = value.strip() if isinstance(value, str) else ""
                        fields[f"{arg_name}-{idx}"] = value
                        if value is not None:
                            metadata[f"length-{arg_name}-{idx}"] = len(value)
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
            fields[self.output_args_names[0]] = instruction
            metadata[f"length-{self.output_args_names[0]}"] = len(instruction)

            # Then we add the model metadata from the `generation_model` and `labelling_model`
            # columns of the dataset, if they exist.
            metadata.update(model_metadata_from_dataset_row(dataset_row=dataset_row))
            # Finally, we append the `FeedbackRecord` with the fields and the metadata
            records.append(rg.FeedbackRecord(fields=fields, metadata=metadata))
        if not records:
            raise ValueError(
                f"Skipping the row {dataset_row} as the list of `FeedbackRecord` is empty as those could not be inferred."
            )
        return records
