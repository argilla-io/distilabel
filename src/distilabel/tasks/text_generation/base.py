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

import random
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from distilabel.tasks.base import Task
from distilabel.tasks.prompt import Prompt
from distilabel.tasks.text_generation.principles import UltraFeedbackPrinciples
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
class TextGenerationTask(Task):
    """A base `Task` definition for text generation using LLMs.

    Args:
        system_prompt (str, optional): the system prompt to be used. Defaults to `None`.
        principles (Dict[str, List[str]], optional): the principles to be used for the system prompt.
            Defaults to `None`.
        principles_distribution (Union[Dict[str, float], Literal["balanced"], None], optional): the
            distribution of principles to be used for the system prompt. Defaults to `None`.

    Examples:
        >>> from distilabel.tasks.text_generation import TextGenerationTask
        >>> task = TextGenerationTask()
    """

    system_prompt: str = (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible,"
        " while being safe. Your answers should not include any harmful, unethical, racist, sexist,"
        " toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased"
        " and positive in nature.\nIf a question does not make any sense, or is not factually coherent,"
        " explain why instead of answering something not correct. If you don't know the answer to a"
        " question, please don't share false information."
    )
    principles: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "harmlessness": UltraFeedbackPrinciples.harmlessness,
            "helpfulness": UltraFeedbackPrinciples.helpfulness,
            "truthfulness": UltraFeedbackPrinciples.truthfulness,
            "honesty": UltraFeedbackPrinciples.honesty,
            "verbalized_calibration": UltraFeedbackPrinciples.verbalized_calibration,
        },
        repr=False,
    )
    principles_distribution: Union[Dict[str, float], Literal["balanced"], None] = None

    __type__: Literal["generation"] = "generation"

    def __post_init__(self) -> None:
        """Validates the `principles_distribution` if it is a dict.

        Raises:
            ValueError: if the `principles_distribution` is a dict and it does not sum to 1.0.
            ValueError: if the `principles` are not included in the `principles_distribution`.
        """
        if isinstance(self.principles_distribution, dict):
            not_included_principles = [
                principle
                for principle in self.principles
                if principle not in self.principles_distribution
            ]
            if not_included_principles:
                principles_str = ", ".join(
                    [f"'{principle}'" for principle in not_included_principles]
                )
                raise ValueError(
                    f"Principles {principles_str} included in `principles` is not in"
                    " `principles_distribution`"
                )

            if sum(self.principles_distribution.values()) != 1.0:
                raise ValueError(
                    "`principles_distribution` must sum to 1.0 if it is a dict containing"
                    " the distribution of principles to use."
                )

    def _get_principle(self) -> str:
        """Gets a principle from the `principles` dict respecting the `principal_distribution`.

        Returns:
            str: the principle to be used.
        """
        if isinstance(self.principles_distribution, dict):
            principle_group = random.choices(
                list(self.principles_distribution.keys()),
                weights=list(self.principles_distribution.values()),
                k=1,
            )[0]
        else:
            principle_group = random.choice(list(self.principles.keys()))
        return random.choice(self.principles[principle_group])

    def generate_prompt(self, input: str) -> Prompt:
        """Generates the prompt to be used for generation.

        Args:
            input (str): the input to be used for generation.

        Returns:
            Prompt: the generated prompt.

        Examples:
            >>> from distilabel.tasks.text_generation import TextGenerationTask
            >>> task = TextGenerationTask(system_prompt="You are a helpful assistant.")
            >>> task.generate_prompt("What are the first 5 Fibonacci numbers?")
            Prompt(system_prompt='You are a helpful assistant.', formatted_prompt='What are the first 5 Fibonacci numbers?')
        """
        system_prompt = self.system_prompt
        if self.principles_distribution is not None:
            principle = self._get_principle()
            system_prompt += " " + principle
        return Prompt(system_prompt=system_prompt, formatted_prompt=input)

    def parse_output(self, output: str) -> Dict[str, str]:
        """Parses the output of the LLM into the desired format."""
        return {"generations": output}

    @property
    def input_args_names(self) -> List[str]:
        """Returns the input args names for the task."""
        return ["input"]

    @property
    def output_args_names(self) -> List[str]:
        """Returns the output args names for the task."""
        return ["generations"]

    def to_argilla_dataset(
        self,
        dataset_row: Dict[str, Any],
        generations_column: Optional[str] = "generations",
    ) -> "FeedbackDataset":
        # First we infer the fields from the input_args_names, but we could also
        # create those manually instead using `rg.TextField(...)`
        fields = infer_fields_from_dataset_row(
            field_names=self.input_args_names + self.output_args_names,
            dataset_row=dataset_row,
        )
        # Then we add a default `RatingQuestion` which asks the users to provide a
        # rating for each of the generations, differing from the scenario where the inputs
        # are the fields and the outputs the ones used to formulate the quesstions. So on,
        # in this scenario we won't have suggestions, as the questions will be related to the
        # combination of inputs and outputs.
        if generations_column is None or generations_column not in dataset_row:
            raise ValueError(
                f"The `generations_column='{generations_column}'` is not present in the dataset"
                f" row. Please provide any of {list(dataset_row.keys())}.",
            )
        questions = []
        for idx in range(1, len(dataset_row[generations_column]) + 1):
            questions.append(
                rg.RatingQuestion(  # type: ignore
                    name=f"{generations_column}-{idx}-rating",
                    title=f"How would you rate the generation at `{generations_column}-{idx}`?",
                    values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                )
            )
        # Finally, we define some metadata properties that can be potentially used
        # while exploring the dataset within Argilla to get more insights on the data.
        metadata_properties = []
        for arg_name in self.input_args_names + self.output_args_names:
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
        # Then we just return the `FeedbackDataset` with the fields, questions, and metadata properties
        # defined above.
        return rg.FeedbackDataset(
            fields=fields,
            questions=questions,
            metadata_properties=metadata_properties,  # Note that these are always optional
        )

    def to_argilla_record(self, dataset_row: Dict[str, Any]) -> "FeedbackRecord":
        """Converts a dataset row to an Argilla `FeedbackRecord`."""
        # We start off with the fields, which are the inputs of the LLM, but also
        # build the metadata from them, as previously specified within the
        fields, metadata = {}, {}
        for arg_name in self.input_args_names + self.output_args_names:
            arg_value = dataset_row[arg_name]
            if isinstance(arg_value, list):
                for idx, value in enumerate(arg_value, start=1):
                    # TODO: value formatting was included here due to some issues
                    # with `SelfInstructTask` but these list-parsing may not be needed
                    # anymore.
                    value = (
                        value.strip()
                        if isinstance(value, str)
                        else "\n".join(value)
                        if isinstance(value, list)
                        else ""
                    )
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
        # Then we add the model metadata from the `generation_model` and `labelling_model`
        # columns of the dataset, if they exist.
        metadata.update(model_metadata_from_dataset_row(dataset_row=dataset_row))
        # Finally, we return the `FeedbackRecord` with the fields and the metadata
        return rg.FeedbackRecord(fields=fields, metadata=metadata)
