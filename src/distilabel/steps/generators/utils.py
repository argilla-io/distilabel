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

from typing import TYPE_CHECKING, Dict, List, Optional, Union

import pandas as pd
from datasets import Dataset

from distilabel.errors import DistilabelUserError
from distilabel.steps.base import StepResources

if TYPE_CHECKING:
    from distilabel.pipeline.base import BasePipeline
    from distilabel.steps import GeneratorStep


def make_generator_step(
    dataset: Union[Dataset, pd.DataFrame, List[Dict[str, str]]],
    pipeline: Union["BasePipeline", None] = None,
    batch_size: int = 50,
    input_mappings: Optional[Dict[str, str]] = None,
    output_mappings: Optional[Dict[str, str]] = None,
    resources: StepResources = StepResources(),
    repo_id: Optional[str] = None,
) -> "GeneratorStep":
    """Helper method to create a `GeneratorStep` from a dataset, to simplify

    Args:
        dataset: The dataset to use in the `Pipeline`.
        batch_size: The batch_size, will default to the same used by the `GeneratorStep`s.
            Defaults to `50`.
        input_mappings: Applies the same as any other step. Defaults to `None`.
        output_mappings: Applies the same as any other step. Defaults to `None`.
        resources: Applies the same as any other step. Defaults to `StepResources()`.

    Raises:
        ValueError: If the format is different from the ones supported.

    Returns:
        A `LoadDataFromDicts` if the input is a list of dicts, or `LoadDataFromHub` instance
        if the input is a `pd.DataFrame` or a `Dataset`.
    """
    from distilabel.steps import LoadDataFromDicts, LoadDataFromHub

    if isinstance(dataset, list):
        return LoadDataFromDicts(
            pipeline=pipeline,
            data=dataset,
            batch_size=batch_size,
            input_mappings=input_mappings or {},
            output_mappings=output_mappings or {},
            resources=resources,
        )

    if isinstance(dataset, pd.DataFrame):
        dataset = Dataset.from_pandas(dataset, preserve_index=False)

    if not isinstance(dataset, Dataset):
        raise DistilabelUserError(
            f"Dataset type not allowed: {type(dataset)}, must be one of: "
            "`datasets.Dataset`, `pd.DataFrame`, `List[Dict[str, str]]`",
            page="sections/how_to_guides/basic/pipeline/?h=make_#__tabbed_1_2",
        )

    loader = LoadDataFromHub(
        pipeline=pipeline,
        repo_id=repo_id,
        batch_size=batch_size,
        input_mappings=input_mappings or {},
        output_mappings=output_mappings or {},
        resources=resources,
    )
    super(loader.__class__, loader).load()
    loader._dataset = dataset
    loader.num_examples = len(dataset)
    loader._dataset_info = {"default": dataset.info}
    return loader
