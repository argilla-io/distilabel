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

from typing import TYPE_CHECKING, Dict, List, Literal, TypedDict, TypeVar, Union

if TYPE_CHECKING:
    import pandas as pd
    from datasets import Dataset

    from distilabel.mixins.runtime_parameters import RuntimeParameterInfo
    from distilabel.steps.base import GeneratorStep, GlobalStep, Step

DownstreamConnectable = Union["Step", "GlobalStep"]
"""Alias for the `Step` types that can be connected as downstream steps."""

UpstreamConnectableSteps = TypeVar(
    "UpstreamConnectableSteps",
    bound=Union["Step", "GlobalStep", "GeneratorStep"],
)
"""Type for the `Step` types that can be connected as upstream steps."""

DownstreamConnectableSteps = TypeVar(
    "DownstreamConnectableSteps",
    bound=DownstreamConnectable,
    covariant=True,
)
"""Type for the `Step` types that can be connected as downstream steps."""


class StepLoadStatus(TypedDict):
    """Dict containing information about if one step was loaded/unloaded or if it's load
    failed"""

    name: str
    status: Literal["loaded", "unloaded", "load_failed"]


PipelineRuntimeParametersInfo = Dict[
    str, Union[List["RuntimeParameterInfo"], Dict[str, "RuntimeParameterInfo"]]
]
"""Alias for the information of the runtime parameters of a `Pipeline`."""

InputDataset = Union["Dataset", "pd.DataFrame", List[Dict[str, str]]]
"""Alias for the types we can process as input dataset."""
