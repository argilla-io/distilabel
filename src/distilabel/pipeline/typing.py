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

from typing import TYPE_CHECKING, TypeVar, Union

if TYPE_CHECKING:
    from distilabel.pipeline.routing_batch_function import RoutingBatchFunction
    from distilabel.steps.base import GeneratorStep, GlobalStep, Step

DownstreamConnectable = Union["Step", "GlobalStep", "RoutingBatchFunction"]

UpstreamConnectableSteps = TypeVar(
    "UpstreamConnectableSteps",
    bound=Union["Step", "GlobalStep", "GeneratorStep"],
)

DownstreamConnectableSteps = TypeVar(
    "DownstreamConnectableSteps",
    bound=Union["Step", "GlobalStep"],
    covariant=True,
)
