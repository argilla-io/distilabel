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

from typing import List

from distilabel.pipeline.base import _Batch
from distilabel.pipeline.local import Pipeline
from distilabel.pipeline.routing_batch_function import (
    RoutingBatchFunction,
    routing_batch_function,
)

from tests.unit.pipeline.utils import DummyGeneratorStep, DummyStep1, DummyStep2


class TestRoutingBatchFunction:
    def test_route_batch(self) -> None:
        routing_batch_function = RoutingBatchFunction(
            routing_function=lambda steps: steps[:2]
        )

        routed_to = routing_batch_function.route_batch(
            batch=_Batch(seq_no=0, step_name="step_0", data=[[]], last_batch=False),
            steps=["step_1", "step_2", "step_3"],
        )

        assert routed_to == ["step_1", "step_2"]
        assert routing_batch_function._routed_batch_registry == {
            "step_0": {0: ["step_1", "step_2"]}
        }

        routed_to = routing_batch_function.route_batch(
            batch=_Batch(seq_no=1, step_name="step_0", data=[[]], last_batch=False),
            steps=["step_1", "step_2", "step_3"],
        )

        assert routed_to == ["step_1", "step_2"]
        assert routing_batch_function._routed_batch_registry == {
            "step_0": {0: ["step_1", "step_2"], 1: ["step_1", "step_2"]}
        }

    def test_call(self) -> None:
        routing_batch_function = RoutingBatchFunction(
            routing_function=lambda steps: steps[:2]
        )

        routed_to = routing_batch_function(
            batch=_Batch(seq_no=0, step_name="step_0", data=[[]], last_batch=False),
            steps=["step_1", "step_2", "step_3"],
        )

        assert routed_to == ["step_1", "step_2"]
        assert routing_batch_function._routed_batch_registry == {
            "step_0": {0: ["step_1", "step_2"]}
        }

    def test_binary_rshift_operator(self) -> None:
        routing_batch_function = RoutingBatchFunction(
            routing_function=lambda steps: steps[:2]
        )

        with Pipeline(name="test") as pipeline:
            upstream_step = DummyGeneratorStep()
            dummy_step_1 = DummyStep1()
            dummy_step_2 = DummyStep2()

            upstream_step >> routing_batch_function >> [dummy_step_1, dummy_step_2]

        assert routing_batch_function._step == upstream_step
        assert list(pipeline.dag.get_step_successors(upstream_step.name)) == [  # type: ignore
            dummy_step_1.name,
            dummy_step_2.name,
        ]


class TestRoutingBatchFunctionDecorator:
    def test_decorator(self) -> None:
        @routing_batch_function
        def random_routing_batch(steps: List[str]) -> List[str]:
            return steps[:2]

        assert isinstance(random_routing_batch, RoutingBatchFunction)

        routed_to = random_routing_batch.route_batch(
            batch=_Batch(seq_no=0, step_name="step_0", data=[[]], last_batch=False),
            steps=["step_1", "step_2", "step_3"],
        )
        assert routed_to == ["step_1", "step_2"]
