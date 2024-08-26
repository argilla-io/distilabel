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

from distilabel.pipeline.batch import _Batch
from distilabel.pipeline.local import Pipeline
from distilabel.pipeline.routing_batch_function import (
    RoutingBatchFunction,
    routing_batch_function,
    sample_n_steps,
)
from distilabel.utils.serialization import TYPE_INFO_KEY
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

    def test_dump(self) -> None:
        routing_batch_function = sample_n_steps(n=2)

        with Pipeline(name="test"):
            upstream_step = DummyGeneratorStep()
            dummy_step_1 = DummyStep1()
            dummy_step_2 = DummyStep2()

            upstream_step >> routing_batch_function >> [dummy_step_1, dummy_step_2]

        assert routing_batch_function.dump() == {
            "step": upstream_step.name,
            "description": routing_batch_function.description,
            TYPE_INFO_KEY: {
                "module": "distilabel.pipeline.routing_batch_function",
                "name": "sample_n_steps",
                "kwargs": {"n": 2},
            },
        }

    def test_from_dict(self) -> None:
        routing_batch_function_dict = {
            "step": "upstream_step",
            "description": "Sample 2 steps from the list of downstream steps.",
            TYPE_INFO_KEY: {
                "module": "distilabel.pipeline.routing_batch_function",
                "name": "sample_n_steps",
                "kwargs": {"n": 2},
            },
        }

        dummy_step_1 = DummyStep1(name="upstream_step")
        routing_batch_function = sample_n_steps(n=2)
        routing_batch_function._step = dummy_step_1

        routing_batch_function_from_dict = RoutingBatchFunction.from_dict(
            routing_batch_function_dict
        )
        routing_batch_function_from_dict._step = dummy_step_1

        assert routing_batch_function_from_dict.dump() == routing_batch_function.dump()


class TestRoutingBatchFunctionDecorator:
    def test_decorator(self) -> None:
        @routing_batch_function()
        def random_routing_batch(steps: List[str]) -> List[str]:
            return steps[:2]

        assert isinstance(random_routing_batch, RoutingBatchFunction)

        routed_to = random_routing_batch.route_batch(
            batch=_Batch(seq_no=0, step_name="step_0", data=[[]], last_batch=False),
            steps=["step_1", "step_2", "step_3"],
        )
        assert routed_to == ["step_1", "step_2"]


def test_sample_n_steps() -> None:
    steps = ["step_a", "step_b", "step_c", "step_d"]

    routing_batch_function = sample_n_steps(n=2)

    selected_steps = routing_batch_function(
        batch=_Batch(seq_no=0, step_name="step_z", last_batch=False, data=[[]]),
        steps=steps,
    )

    assert len(selected_steps) == 2
    for step in selected_steps:
        assert step in steps
