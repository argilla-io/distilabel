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

from typing import TYPE_CHECKING, Any, Dict, List

import pytest
from distilabel.pipeline._dag import DAG
from distilabel.step.base import GeneratorStep, Step, StepInput

if TYPE_CHECKING:
    from distilabel.pipeline.local import Pipeline


class DummyGeneratorStep(GeneratorStep):
    @property
    def inputs(self) -> List[str]:
        return []

    def process(self) -> List[Dict[str, Any]]:
        return [{"instruction": "Generate an email..."}]

    @property
    def outputs(self) -> List[str]:
        return ["instruction"]


class DummyStep1(Step):
    @property
    def inputs(self) -> List[str]:
        return ["instruction"]

    def process(self, StepInput) -> List[Dict[str, Any]]:
        return [{"response": "response1"}]

    @property
    def outputs(self) -> List[str]:
        return ["response"]


class DummyStep2(Step):
    @property
    def inputs(self) -> List[str]:
        return ["response"]

    def process(self, *inputs: StepInput) -> List[Dict[str, Any]]:
        return [{"response": "response1"}]

    @property
    def outputs(self) -> List[str]:
        return ["evol_response"]


@pytest.fixture(name="dummy_step_1")
def dummy_step_1_fixture(pipeline: "Pipeline") -> DummyStep1:
    return DummyStep1(name="dummy_step_1", pipeline=pipeline)


@pytest.fixture(name="dummy_step_2")
def dummy_step_2_fixture(pipeline: "Pipeline") -> DummyStep2:
    return DummyStep2(name="dummy_step_2", pipeline=pipeline)


@pytest.fixture(name="dummy_generator_step")
def dummy_generator_step_fixture(pipeline: "Pipeline") -> DummyGeneratorStep:
    return DummyGeneratorStep(name="dummy_generator_step", pipeline=pipeline)


class TestDAG:
    def test_add_step(self, dummy_step_1: "Step") -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)

        assert "dummy_step_1" in dag.dag

    def test_add_step_with_existing_name(self, dummy_step_1: "Step") -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)

        with pytest.raises(
            ValueError, match="Step with name 'dummy_step_1' already exists"
        ):
            dag.add_step(dummy_step_1)

    def test_add_edge(self, dummy_step_1: "Step", dummy_step_2: "Step") -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)
        dag.add_step(dummy_step_2)
        dag.add_edge("dummy_step_1", "dummy_step_2")

        assert "dummy_step_2" in dag.dag["dummy_step_1"]

    def test_add_edge_with_nonexistent_step(self, dummy_step_1: "Step") -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)

        with pytest.raises(
            ValueError, match="Step with name 'dummy_step_2' does not exist"
        ):
            dag.add_edge("dummy_step_1", "dummy_step_2")

        with pytest.raises(
            ValueError, match="Step with name 'dummy_step_2' does not exist"
        ):
            dag.add_edge("dummy_step_2", "dummy_step_1")

    def test_add_edge_duplicate(
        self, dummy_step_1: "Step", dummy_step_2: "Step"
    ) -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)
        dag.add_step(dummy_step_2)
        dag.add_edge("dummy_step_1", "dummy_step_2")

        with pytest.raises(
            ValueError,
            match="There is already a edge from 'dummy_step_2' to 'dummy_step_1'",
        ):
            dag.add_edge("dummy_step_1", "dummy_step_2")

    def test_add_edge_cycle(self, dummy_step_1: "Step", dummy_step_2: "Step") -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)
        dag.add_step(dummy_step_2)
        dag.add_edge("dummy_step_1", "dummy_step_2")

        with pytest.raises(
            ValueError,
            match="Cannot add edge from 'dummy_step_2' to 'dummy_step_1' as it would create a cycle.",
        ):
            dag.add_edge("dummy_step_2", "dummy_step_1")

    def test_validate_first_step_not_generator(
        self, dummy_step_1: "Step", dummy_step_2: "Step"
    ) -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)
        dag.add_step(dummy_step_2)
        dag.add_edge("dummy_step_1", "dummy_step_2")

        with pytest.raises(
            ValueError,
            match="Step 'dummy_step_1' should be `GeneratorStep` as it doesn't have any previous steps",
        ):
            dag.validate()

    def test_validate_inputs_not_available(
        self, dummy_generator_step: "GeneratorStep", dummy_step_2: "Step"
    ) -> None:
        dag = DAG()
        dag.add_step(dummy_generator_step)
        dag.add_step(dummy_step_2)
        dag.add_edge("dummy_generator_step", "dummy_step_2")

        with pytest.raises(
            ValueError,
            match="Step 'dummy_step_2' requires inputs 'response'",
        ):
            dag.validate()

    def test_validate_missing_step_input(
        self, dummy_generator_step: "GeneratorStep", pipeline: "Pipeline"
    ) -> None:
        class DummyStep3(Step):
            @property
            def inputs(self) -> List[str]:
                return ["instruction"]

            @property
            def outputs(self) -> List[str]:
                return ["response"]

            def process(self) -> List[Dict[str, Any]]:
                return [{"response": "response1"}]

        dag = DAG()
        dag.add_step(dummy_generator_step)
        dag.add_step(DummyStep3(name="dummy_step_3", pipeline=pipeline))
        dag.add_edge("dummy_generator_step", "dummy_step_3")

        with pytest.raises(
            ValueError,
            match="Step 'dummy_step_3' should have a parameter with type hint `StepInput`",
        ):
            dag.validate()

    def test_validate_missing_var_positional_step_input(
        self, pipeline: "Pipeline"
    ) -> None:
        class DummyStep3(Step):
            @property
            def inputs(self) -> List[str]:
                return []

            @property
            def outputs(self) -> List[str]:
                return []

            def process(self) -> List[Dict[str, Any]]:
                return []

        dag = DAG()
        dag.add_step(
            DummyGeneratorStep(name="dummy_generator_step_11", pipeline=pipeline)
        )
        dag.add_step(
            DummyGeneratorStep(name="dummy_generator_step_12", pipeline=pipeline)
        )
        dag.add_step(DummyStep3(name="dummy_step_3", pipeline=pipeline))

        dag.add_edge("dummy_generator_step_11", "dummy_step_3")
        dag.add_edge("dummy_generator_step_12", "dummy_step_3")

        with pytest.raises(
            ValueError,
            match=r"Step 'dummy_step_3' should have a `\*args` parameter with type hint `StepInput`",
        ):
            dag.validate()
