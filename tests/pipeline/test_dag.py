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

from typing import Any, Dict, List

import pytest
from distilabel.pipeline._dag import DAG
from distilabel.step.base import GeneratorStep, Step, StepInput


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


class TestDAG:
    def test_add_step(self) -> None:
        dag = DAG()
        dag.add_step(DummyStep1(), "step1")

        assert "step1" in dag.dag

    def test_add_step_with_existing_name(self) -> None:
        dag = DAG()
        dag.add_step(DummyStep1(), "step1")

        with pytest.raises(ValueError, match="Step with name 'step1' already exists"):
            dag.add_step(DummyStep1(), "step1")

    def test_add_edge(self) -> None:
        dag = DAG()
        dag.add_step(DummyStep1(), "step1")
        dag.add_step(DummyStep2(), "step2")
        dag.add_edge("step1", "step2")

        assert "step2" in dag.dag["step1"]

    def test_add_edge_with_nonexistent_step(self) -> None:
        dag = DAG()
        dag.add_step(DummyStep1(), "step1")

        with pytest.raises(ValueError, match="Step with name 'step2' does not exist"):
            dag.add_edge("step1", "step2")

        with pytest.raises(ValueError, match="Step with name 'step2' does not exist"):
            dag.add_edge("step2", "step1")

    def test_add_edge_duplicate(self) -> None:
        dag = DAG()
        dag.add_step(DummyStep1(), "step1")
        dag.add_step(DummyStep2(), "step2")
        dag.add_edge("step1", "step2")

        with pytest.raises(
            ValueError, match="There is already a edge from 'step2' to 'step1'"
        ):
            dag.add_edge("step1", "step2")

    def test_add_edge_cycle(self) -> None:
        dag = DAG()
        dag.add_step(DummyStep1(), "step1")
        dag.add_step(DummyStep2(), "step2")
        dag.add_edge("step1", "step2")

        with pytest.raises(
            ValueError,
            match="Cannot add edge from 'step2' to 'step1' as it would create a cycle.",
        ):
            dag.add_edge("step2", "step1")

    def test_validate_first_step_not_generator(self) -> None:
        dag = DAG()
        dag.add_step(DummyStep1(), "step1")
        dag.add_step(DummyStep2(), "step2")
        dag.add_edge("step1", "step2")

        with pytest.raises(
            ValueError,
            match="Step 'step1' should be `GeneratorStep` as it doesn't have any previous steps",
        ):
            dag.validate()

    def test_validate_inputs_not_available(self) -> None:
        dag = DAG()
        dag.add_step(DummyGeneratorStep(), "step1")
        dag.add_step(DummyStep2(), "step2")
        dag.add_edge("step1", "step2")

        with pytest.raises(
            ValueError,
            match="Step 'step2' requires inputs 'response'",
        ):
            dag.validate()

    def test_validate_missing_step_input(self) -> None:
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
        dag.add_step(DummyGeneratorStep(), "step1")
        dag.add_step(DummyStep3(), "step3")
        dag.add_edge("step1", "step3")

        with pytest.raises(
            ValueError,
            match="Step 'step3' should have a parameter with type hint `StepInput`",
        ):
            dag.validate()

    def test_validate_missing_var_positional_step_input(self) -> None:
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
        dag.add_step(DummyGeneratorStep(), "step11")
        dag.add_step(DummyGeneratorStep(), "step12")
        dag.add_step(DummyStep3(), "step3")

        dag.add_edge("step11", "step3")
        dag.add_edge("step12", "step3")

        with pytest.raises(
            ValueError,
            match=r"Step 'step3' should have a `\*args` parameter with type hint `StepInput`",
        ):
            dag.validate()
