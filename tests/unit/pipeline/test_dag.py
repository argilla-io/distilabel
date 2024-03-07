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

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import pytest
from distilabel.pipeline._dag import DAG
from distilabel.pipeline.local import Pipeline
from distilabel.steps.base import GeneratorStep, RuntimeParameter, Step, StepInput

from .utils import DummyGeneratorStep

if TYPE_CHECKING:
    from distilabel.steps.typing import (
        GeneratorStepOutput,
        StepOutput,
    )


class TestDAG:
    def test_add_step(self, dummy_step_1: "Step") -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)

        assert "dummy_step_1" in dag.G

    def test_add_step_with_existing_name(self, dummy_step_1: "Step") -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)

        with pytest.raises(
            ValueError, match="Step with name 'dummy_step_1' already exists"
        ):
            dag.add_step(dummy_step_1)

    def test_get_step(self, dummy_step_1: "Step") -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)

        assert dag.get_step("dummy_step_1")["step"] == dummy_step_1

    def test_get_step_nonexistent(self) -> None:
        dag = DAG()
        with pytest.raises(
            ValueError, match="Step with name 'dummy_step_1' does not exist"
        ):
            dag.get_step("dummy_step_1")

    def test_set_step_attr(self, dummy_step_1: "Step") -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)
        dag.set_step_attr("dummy_step_1", "attr", "value")
        assert dag.get_step("dummy_step_1")["attr"] == "value"

    def test_set_step_attr_nonexistent(self) -> None:
        dag = DAG()
        with pytest.raises(
            ValueError, match="Step with name 'dummy_step_1' does not exist"
        ):
            dag.set_step_attr("dummy_step_1", "attr", "value")

    def test_add_edge(self, dummy_step_1: "Step", dummy_step_2: "Step") -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)
        dag.add_step(dummy_step_2)
        dag.add_edge("dummy_step_1", "dummy_step_2")

        assert "dummy_step_2" in dag.G["dummy_step_1"]

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

    def test_root_steps(self, dummy_step_1: "Step", dummy_step_2: "Step") -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)
        dag.add_step(dummy_step_2)
        dag.add_edge("dummy_step_1", "dummy_step_2")
        assert dag.root_steps == {"dummy_step_1"}

    def test_leaf_steps(self, dummy_step_1: "Step", dummy_step_2: "Step") -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)
        dag.add_step(dummy_step_2)
        dag.add_edge("dummy_step_1", "dummy_step_2")
        assert dag.leaf_steps == {"dummy_step_2"}

    def get_step_predecessors(self, dummy_step_1: "Step", dummy_step_2: "Step") -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)
        dag.add_step(dummy_step_2)
        dag.add_edge("dummy_step_1", "dummy_step_2")
        assert list(dag.get_step_predecessors("dummy_step_2")) == ["dummy_step_1"]

    def test_get_step_successors(
        self, dummy_step_1: "Step", dummy_step_2: "Step"
    ) -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)
        dag.add_step(dummy_step_2)
        dag.add_edge("dummy_step_1", "dummy_step_2")
        assert list(dag.get_step_successors("dummy_step_1")) == ["dummy_step_2"]

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
            match=r"Step 'dummy_step_2' requires inputs \['response'\]",
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

            def process(self) -> "StepOutput":  # type: ignore
                yield [{"response": "response1"}]

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

    def test_validate_missing_runtime_parameter(self, pipeline: "Pipeline") -> None:
        class DummyGeneratorStep(GeneratorStep):
            runtime_param1: RuntimeParameter[int]
            runtime_param2: RuntimeParameter[int] = 5

            @property
            def inputs(self) -> List[str]:
                return ["instruction"]

            @property
            def outputs(self) -> List[str]:
                return ["response"]

            def process(self) -> "GeneratorStepOutput":
                yield [{"response": "response1"}], False

        step = DummyGeneratorStep(name="dummy_generator_step", pipeline=pipeline)  # type: ignore
        step._set_runtime_parameters({})

        dag = DAG()
        dag.add_step(step)

        with pytest.raises(
            ValueError,
            match="Step 'dummy_generator_step' is missing required runtime parameter 'runtime_param1'",
        ):
            dag.validate()

    def test_step_invalid_input_mappings(self, pipeline: "Pipeline") -> None:
        class DummyStep(Step):
            @property
            def inputs(self) -> List[str]:
                return ["instruction"]

            @property
            def outputs(self) -> List[str]:
                return ["response"]

            def process(self, *inputs: StepInput) -> "StepOutput":
                yield []

        step = DummyStep(
            name="dummy_step",
            pipeline=pipeline,
            input_mappings={"i_do_not_exist": "prompt"},
        )

        dag = DAG()
        dag.add_step(step)

        with pytest.raises(
            ValueError,
            match="The input column 'i_do_not_exist' doesn't exist in the inputs",
        ):
            dag.validate()

    def test_step_invalid_output_mappings(self, pipeline: "Pipeline") -> None:
        class DummyStep(Step):
            @property
            def inputs(self) -> List[str]:
                return ["instruction"]

            @property
            def outputs(self) -> List[str]:
                return ["response"]

            def process(self, *inputs: StepInput) -> "StepOutput":
                yield []

        step = DummyStep(
            name="dummy_step",
            pipeline=pipeline,
            output_mappings={"i_do_not_exist": "generation"},
        )

        dag = DAG()
        dag.add_step(step)

        with pytest.raises(
            ValueError,
            match="The output column 'i_do_not_exist' doesn't exist in the outputs",
        ):
            dag.validate()


class TestDagSerialization:
    def test_dag_dump(self, dummy_step_1: "Step", dummy_step_2: "Step") -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)
        dag.add_step(dummy_step_2)
        dag.add_edge("dummy_step_1", "dummy_step_2")

        dump = dag.dump()

        assert "steps" in dump
        assert len(dump["steps"]) == 2
        assert "connections" in dump
        assert dump["connections"] == [
            {"from": "dummy_step_1", "to": ["dummy_step_2"]},
            {"from": "dummy_step_2", "to": []},
        ]

    def test_dag_from_dict(self, dummy_step_1: "Step", dummy_step_2: "Step") -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)
        dag.add_step(dummy_step_2)
        dag.add_edge("dummy_step_1", "dummy_step_2")

        with Pipeline():
            new_dag = DAG.from_dict(dag.dump())
        assert isinstance(new_dag, DAG)
        assert "dummy_step_1" in new_dag.G
        assert "dummy_step_2" in new_dag.G
        assert "dummy_step_2" in new_dag.G["dummy_step_1"]

    def test_dag_from_dict_errored_without_pipeline(
        self, dummy_step_1: "Step", dummy_step_2: "Step"
    ) -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)
        dag.add_step(dummy_step_2)
        dag.add_edge("dummy_step_1", "dummy_step_2")

        with pytest.raises(ValueError):
            DAG.from_dict(dag.dump())

    @pytest.mark.parametrize(
        "format, name, loader",
        [
            ("yaml", "dag.yaml", DAG.from_yaml),
            ("json", "dag.json", DAG.from_json),
            ("invalid", "dag.invalid", None),
        ],
    )
    def test_dag_to_from_file_format(
        self,
        dummy_step_1: "Step",
        dummy_step_2: "Step",
        format: str,
        name: str,
        loader: Callable,
    ) -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)
        dag.add_step(dummy_step_2)
        dag.add_edge("dummy_step_1", "dummy_step_2")

        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = Path(tmpdirname) / name
            if format == "invalid":
                with pytest.raises(ValueError):
                    dag.save(filename, format=format)
            else:
                dag.save(filename, format=format)
                assert filename.exists()
                with Pipeline():
                    dag_from_file = loader(filename)
                    assert isinstance(dag_from_file, DAG)
