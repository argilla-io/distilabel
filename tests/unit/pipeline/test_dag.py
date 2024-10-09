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

from distilabel.constants import STEP_ATTR_NAME
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.pipeline._dag import DAG
from distilabel.pipeline.local import Pipeline
from distilabel.pipeline.routing_batch_function import routing_batch_function
from distilabel.steps.base import GeneratorStep, Step, StepInput, StepResources
from distilabel.steps.typing import StepColumns

from .utils import DummyGeneratorStep, DummyGlobalStep, DummyStep1, DummyStep2

if TYPE_CHECKING:
    from distilabel.steps.typing import (
        GeneratorStepOutput,
        StepOutput,
    )

import base64
from unittest.mock import MagicMock, patch

import requests


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

        assert dag.get_step("dummy_step_1")[STEP_ATTR_NAME] == dummy_step_1

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

    def test_trophic_levels(
        self,
        dummy_generator_step: "GeneratorStep",
        dummy_step_1: "Step",
        dummy_step_2: "Step",
    ) -> None:
        dag = DAG()
        dag.add_step(dummy_generator_step)
        dag.add_step(dummy_step_1)
        dag.add_step(dummy_step_2)
        dag.add_edge("dummy_generator_step", "dummy_step_1")
        dag.add_edge("dummy_step_1", "dummy_step_2")
        assert dag.trophic_levels == {
            "dummy_generator_step": 1,
            "dummy_step_1": 2,
            "dummy_step_2": 3,
        }

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

    def test_iter_based_on_trophic_levels(
        self,
        dummy_generator_step: "GeneratorStep",
        dummy_step_1: "Step",
        dummy_step_2: "Step",
    ) -> None:
        dag = DAG()
        dag.add_step(dummy_generator_step)
        dag.add_step(dummy_step_1)
        dag.add_step(dummy_step_2)
        dag.add_edge("dummy_generator_step", "dummy_step_1")
        dag.add_edge("dummy_step_1", "dummy_step_2")

        steps = list(dag.iter_based_on_trophic_levels())
        assert steps == [["dummy_generator_step"], ["dummy_step_1"], ["dummy_step_2"]]

    def test_get_step_trophic_level(
        self,
        dummy_generator_step: "GeneratorStep",
        dummy_step_1: "Step",
        dummy_step_2: "Step",
    ) -> None:
        dag = DAG()
        dag.add_step(dummy_generator_step)
        dag.add_step(dummy_step_1)
        dag.add_step(dummy_step_2)
        dag.add_edge("dummy_generator_step", "dummy_step_1")
        dag.add_edge("dummy_step_1", "dummy_step_2")

        assert dag.get_step_trophic_level("dummy_generator_step") == 1
        assert dag.get_step_trophic_level("dummy_step_1") == 2
        assert dag.get_step_trophic_level("dummy_step_2") == 3

    def test_step_in_last_trophic_level(
        self,
        dummy_generator_step: "GeneratorStep",
        dummy_step_1: "Step",
        dummy_step_2: "Step",
    ) -> None:
        dag = DAG()
        dag.add_step(dummy_generator_step)
        dag.add_step(dummy_step_1)
        dag.add_step(dummy_step_2)
        dag.add_edge("dummy_generator_step", "dummy_step_1")
        dag.add_edge("dummy_step_1", "dummy_step_2")

        assert not dag.step_in_last_trophic_level("dummy_generator_step")
        assert not dag.step_in_last_trophic_level("dummy_step_1")
        assert dag.step_in_last_trophic_level("dummy_step_2")

    def test_get_total_replica_count(self) -> None:
        dag = DAG()

        # `replicas` should be ignored for `GeneratorStep`
        dag.add_step(DummyGeneratorStep(resources=StepResources(replicas=100)))
        dag.add_step(DummyStep1(resources=StepResources(replicas=5)))
        dag.add_step(DummyStep2(resources=StepResources(replicas=5)))
        # `replicas` should be ignored for `GlobalStep`
        dag.add_step(DummyGlobalStep(resources=StepResources(replicas=100)))

        assert dag.get_total_replica_count() == 12

    def test_get_steps_load_stages(self) -> None:
        with Pipeline(name="dummy") as pipeline:
            generator = DummyGeneratorStep(name="dummy_generator_step")
            dummies_0 = [DummyStep1(name=f"dummy_step_0_{i}") for i in range(3)]
            global_0 = DummyGlobalStep(name="global_0")
            dummies_1 = [DummyStep1(name=f"dummy_step_1_{i}") for i in range(3)]
            global_1 = DummyGlobalStep(name="global_1")

            generator >> dummies_0 >> global_0 >> dummies_1 >> global_1

        assert pipeline.dag.get_steps_load_stages() == (
            [
                [
                    "dummy_generator_step",
                    "dummy_step_0_0",
                    "dummy_step_0_1",
                    "dummy_step_0_2",
                ],
                ["global_0"],
                [
                    "dummy_step_1_0",
                    "dummy_step_1_1",
                    "dummy_step_1_2",
                ],
                ["global_1"],
            ],
            [
                [
                    "dummy_step_0_0",
                    "dummy_step_0_1",
                    "dummy_step_0_2",
                ],
                ["global_0"],
                [
                    "dummy_step_1_0",
                    "dummy_step_1_1",
                    "dummy_step_1_2",
                ],
                ["global_1"],
            ],
        )

    def test_get_steps_load_stages_global_steps_chained(self) -> None:
        with Pipeline(name="dummy") as pipeline:
            generator = DummyGeneratorStep(name="dummy_generator_step")
            dummies_0 = [DummyStep1(name=f"dummy_step_0_{i}") for i in range(3)]
            global_0 = DummyGlobalStep(name="global_0")
            global_1 = DummyGlobalStep(name="global_1")

            generator >> dummies_0 >> global_0 >> global_1

        assert pipeline.dag.get_steps_load_stages() == (
            [
                [
                    "dummy_generator_step",
                    "dummy_step_0_0",
                    "dummy_step_0_1",
                    "dummy_step_0_2",
                ],
                ["global_0"],
                ["global_1"],
            ],
            [
                [
                    "dummy_step_0_0",
                    "dummy_step_0_1",
                    "dummy_step_0_2",
                ],
                ["global_0"],
                ["global_1"],
            ],
        )

    def test_get_steps_load_stages_simple(self) -> None:
        with Pipeline(name="dummy") as pipeline:
            generator = DummyGeneratorStep(name="dummy_generator_step")
            dummies_0 = [DummyStep1(name=f"dummy_step_0_{i}") for i in range(3)]

            generator >> dummies_0

        assert pipeline.dag.get_steps_load_stages() == (
            [
                [
                    "dummy_generator_step",
                    "dummy_step_0_0",
                    "dummy_step_0_1",
                    "dummy_step_0_2",
                ]
            ],
            [
                [
                    "dummy_step_0_0",
                    "dummy_step_0_1",
                    "dummy_step_0_2",
                ]
            ],
        )

    def test_validate_first_step_not_generator(
        self, dummy_step_1: "Step", dummy_step_2: "Step"
    ) -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)
        dag.add_step(dummy_step_2)
        dag.add_edge("dummy_step_1", "dummy_step_2")

        with pytest.raises(
            ValueError,
            match="Step 'dummy_step_1' cannot be a root step because it is not a `GeneratorStep`.",
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
            inputs: StepColumns = ["instruction"]
            outputs: StepColumns = ["response"]

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
            inputs: StepColumns = []
            outputs: StepColumns = []

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
            inputs: StepColumns = []
            outputs: StepColumns = []

            def process(self, offset: int = 0) -> "GeneratorStepOutput":
                yield [{"response": "response1"}], False

        step = DummyGeneratorStep(name="dummy_generator_step", pipeline=pipeline)  # type: ignore
        step.set_runtime_parameters({})

        dag = DAG()
        dag.add_step(step)

        with pytest.raises(
            ValueError,
            match="Step 'dummy_generator_step' is missing required runtime parameter 'runtime_param1'",
        ):
            dag.validate()

    def test_validate_step_process_runtime_parameters(
        self, pipeline: "Pipeline"
    ) -> None:
        class DummyGeneratorStep(GeneratorStep):
            runtime_param1: RuntimeParameter[int]
            runtime_param2: RuntimeParameter[int] = 5
            inputs: StepColumns = ["instruction"]
            outputs: StepColumns = ["response"]

            def process(self, offset: int = 0) -> "GeneratorStepOutput":  # type: ignore
                yield [{"response": "response1"}], False

        step = DummyGeneratorStep(
            name="dummy_generator_step", runtime_param1=2, pipeline=pipeline
        )
        step.set_runtime_parameters({})

        dag = DAG()
        dag.add_step(step)

        dag.validate()

    def test_validate_step_invalid_input_mappings(self, pipeline: "Pipeline") -> None:
        class DummyStep(Step):
            inputs: StepColumns = ["instruction"]
            outputs: StepColumns = ["response"]

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

    def test_validate_step_invalid_output_mappings(self, pipeline: "Pipeline") -> None:
        class DummyStep(Step):
            inputs: StepColumns = ["instruction"]
            outputs: StepColumns = ["response"]

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

    def test_validate_generator_step_process_method_with_step_input(
        self, pipeline: "Pipeline"
    ) -> None:
        class DummyGeneratorStep(GeneratorStep):
            inputs: StepColumns = ["instruction"]
            outputs: StepColumns = ["response"]

            def process(
                self, *inputs: StepInput, offset: int = 0
            ) -> "GeneratorStepOutput":  # type: ignore
                yield [{"response": "response1"}], False

        step = DummyGeneratorStep(name="dummy_generator_step", pipeline=pipeline)

        dag = DAG()
        dag.add_step(step)

        with pytest.raises(
            ValueError,
            match="Generator step 'dummy_generator_step' should not have a parameter with type hint `StepInput`",
        ):
            dag.validate()

    def test_validate_generator_step_process_without_offset_parameter(
        self, pipeline: "Pipeline"
    ) -> None:
        from distilabel.steps.typing import StepColumns

        class DummyGeneratorStep(GeneratorStep):
            inputs: StepColumns = ["instruction"]
            outputs: StepColumns = ["response"]

            def process(self) -> "GeneratorStepOutput":  # type: ignore
                yield [{"response": "response1"}], False

        step = DummyGeneratorStep(name="dummy_generator_step", pipeline=pipeline)

        dag = DAG()
        dag.add_step(step)

        with pytest.raises(
            ValueError,
            match="Generator step 'dummy_generator_step' should have an `offset` parameter",
        ):
            dag.validate()

    def test_validate_convergence_step_receiving_from_same_routed_batch_function(
        self, pipeline: "Pipeline"
    ) -> None:
        generator_step_1 = DummyGeneratorStep(pipeline=pipeline)
        dummy_step_1 = DummyStep1(pipeline=pipeline)
        dummy_step_2 = DummyStep1(pipeline=pipeline)

        generator_step_2 = DummyGeneratorStep(pipeline=pipeline)
        dummy_step_3 = DummyStep1(pipeline=pipeline)
        dummy_step_4 = DummyStep1(pipeline=pipeline)

        convergence_step = DummyStep2(name="convergence_step", pipeline=pipeline)

        @routing_batch_function()
        def routing_batch_function_1(steps: List[str]) -> List[str]:
            return steps

        @routing_batch_function()
        def routing_batch_function_2(steps: List[str]) -> List[str]:
            return steps

        generator_step_1 >> routing_batch_function_1 >> [dummy_step_1, dummy_step_2]

        generator_step_2 >> routing_batch_function_2 >> [dummy_step_3, dummy_step_4]

        [dummy_step_1, dummy_step_2, dummy_step_3, dummy_step_4] >> convergence_step

        with pytest.raises(
            ValueError,
            match=r"Convergence step 'convergence_step' should receive batches from steps receiving"
            " routed batches from the same previous step and `routing_batch_function`.",
        ):
            pipeline.dag.validate()

    def test_validate_convergence_step_input_batch_size(
        self, pipeline: "Pipeline"
    ) -> None:
        generator_step_1 = DummyGeneratorStep(pipeline=pipeline)
        dummy_step_1 = DummyStep1(pipeline=pipeline)
        dummy_step_2 = DummyStep1(pipeline=pipeline)
        convergence_step = DummyStep2(
            name="convergence_step",
            pipeline=pipeline,
            input_batch_size=666,
        )

        @routing_batch_function()
        def routing_batch_function_1(steps: List[str]) -> List[str]:
            return steps

        (
            generator_step_1
            >> routing_batch_function_1
            >> [dummy_step_1, dummy_step_2]
            >> convergence_step
        )

        with pytest.raises(
            ValueError,
            match="A convergence step should have an `input_batch_size` equal or lower",
        ):
            pipeline.dag.validate()

    def test_validate_step_receiving_routed_batches_multiple_predecessors(
        self, pipeline: "Pipeline"
    ) -> None:
        generator_step_1 = DummyGeneratorStep(pipeline=pipeline)
        dummy_step_1 = DummyStep1(pipeline=pipeline)

        generator_step_2 = DummyGeneratorStep(pipeline=pipeline)
        dummy_step_2 = DummyStep1(pipeline=pipeline)

        dummy_step_3 = DummyStep2(name="doomed", pipeline=pipeline)
        dummy_step_4 = DummyStep2(pipeline=pipeline)

        @routing_batch_function()
        def routing_batch_function_1(steps: List[str]) -> List[str]:
            return steps

        (
            generator_step_1
            >> dummy_step_1
            >> routing_batch_function_1
            >> [dummy_step_3, dummy_step_4]
        )

        generator_step_2 >> dummy_step_2 >> dummy_step_3

        with pytest.raises(
            ValueError,
            match="Step 'doomed' cannot have multiple predecessors when the batches of one"
            " are being routed with a `routing_batch_function`.",
        ):
            pipeline.dag.validate()

    def test_validate_step_receiving_routed_batches_input_batch_size(
        self, pipeline: "Pipeline"
    ) -> None:
        generator_step_1 = DummyGeneratorStep(pipeline=pipeline)
        dummy_step_1 = DummyStep1(pipeline=pipeline)
        dummy_step_2 = DummyStep1(name="demon", pipeline=pipeline, input_batch_size=666)

        @routing_batch_function()
        def routing_batch_function_1(steps: List[str]) -> List[str]:
            return steps

        convergence_step = DummyStep2(name="convergence_step", pipeline=pipeline)

        (
            generator_step_1
            >> routing_batch_function_1
            >> [dummy_step_1, dummy_step_2]
            >> convergence_step
        )

        with pytest.raises(
            ValueError,
            match="Step 'demon' should have an `input_batch_size` equal or lower",
        ):
            pipeline.dag.validate()

    def test_validate_step_receiving_routed_batches_input_batch_size_multiple(
        self, pipeline: "Pipeline"
    ) -> None:
        generator_step_1 = DummyGeneratorStep(pipeline=pipeline)
        dummy_step_1 = DummyStep1(pipeline=pipeline)
        dummy_step_2 = DummyStep1(name="demon", pipeline=pipeline, input_batch_size=7)

        @routing_batch_function()
        def routing_batch_function_1(steps: List[str]) -> List[str]:
            return steps

        convergence_step = DummyStep2(name="convergence_step", pipeline=pipeline)
        (
            generator_step_1
            >> routing_batch_function_1
            >> [dummy_step_1, dummy_step_2]
            >> convergence_step
        )
        with pytest.raises(
            ValueError,
            match="Step 'demon' should have an `input_batch_size` that is a multiple of the `input_batch_size` or `batch_size`",
        ):
            pipeline.dag.validate()


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

        with Pipeline(name="unit-test-pipeline"):
            new_dag = DAG.from_dict(dag.dump())
        assert isinstance(new_dag, DAG)
        assert "dummy_step_1" in new_dag.G
        assert "dummy_step_2" in new_dag.G
        assert "dummy_step_2" in new_dag.G["dummy_step_1"]

    def test_dag_from_dict_errored_without_pipeline(
        self,
        caplog: pytest.LogCaptureFixture,
        dummy_step_1: "Step",
        dummy_step_2: "Step",
    ) -> None:
        dag = DAG()
        dag.add_step(dummy_step_1)
        dag.add_step(dummy_step_2)
        dag.add_edge("dummy_step_1", "dummy_step_2")

        DAG.from_dict(dag.dump())
        assert "Step 'dummy_step_1' hasn't received a pipeline" in caplog.text

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
                with Pipeline(name="unit-test-pipeline"):
                    dag_from_file = loader(filename)
                    assert isinstance(dag_from_file, DAG)


class TestDAGDraw:
    @patch("distilabel.pipeline._dag.requests.get")
    def test_draw_basic(self, mock_get):
        # Mock the response from mermaid.ink
        mock_response = MagicMock()
        mock_response.content = b"mocked_image_content"
        mock_get.return_value = mock_response

        dag = DAG()
        generator_step = DummyGeneratorStep(name="generator")
        step1 = DummyStep1(name="step1")
        step2 = DummyStep2(name="step2")

        dag.add_step(generator_step)
        dag.add_step(step1)
        dag.add_step(step2)
        dag.add_edge("generator", "step1")
        dag.add_edge("step1", "step2")

        image_content = dag.draw()

        assert image_content == b"mocked_image_content"
        mock_get.assert_called_once()
        called_url = mock_get.call_args[0][0]
        assert "https://mermaid.ink/img/" in called_url

    @patch("distilabel.pipeline._dag.requests.get")
    def test_draw_top_to_bottom(self, mock_get):
        mock_response = MagicMock()
        mock_response.content = b"mocked_image_content"
        mock_get.return_value = mock_response

        dag = DAG()
        generator_step = DummyGeneratorStep(name="generator")
        step1 = DummyStep1(name="step1")
        dag.add_step(generator_step)
        dag.add_step(step1)
        dag.add_edge("generator", "step1")

        dag.draw(top_to_bottom=True)

        called_url = mock_get.call_args[0][0]
        decoded_graph = base64.b64decode(
            called_url.split("/")[-1].split("?")[0]
        ).decode("ascii")
        assert "flowchart TD" in decoded_graph

    @patch("distilabel.pipeline._dag.requests.get")
    def test_draw_without_edge_labels(self, mock_get):
        mock_response = MagicMock()
        mock_response.content = b"mocked_image_content"
        mock_get.return_value = mock_response

        dag = DAG()
        generator_step = DummyGeneratorStep(name="generator")
        step1 = DummyStep1(name="step1")
        dag.add_step(generator_step)
        dag.add_step(step1)
        dag.add_edge("generator", "step1")

        dag.draw(show_edge_labels=False)

        called_url = mock_get.call_args[0][0]
        decoded_graph = base64.b64decode(
            called_url.split("/")[-1].split("?")[0]
        ).decode("ascii")
        assert "generator --> step1" in decoded_graph
        assert "|" not in decoded_graph  # No edge labels

    @patch("distilabel.pipeline._dag.requests.get")
    def test_draw_with_argilla_step(self, mock_get):
        mock_response = MagicMock()
        mock_response.content = b"mocked_image_content"
        mock_get.return_value = mock_response

        dag = DAG()
        generator_step = DummyGeneratorStep(name="generator")
        step1 = DummyStep1(name="to_argilla")
        dag.add_step(generator_step)
        dag.add_step(step1)
        dag.add_edge("generator", "to_argilla")

        dag.draw()

        called_url = mock_get.call_args[0][0]
        decoded_graph = base64.b64decode(
            called_url.split("/")[-1].split("?")[0]
        ).decode("ascii")
        assert 'to_argilla_0["Argilla"]' in decoded_graph

    @patch("distilabel.pipeline._dag.requests.get")
    def test_draw_with_distiset_step(self, mock_get):
        mock_response = MagicMock()
        mock_response.content = b"mocked_image_content"
        mock_get.return_value = mock_response

        dag = DAG()
        generator_step = DummyGeneratorStep(name="generator")
        step1 = DummyStep1(name="step1")
        dag.add_step(generator_step)
        dag.add_step(step1)
        dag.add_edge("generator", "step1")

        dag.draw()

        called_url = mock_get.call_args[0][0]
        decoded_graph = base64.b64decode(
            called_url.split("/")[-1].split("?")[0]
        ).decode("ascii")
        assert 'distiset_0["Distiset"]' in decoded_graph

    @patch("distilabel.pipeline._dag.requests.get")
    def test_draw_error_handling(self, mock_get):
        mock_get.side_effect = requests.RequestException("Mocked error")

        dag = DAG()
        generator_step = DummyGeneratorStep(name="generator")
        dag.add_step(generator_step)

        with pytest.raises(ValueError, match="Error accessing https://mermaid.ink/"):
            dag.draw()
