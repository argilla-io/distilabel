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

from typing import List, Optional

import pytest
from pydantic import ValidationError

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.pipeline.constants import ROUTING_BATCH_FUNCTION_ATTR_NAME
from distilabel.pipeline.local import Pipeline
from distilabel.steps.base import GeneratorStep, GlobalStep, Step, StepInput
from distilabel.steps.decorator import step
from distilabel.steps.typing import GeneratorStepOutput, StepOutput
from distilabel.utils.serialization import TYPE_INFO_KEY


class DummyStep(Step):
    @property
    def inputs(self) -> List[str]:
        return ["instruction"]

    @property
    def outputs(self) -> List[str]:
        return ["response"]

    def process(self, inputs: StepInput) -> StepOutput:  # type: ignore
        for input in inputs:
            input["response"] = "unit test"
        yield inputs


class DummyGeneratorStep(GeneratorStep):
    @property
    def outputs(self) -> List[str]:
        return []

    def process(self, inputs: StepInput) -> GeneratorStepOutput:  # type: ignore
        yield [], False


class DummyGlobalStep(GlobalStep):
    @property
    def inputs(self) -> List[str]:
        return []

    @property
    def outputs(self) -> List[str]:
        return []

    def process(self, inputs: StepInput) -> StepOutput:
        yield []


class TestStep:
    def test_create_step_with_invalid_name(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")

        with pytest.raises(ValidationError):
            DummyStep(
                name="this-is-not-va.li.d-because-it-contains-dots", pipeline=pipeline
            )

        with pytest.raises(ValidationError):
            DummyStep(name="this is not valid because spaces", pipeline=pipeline)

    def test_create_step_and_infer_name(self) -> None:
        dummy_step = DummyStep(pipeline=Pipeline(name="unit-test-pipeline"))
        assert dummy_step.name == "dummy_step_0"

    def test_create_step_passing_pipeline(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        step = DummyStep(name="dummy", pipeline=pipeline)
        assert step.pipeline == pipeline

    def test_create_step_within_pipeline_context(self) -> None:
        with Pipeline(name="unit-test-pipeline") as pipeline:
            step = DummyStep(name="dummy")

        assert step.pipeline == pipeline

    def test_creating_step_without_pipeline(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # This test is to ensure that the warning is raised when creating a step without a pipeline,
        # vs the error we raised before.
        dummy_step = DummyStep(name="dummy")
        assert f"Step '{dummy_step.name}' hasn't received a pipeline" in caplog.text

    def test_is_generator(self) -> None:
        step = DummyStep(name="dummy", pipeline=Pipeline(name="unit-test-pipeline"))
        assert not step.is_generator

    def test_is_global(self) -> None:
        step = DummyStep(name="dummy", pipeline=Pipeline(name="unit-test-pipeline"))
        assert not step.is_global

    def test_runtime_parameters_names(self) -> None:
        class StepWithRuntimeParameters(Step):
            runtime_param1: RuntimeParameter[int]
            runtime_param2: RuntimeParameter[str] = "hello"
            runtime_param3: Optional[RuntimeParameter[str]] = None

            def process(self, *inputs: StepInput) -> StepOutput:
                yield []

        step = StepWithRuntimeParameters(
            name="dummy", pipeline=Pipeline(name="unit-test-pipeline")
        )  # type: ignore

        assert step.runtime_parameters_names == {
            "input_batch_size": True,
            "resources": {
                "cpus": True,
                "gpus": True,
                "replicas": True,
                "memory": True,
                "resources": True,
            },
            "runtime_param1": False,
            "runtime_param2": True,
            "runtime_param3": True,
        }

    def test_verify_inputs_mappings(self) -> None:
        step = DummyStep(name="dummy", pipeline=Pipeline(name="unit-test-pipeline"))

        step.verify_inputs_mappings()

        step.input_mappings = {"im_not_an_input": "prompt"}
        with pytest.raises(
            ValueError, match="The input column 'im_not_an_input' doesn't exist"
        ):
            step.verify_inputs_mappings()

    def test_verify_outputs_mappings(self) -> None:
        step = DummyStep(name="dummy", pipeline=Pipeline(name="unit-test-pipeline"))

        step.verify_outputs_mappings()

        step.output_mappings = {"im_not_an_output": "prompt"}
        with pytest.raises(
            ValueError, match="The output column 'im_not_an_output' doesn't exist"
        ):
            step.verify_outputs_mappings()

    def test_get_inputs(self) -> None:
        step = DummyStep(
            name="dummy",
            pipeline=Pipeline(name="unit-test-pipeline"),
            input_mappings={"instruction": "prompt"},
        )
        assert step.get_inputs() == ["prompt"]

    def test_get_outputs(self) -> None:
        step = DummyStep(
            name="dummy",
            pipeline=Pipeline(name="unit-test-pipeline"),
            output_mappings={"response": "generation"},
        )
        assert step.get_outputs() == ["generation"]

    def test_apply_input_mappings(self) -> None:
        step = DummyStep(
            name="dummy",
            pipeline=Pipeline(name="unit-test-pipeline"),
            input_mappings={"instruction": "prompt"},
        )

        inputs = step._apply_input_mappings(
            (
                [
                    {"prompt": "hello 1"},
                    {"prompt": "hello 2"},
                    {"prompt": "hello 3"},
                ],
                [
                    {"prompt": "bye 1"},
                    {"prompt": "bye 2"},
                    {"prompt": "bye 3"},
                ],
            )
        )

        assert inputs == [
            [
                {"instruction": "hello 1"},
                {"instruction": "hello 2"},
                {"instruction": "hello 3"},
            ],
            [
                {"instruction": "bye 1"},
                {"instruction": "bye 2"},
                {"instruction": "bye 3"},
            ],
        ]

    def test_process_applying_mappings(self) -> None:
        step = DummyStep(
            name="dummy",
            pipeline=Pipeline(name="unit-test-pipeline"),
            input_mappings={"instruction": "prompt"},
            output_mappings={"response": "generation"},
        )

        outputs = next(
            step.process_applying_mappings(
                [
                    {"prompt": "hello 1"},
                    {"prompt": "hello 2"},
                    {"prompt": "hello 3"},
                ]
            )
        )

        assert outputs == [
            {"prompt": "hello 1", "generation": "unit test"},
            {"prompt": "hello 2", "generation": "unit test"},
            {"prompt": "hello 3", "generation": "unit test"},
        ]

    def test_connect(self) -> None:
        @step(inputs=["instruction"], outputs=["generation"])
        def GenerationStep(input: StepInput):
            pass

        def routing_batch_function(downstream_step_names: List[str]) -> List[str]:
            return downstream_step_names

        with Pipeline(name="unit-test-pipeline") as pipeline:
            generator_step = DummyGeneratorStep(name="dummy_generator")

            generate_1 = GenerationStep(name="generate_1")
            generate_2 = GenerationStep(name="generate_2")
            generate_3 = GenerationStep(name="generate_3")

            generator_step.connect(
                generate_1,
                generate_2,
                generate_3,
                routing_batch_function=routing_batch_function,
            )

        assert "generate_1" in pipeline.dag.G["dummy_generator"]
        assert "generate_2" in pipeline.dag.G["dummy_generator"]
        assert "generate_3" in pipeline.dag.G["dummy_generator"]
        assert (
            pipeline.dag.get_step("dummy_generator")[ROUTING_BATCH_FUNCTION_ATTR_NAME]
            == routing_batch_function
        )


class TestGeneratorStep:
    def test_is_generator(self) -> None:
        step = DummyGeneratorStep(
            name="dummy", pipeline=Pipeline(name="unit-test-pipeline")
        )
        assert step.is_generator

    def test_is_global(self) -> None:
        step = DummyGeneratorStep(
            name="dummy", pipeline=Pipeline(name="unit-test-pipeline")
        )
        assert not step.is_global


class TestGlobalStep:
    def test_is_generator(self) -> None:
        step = DummyGlobalStep(
            name="dummy", pipeline=Pipeline(name="unit-test-pipeline")
        )
        assert not step.is_generator

    def test_is_global(self) -> None:
        step = DummyGlobalStep(
            name="dummy", pipeline=Pipeline(name="unit-test-pipeline")
        )
        assert step.is_global


class TestStepSerialization:
    def test_step_dump(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        step = DummyStep(name="dummy", pipeline=pipeline)
        assert step.dump() == {
            "name": "dummy",
            "input_batch_size": 50,
            "input_mappings": {},
            "output_mappings": {},
            "resources": {
                "cpus": None,
                "gpus": None,
                "memory": None,
                "replicas": 1,
                "resources": None,
            },
            "runtime_parameters_info": [
                {
                    "name": "resources",
                    "runtime_parameters_info": [
                        {
                            "description": "The number of replicas for the step.",
                            "name": "replicas",
                            "optional": True,
                        },
                        {
                            "description": "The number of CPUs assigned to each step replica.",
                            "name": "cpus",
                            "optional": True,
                        },
                        {
                            "description": "The number of GPUs assigned to each step replica.",
                            "name": "gpus",
                            "optional": True,
                        },
                        {
                            "description": "The memory in bytes required for each step replica.",
                            "name": "memory",
                            "optional": True,
                        },
                        {
                            "description": "A dictionary containing names of custom resources and the number of those resources required for each step replica.",
                            "name": "resources",
                            "optional": True,
                        },
                    ],
                },
                {
                    "description": "The number of rows that will contain the batches processed by the step.",
                    "name": "input_batch_size",
                    "optional": True,
                },
            ],
            TYPE_INFO_KEY: {
                "module": "tests.unit.steps.test_base",
                "name": "DummyStep",
            },
        }

    def test_step_from_dict(self) -> None:
        with Pipeline(name="unit-test-pipeline"):
            step = DummyStep.from_dict(
                {
                    **{
                        "name": "dummy",
                        TYPE_INFO_KEY: {
                            "module": "tests.unit.steps.test_base",
                            "name": "DummyStep",
                        },
                    }
                }
            )

        assert isinstance(step, DummyStep)

    def test_step_from_dict_without_pipeline_context(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        dummy_step = DummyStep.from_dict(
            {
                **{
                    "name": "dummy",
                    TYPE_INFO_KEY: {
                        "module": "tests.unit.steps.test_base",
                        "name": "DummyStep",
                    },
                }
            }
        )
        assert f"Step '{dummy_step.name}' hasn't received a pipeline" in caplog.text
