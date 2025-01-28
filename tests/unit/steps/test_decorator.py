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

from unittest import mock

import pytest

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.pipeline.local import Pipeline
from distilabel.steps.base import (
    GeneratorStep,
    GlobalStep,
    Step,
    StepInput,
)
from distilabel.steps.decorator import step
from distilabel.typing import GeneratorStepOutput, StepOutput


class TestStepDecorator:
    def test_creating_step(self) -> None:
        @step(inputs=["instruction"], outputs=["generation"])
        def UnitTestStep(
            inputs: StepInput,
            runtime_param1: RuntimeParameter[int],
            runtime_param2: RuntimeParameter[float] = 5.0,
        ) -> StepOutput:
            """A dummy step for the unit test"""
            yield []

        assert issubclass(UnitTestStep, Step)
        assert UnitTestStep.__doc__ == "A dummy step for the unit test"
        assert UnitTestStep.__module__ == "tests.unit.steps.test_decorator"

        unit_test_step = UnitTestStep(
            name="unit_test_step", pipeline=Pipeline(name="unit-test-pipeline")
        )
        assert unit_test_step._built_from_decorator is True
        assert unit_test_step.inputs == ["instruction"]
        assert unit_test_step.outputs == ["generation"]
        assert unit_test_step.runtime_parameters_names == {
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
        }

    def test_step_decoraror_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid step type 'invalid'"):

            @step(step_type="invalid")
            def UnitTestStep(inputs: StepInput) -> StepOutput:
                yield []

    def test_creating_step_with_more_than_one_step_input(self) -> None:
        with pytest.raises(
            ValueError,
            match="Function 'UnitTestStep' has more than one parameter annotated with `StepInput`.",
        ):

            @step(inputs=["instruction"], outputs=["generation"])
            def UnitTestStep(inputs: StepInput, inputs2: StepInput) -> StepOutput:
                """A dummy step for the unit test"""
                yield []

    def test_creating_global_step(self) -> None:
        @step(inputs=["instruction"], outputs=["generation"], step_type="global")
        def UnitTestStep(
            inputs: StepInput,
            runtime_param1: RuntimeParameter[int],
            runtime_param2: RuntimeParameter[float] = 5.0,
        ) -> StepOutput:
            yield []

        assert issubclass(UnitTestStep, GlobalStep)

    def test_creating_generator_step(self) -> None:
        @step(outputs=["generation"], step_type="generator")
        def UnitTestStep(
            inputs: StepInput,
            runtime_param1: RuntimeParameter[int],
            runtime_param2: RuntimeParameter[float] = 5.0,
        ) -> GeneratorStepOutput:
            yield [], False

        assert issubclass(UnitTestStep, GeneratorStep)

    def test_processing(self) -> None:
        @step(inputs=["instruction"], outputs=["generation"])
        def UnitTestStep(
            inputs: StepInput,
            runtime_param1: RuntimeParameter[int],
            runtime_param2: RuntimeParameter[float] = 5.0,
        ) -> StepOutput:
            yield []

        inputs = [[{"instruction": "Build AGI please"}]]

        with mock.patch.object(
            UnitTestStep,
            "process",
            return_value=[[{"instruction": "Build AGI please", "generation": ""}]],
        ) as process_mock:
            unit_test_step = UnitTestStep(
                name="unit_test_step", pipeline=Pipeline(name="unit-test-pipeline")
            )
            next(
                unit_test_step.process_applying_mappings(
                    inputs, **unit_test_step._runtime_parameters
                )
            )

        process_mock.assert_called_once_with(
            inputs, **unit_test_step._runtime_parameters
        )
