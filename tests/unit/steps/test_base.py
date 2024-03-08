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
from distilabel.pipeline.local import Pipeline
from distilabel.steps.base import (
    GeneratorStep,
    GlobalStep,
    RuntimeParameter,
    Step,
    StepInput,
)
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
    def test_create_step_passing_pipeline(self) -> None:
        pipeline = Pipeline()
        step = DummyStep(name="dummy", pipeline=pipeline)
        assert step.pipeline == pipeline

    def test_create_step_within_pipeline_context(self) -> None:
        with Pipeline() as pipeline:
            step = DummyStep(name="dummy")

        assert step.pipeline == pipeline

    def test_creating_step_without_pipeline(self) -> None:
        with pytest.raises(ValueError, match="Step 'dummy' hasn't received a pipeline"):
            DummyStep(name="dummy")

    def test_is_generator(self) -> None:
        step = DummyStep(name="dummy", pipeline=Pipeline())
        assert not step.is_generator

    def test_is_global(self) -> None:
        step = DummyStep(name="dummy", pipeline=Pipeline())
        assert not step.is_global

    def test_runtime_parameters_names(self) -> None:
        class StepWithRuntimeParameters(Step):
            runtime_param1: RuntimeParameter[int]
            runtime_param2: RuntimeParameter[str] = "hello"
            runtime_param3: Optional[RuntimeParameter[str]] = None

            def process(self, *inputs: StepInput) -> StepOutput:
                yield []

        step = StepWithRuntimeParameters(name="dummy", pipeline=Pipeline())  # type: ignore

        assert step.runtime_parameters_names == {
            "runtime_param1": False,
            "runtime_param2": True,
            "runtime_param3": True,
        }

    def test_verify_inputs_mappings(self) -> None:
        step = DummyStep(name="dummy", pipeline=Pipeline())

        step.verify_inputs_mappings()

        step.input_mappings = {"im_not_an_input": "prompt"}
        with pytest.raises(
            ValueError, match="The input column 'im_not_an_input' doesn't exist"
        ):
            step.verify_inputs_mappings()

    def test_verify_outputs_mappings(self) -> None:
        step = DummyStep(name="dummy", pipeline=Pipeline())

        step.verify_outputs_mappings()

        step.output_mappings = {"im_not_an_output": "prompt"}
        with pytest.raises(
            ValueError, match="The output column 'im_not_an_output' doesn't exist"
        ):
            step.verify_outputs_mappings()

    def test_get_inputs(self) -> None:
        step = DummyStep(
            name="dummy", pipeline=Pipeline(), input_mappings={"instruction": "prompt"}
        )
        assert step.get_inputs() == ["prompt"]

    def test_get_outputs(self) -> None:
        step = DummyStep(
            name="dummy",
            pipeline=Pipeline(),
            output_mappings={"response": "generation"},
        )
        assert step.get_outputs() == ["generation"]

    def test_apply_input_mappings(self) -> None:
        step = DummyStep(
            name="dummy", pipeline=Pipeline(), input_mappings={"instruction": "prompt"}
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
            pipeline=Pipeline(),
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
            {"prompt": "hello 1", "response": "unit test"},
            {"prompt": "hello 2", "response": "unit test"},
            {"prompt": "hello 3", "response": "unit test"},
        ]


class TestGeneratorStep:
    def test_is_generator(self) -> None:
        step = DummyGeneratorStep(name="dummy", pipeline=Pipeline())
        assert step.is_generator

    def test_is_global(self) -> None:
        step = DummyGeneratorStep(name="dummy", pipeline=Pipeline())
        assert not step.is_global


class TestGlobalStep:
    def test_is_generator(self) -> None:
        step = DummyGlobalStep(name="dummy", pipeline=Pipeline())
        assert not step.is_generator

    def test_is_global(self) -> None:
        step = DummyGlobalStep(name="dummy", pipeline=Pipeline())
        assert step.is_global


class TestStepSerialization:
    def test_step_dump(self) -> None:
        pipeline = Pipeline()
        step = DummyStep(name="dummy", pipeline=pipeline)
        assert step.dump() == {
            "name": "dummy",
            "input_batch_size": 50,
            "input_mappings": {},
            "output_mappings": {},
            "runtime_parameters_info": [],
            TYPE_INFO_KEY: {
                "module": "tests.unit.steps.test_base",
                "name": "DummyStep",
            },
        }

    def test_step_from_dict(self) -> None:
        with Pipeline():
            step = DummyStep.from_dict(
                {
                    **{
                        "name": "dummy",
                        TYPE_INFO_KEY: {
                            "module": "tests.unit.pipeline.step.test_base",
                            "name": "DummyStep",
                        },
                    }
                }
            )

        assert isinstance(step, DummyStep)

    def test_step_from_dict_without_pipeline_context(self) -> None:
        with pytest.raises(ValueError):
            DummyStep.from_dict(
                {
                    **{
                        "name": "dummy",
                        TYPE_INFO_KEY: {
                            "module": "tests.pipeline.step.test_base",
                            "name": "DummyStep",
                        },
                    }
                }
            )
