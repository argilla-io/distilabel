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

import pytest

from distilabel.pipeline import Pipeline
from distilabel.steps import Step
from distilabel.steps.base import StepInput
from distilabel.steps.typing import StepOutput
from distilabel.utils.requirements import requirements

from ..pipeline.utils import DummyGeneratorStep


def test_add_requirements_decorator():
    @requirements(["distilabel>=0.0.1"])
    class CustomStep(Step):
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

    assert CustomStep.requirements == ["distilabel>=0.0.1"]


@pytest.mark.parametrize(
    "requirements_pipeline, expected",
    [
        ([], ["distilabel>=0.0.1", "numpy"]),
        (["candle_holder"], ["candle_holder", "distilabel>=0.0.1", "numpy"]),
    ],
)
def test_add_requirements_to_pipeline(
    requirements_pipeline: List[str], expected: List[str]
) -> None:
    # Check the pipeline has the requirements from the steps defined within it.

    @requirements(["distilabel>=0.0.1"])
    class CustomStep(Step):
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

    @requirements(["numpy"])
    class OtherStep(Step):
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

    with Pipeline(
        name="unit-test-pipeline", requirements=requirements_pipeline
    ) as pipeline:
        generator = DummyGeneratorStep()
        step = CustomStep()
        global_step = OtherStep()

        generator >> [step, global_step]
    print("REQS", pipeline.requirements)
    print("REQS_PRIVATE", pipeline._requirements)
    assert pipeline.requirements == expected


def test_requirements_on_step_decorator() -> None:
    from distilabel.mixins.runtime_parameters import RuntimeParameter
    from distilabel.steps.decorator import step

    @requirements(["distilabel>=0.0.1"])
    @step(inputs=["instruction"], outputs=["generation"])
    def UnitTestStep(
        inputs: StepInput,
        runtime_param1: RuntimeParameter[int],
        runtime_param2: RuntimeParameter[float] = 5.0,
    ) -> StepOutput:
        """A dummy step for the unit test"""
        yield []

    assert UnitTestStep.requirements == ["distilabel>=0.0.1"]
