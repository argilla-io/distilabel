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

from typing import Any, Dict, Generator, List

import pytest
from distilabel.pipeline.local import Pipeline
from distilabel.pipeline.step.base import GeneratorStep, GlobalStep, Step, StepInput


class DummyStep(Step):
    def inputs(self) -> List[str]:
        return []

    def outputs(self) -> List[str]:
        return []

    def process(self, inputs: StepInput) -> Generator[List[Dict[str, Any]], None, None]:
        yield []


class DummyGeneratorStep(GeneratorStep):
    def outputs(self) -> List[str]:
        return []

    def process(self, inputs: StepInput) -> Generator[List[Dict[str, Any]], None, None]:
        yield []


class DummyGlobalStep(GlobalStep):
    def inputs(self) -> List[str]:
        return []

    def outputs(self) -> List[str]:
        return []

    def process(self, inputs: StepInput) -> Generator[List[Dict[str, Any]], None, None]:
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
    def test_step_dump(self):
        pipeline = Pipeline()
        step = DummyStep(name="dummy", pipeline=pipeline)
        assert step.dump() == {
            "name": "dummy",
            "_type_info_": {
                "module": "tests.pipeline.step.test_base",
                "name": "DummyStep",
            },
        }

    def test_step_from_dict(self):
        pipeline = Pipeline()
        assert isinstance(
            DummyStep.from_dict(
                {
                    **{
                        "name": "dummy",
                        "_type_info_": {
                            "module": "tests.pipeline.step.test_base",
                            "name": "DummyStep",
                        },
                    },
                    **pipeline.dump(),
                }
            ),
            DummyStep,
        )
