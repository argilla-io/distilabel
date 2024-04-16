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

import os
from typing import TYPE_CHECKING, List

import pytest
from distilabel.pipeline.local import Pipeline
from distilabel.steps.argilla.base import Argilla
from distilabel.steps.base import StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class CustomArgilla(Argilla):
    def load(self) -> None:
        pass

    @property
    def inputs(self) -> List[str]:
        return ["instruction"]

    def process(self, *inputs: StepInput) -> "StepOutput":
        yield [{}]


class TestArgilla:
    def test_passing_pipeline(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        step = CustomArgilla(
            name="step",
            api_url="https://example.com",
            api_key="api.key",  # type: ignore
            dataset_name="argilla",
            dataset_workspace="argilla",
            pipeline=pipeline,
        )
        assert step.name == "step"
        assert step.api_url == "https://example.com"
        assert step.api_key.get_secret_value() == "api.key"  # type: ignore
        assert step.dataset_name == "argilla"
        assert step.dataset_workspace == "argilla"
        assert step.pipeline is pipeline

    def test_within_pipeline_context(self) -> None:
        with Pipeline(name="unit-test-pipeline") as pipeline:
            step = CustomArgilla(
                name="step",
                api_url="https://example.com",
                api_key="api.key",  # type: ignore
                dataset_name="argilla",
                dataset_workspace="argilla",
                pipeline=pipeline,
            )
            assert step.name == "step"
            assert step.api_url == "https://example.com"
            assert step.api_key.get_secret_value() == "api.key"  # type: ignore
            assert step.dataset_name == "argilla"
            assert step.dataset_workspace == "argilla"
        assert step.pipeline is pipeline

    def test_with_errors(self) -> None:
        with pytest.raises(ValueError, match="Step 'step' hasn't received a pipeline"):
            CustomArgilla(
                name="step",
                api_url="https://example.com",
                api_key="api.key",  # type: ignore
                dataset_name="argilla",
                dataset_workspace="argilla",
            )

        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class Argilla with abstract methods inputs, process",
        ):
            Argilla(name="step", pipeline=Pipeline(name="unit-test-pipeline"))  # type: ignore

    def test_process(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        step = CustomArgilla(
            name="step",
            api_url="https://example.com",
            api_key="api.key",  # type: ignore
            dataset_name="argilla",
            dataset_workspace="argilla",
            pipeline=pipeline,
        )
        assert list(step.process([{"instruction": "test"}])) == [[{}]]

    def test_serialization(self) -> None:
        os.environ["ARGILLA_API_KEY"] = "api.key"

        pipeline = Pipeline(name="unit-test-pipeline")
        step = CustomArgilla(
            name="step",
            api_url="https://example.com",
            dataset_name="argilla",
            dataset_workspace="argilla",
            pipeline=pipeline,
        )
        assert step.dump() == {
            "name": "step",
            "input_mappings": {},
            "output_mappings": {},
            "input_batch_size": 50,
            "dataset_name": "argilla",
            "dataset_workspace": "argilla",
            "api_url": "https://example.com",
            "runtime_parameters_info": [
                {
                    "description": "The number of rows that will contain the batches processed by the step.",
                    "name": "input_batch_size",
                    "optional": True,
                },
                {
                    "description": "The name of the dataset in Argilla.",
                    "name": "dataset_name",
                    "optional": False,
                },
                {
                    "description": "The workspace where the dataset will be created in Argilla. "
                    "Defaultsto `None` which means it will be created in the default "
                    "workspace.",
                    "name": "dataset_workspace",
                    "optional": True,
                },
                {
                    "name": "api_url",
                    "optional": True,
                    "description": "The base URL to use for the Argilla API requests.",
                },
                {
                    "name": "api_key",
                    "optional": True,
                    "description": "The API key to authenticate the requests to the Argilla API.",
                },
            ],
            "type_info": {
                "module": "tests.unit.steps.argilla.test_base",
                "name": "CustomArgilla",
            },
        }

        with Pipeline(name="unit-test-pipeline") as pipeline:
            new_step = CustomArgilla.from_dict(step.dump())
            assert isinstance(new_step, CustomArgilla)
