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

from typing import TYPE_CHECKING

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, StepInput, StepOutput, step
from distilabel.steps.base import GeneratorStep

if TYPE_CHECKING:
    from distilabel.steps import GeneratorStepOutput


class RuntimeParameterGenerator(GeneratorStep):
    greeting: RuntimeParameter[str] = "default"

    @property
    def outputs(self) -> list[str]:
        return ["instruction"]

    def process(self, offset: int = 0) -> "GeneratorStepOutput":
        yield ([{"instruction": self.greeting}] * self.batch_size, True)


@step(inputs=["instruction"], outputs=["response"])
def SucceedAlways(inputs: StepInput) -> "StepOutput":
    for input in inputs:
        input["response"] = "This step always succeeds"
    yield inputs


def test_dry_run():
    load_dataset_name = "load_dataset"

    def get_pipeline():
        with Pipeline(name="other-pipe") as pipeline:
            load_dataset = LoadDataFromDicts(
                name=load_dataset_name,
                data=[
                    {"instruction": "Tell me a joke."},
                ]
                * 50,
                batch_size=20,
            )
            text_generation = SucceedAlways()

            load_dataset >> text_generation
        return pipeline

    # Test with and without parameters
    pipeline = get_pipeline()
    distiset = pipeline.dry_run(batch_size=2)
    assert len(distiset["default"]["train"]) == 2
    assert pipeline._dry_run is False

    pipeline = get_pipeline()
    distiset = pipeline.dry_run(parameters={load_dataset_name: {"batch_size": 8}})
    assert len(distiset["default"]["train"]) == 1
    assert pipeline._dry_run is False

    pipeline = get_pipeline()
    distiset = pipeline.run(
        parameters={load_dataset_name: {"batch_size": 10}}, use_cache=False
    )
    assert len(distiset["default"]["train"]) == 50


def test_dry_run_preserves_generator_runtime_parameters():
    with Pipeline(name="runtime-parameter-dry-run") as pipeline:
        generator = RuntimeParameterGenerator(name="generator", batch_size=5)
        text_generation = SucceedAlways()

        generator >> text_generation

    distiset = pipeline.dry_run(
        parameters={"generator": {"greeting": "runtime override"}},
        batch_size=2,
    )

    assert len(distiset["default"]["train"]) == 2
    assert distiset["default"]["train"][0]["instruction"] == "runtime override"
    assert distiset["default"]["train"][1]["instruction"] == "runtime override"
    assert pipeline._dry_run is False
