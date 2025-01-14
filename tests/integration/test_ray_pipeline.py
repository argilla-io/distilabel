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

from typing import TYPE_CHECKING, Dict, List

import pytest

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.pipeline.ray import RayPipeline
from distilabel.steps.base import Step, StepInput
from distilabel.steps.generators.data import LoadDataFromDicts

if TYPE_CHECKING:
    from distilabel.typing import StepOutput

DATA = [
    {"prompt": "Tell me a joke"},
    {"prompt": "Write a short haiku"},
    {"prompt": "Translate 'My name is Alvaro' to Spanish"},
    {"prompt": "What's the capital of Spain?"},
    {"prompt": "Tell me a joke"},
    {"prompt": "Write a short haiku"},
    {"prompt": "Translate 'My name is Alvaro' to Spanish"},
    {"prompt": "What's the capital of Spain?"},
    {"prompt": "Tell me a joke"},
    {"prompt": "Write a short haiku"},
    {"prompt": "Translate 'My name is Alvaro' to Spanish"},
    {"prompt": "What's the capital of Spain?"},
    {"prompt": "Tell me a joke"},
    {"prompt": "Write a short haiku"},
    {"prompt": "Translate 'My name is Alvaro' to Spanish"},
    {"prompt": "What's the capital of Spain?"},
    {"prompt": "Tell me a joke"},
    {"prompt": "Write a short haiku"},
    {"prompt": "Translate 'My name is Alvaro' to Spanish"},
    {"prompt": "What's the capital of Spain?"},
    {"prompt": "Tell me a joke"},
    {"prompt": "Write a short haiku"},
    {"prompt": "Translate 'My name is Alvaro' to Spanish"},
    {"prompt": "What's the capital of Spain?"},
    {"prompt": "Tell me a joke"},
    {"prompt": "Write a short haiku"},
    {"prompt": "Translate 'My name is Alvaro' to Spanish"},
    {"prompt": "What's the capital of Spain?"},
    {"prompt": "Tell me a joke"},
    {"prompt": "Write a short haiku"},
    {"prompt": "Translate 'My name is Alvaro' to Spanish"},
    {"prompt": "What's the capital of Spain?"},
    {"prompt": "Tell me a joke"},
    {"prompt": "Write a short haiku"},
    {"prompt": "Translate 'My name is Alvaro' to Spanish"},
    {"prompt": "What's the capital of Spain?"},
    {"prompt": "Tell me a joke"},
    {"prompt": "Write a short haiku"},
    {"prompt": "Translate 'My name is Alvaro' to Spanish"},
    {"prompt": "What's the capital of Spain?"},
    {"prompt": "Tell me a joke"},
    {"prompt": "Write a short haiku"},
    {"prompt": "Translate 'My name is Alvaro' to Spanish"},
    {"prompt": "What's the capital of Spain?"},
    {"prompt": "Tell me a joke"},
    {"prompt": "Write a short haiku"},
    {"prompt": "Translate 'My name is Alvaro' to Spanish"},
    {"prompt": "What's the capital of Spain?"},
    {"prompt": "Tell me a joke"},
    {"prompt": "Write a short haiku"},
    {"prompt": "Translate 'My name is Alvaro' to Spanish"},
    {"prompt": "What's the capital of Spain?"},
    {"prompt": "Tell me a joke"},
    {"prompt": "Write a short haiku"},
    {"prompt": "Translate 'My name is Alvaro' to Spanish"},
    {"prompt": "What's the capital of Spain?"},
    {"prompt": "Tell me a joke"},
    {"prompt": "Write a short haiku"},
    {"prompt": "Translate 'My name is Alvaro' to Spanish"},
    {"prompt": "What's the capital of Spain?"},
    {"prompt": "Tell me a joke"},
    {"prompt": "Write a short haiku"},
    {"prompt": "Translate 'My name is Alvaro' to Spanish"},
    {"prompt": "What's the capital of Spain?"},
    {"prompt": "Tell me a joke"},
    {"prompt": "Write a short haiku"},
    {"prompt": "Translate 'My name is Alvaro' to Spanish"},
    {"prompt": "What's the capital of Spain?"},
    {"prompt": "Tell me a joke"},
    {"prompt": "Write a short haiku"},
    {"prompt": "Translate 'My name is Alvaro' to Spanish"},
    {"prompt": "What's the capital of Spain?"},
    {"prompt": "Tell me a joke"},
    {"prompt": "Write a short haiku"},
    {"prompt": "Translate 'My name is Alvaro' to Spanish"},
    {"prompt": "What's the capital of Spain?"},
    {"prompt": "Tell me a joke"},
    {"prompt": "Write a short haiku"},
    {"prompt": "Translate 'My name is Alvaro' to Spanish"},
    {"prompt": "What's the capital of Spain?"},
]


class RenameColumns(Step):
    rename_mappings: RuntimeParameter[Dict[str, str]] = None

    @property
    def inputs(self) -> List[str]:
        return []

    @property
    def outputs(self) -> List[str]:
        return list(self.rename_mappings.values())  # type: ignore

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        outputs = []
        for input in inputs:
            outputs.append(
                {self.rename_mappings.get(k, k): v for k, v in input.items()}  # type: ignore
            )
        yield outputs


class GenerateResponse(Step):
    @property
    def inputs(self) -> List[str]:
        return ["instruction"]

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        import time

        time.sleep(1)

        for input in inputs:
            input["response"] = "I don't know"

        yield inputs

    @property
    def outputs(self) -> List[str]:
        return ["response"]


@pytest.mark.skip_python_versions(["3.12"])
def test_run_pipeline() -> None:
    import ray
    from ray.cluster_utils import Cluster

    # TODO: if we add more tests, this should be a fixture
    cluster = Cluster(initialize_head=True, head_node_args={"num_cpus": 10})
    ray.init(address=cluster.address)

    with RayPipeline(
        name="unit-test-pipeline", ray_init_kwargs={"ignore_reinit_error": True}
    ) as pipeline:
        load_dataset = LoadDataFromDicts(name="load_dataset", data=DATA, batch_size=8)
        rename_columns = RenameColumns(name="rename_columns", input_batch_size=12)
        generate_response = GenerateResponse(
            name="generate_response", input_batch_size=16
        )

        load_dataset >> rename_columns >> generate_response

    distiset = pipeline.run(
        parameters={
            "rename_columns": {
                "rename_mappings": {
                    "prompt": "instruction",
                },
            },
        }
    )

    assert len(distiset["default"]["train"]) == 80
