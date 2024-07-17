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
from unittest.mock import patch

import argilla as rg
from distilabel.pipeline.local import Pipeline
from distilabel.steps.argilla.text_generation import TextGenerationToArgilla

MockDataset = rg.Dataset(
    name="dataset",
    settings=rg.Settings(
        fields=[
            rg.TextField(name="id", title="id"),  # type: ignore
            rg.TextField(name="instruction", title="instruction"),  # type: ignore
            rg.TextField(name="generation", title="generation"),  # type: ignore
        ],
        questions=[
            rg.LabelQuestion(  # type: ignore
                name="quality",
                title="What's the quality of the generation for the given instruction?",
                labels={"bad": "👎", "good": "👍"},  # type: ignore
            )
        ],
    ),
)


class TestTextGenerationToArgilla:
    def test_process(self) -> None:
        pipeline = Pipeline(name="unit-test-pipeline")
        step = TextGenerationToArgilla(
            name="step",
            api_url="https://example.com",
            api_key="api.key",  # type: ignore
            dataset_name="argilla",
            dataset_workspace="argilla",
            pipeline=pipeline,
        )
        with patch.object(TextGenerationToArgilla, "load"):
            step.load()
        step._instruction = "instruction"
        step._generation = "generation"
        step._dataset = MockDataset  # type: ignore

        step._dataset.records.log = lambda x: x
        assert list(step.process([{"instruction": "test", "generation": "test"}])) == [
            [{"instruction": "test", "generation": "test"}]
        ]
        assert step._dataset.records  # type: ignore

    def test_serialization(self) -> None:
        os.environ["ARGILLA_API_KEY"] = "api.key"

        pipeline = Pipeline(name="unit-test-pipeline")
        step = TextGenerationToArgilla(
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
                    "Defaults to `None` which means it will be created in the default "
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
                "module": "distilabel.steps.argilla.text_generation",
                "name": "TextGenerationToArgilla",
            },
        }

        with Pipeline(name="unit-test-pipeline") as pipeline:
            new_step = TextGenerationToArgilla.from_dict(step.dump())
            assert isinstance(new_step, TextGenerationToArgilla)
