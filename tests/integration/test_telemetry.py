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
from typing import TYPE_CHECKING, Dict, List

import pytest
from distilabel.llms.huggingface.transformers import TransformersLLM
from distilabel.llms.openai import OpenAILLM
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.pipeline.local import Pipeline
from distilabel.steps.base import Step, StepInput
from distilabel.steps.generators.huggingface import LoadDataFromHub
from distilabel.steps.tasks.text_generation import TextGeneration

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class RenameColumns(Step):
    rename_mappings: RuntimeParameter[Dict[str, str]] = None

    @property
    def inputs(self) -> List[str]:
        return []

    @property
    def outputs(self) -> List[str]:
        return list(self.rename_mappings.values())  # type: ignore

    def process(self, *inputs: StepInput) -> "StepOutput":
        outputs = []
        for input in inputs:
            outputs = []
            for item in input:
                outputs.append(
                    {self.rename_mappings.get(k, k): v for k, v in item.items()}  # type: ignore
                )
            yield outputs


def test_pipeline_telemetry(mock_telemetry) -> None:
    with Pipeline(name="integration-test-pipeline") as pipeline:
        load_hub_dataset = LoadDataFromHub(name="load_dataset")
        rename_columns = RenameColumns(name="rename_columns")
        load_hub_dataset.connect(rename_columns)

        os.environ["OPENAI_API_KEY"] = "sk-***"
        generate_response = TextGeneration(
            name="generate_response",
            llm=OpenAILLM(model="gpt-3.5-turbo"),
            output_mappings={"generation": "output"},
        )
        rename_columns.connect(generate_response)

        generate_response_mini = TextGeneration(
            name="generate_response_mini",
            llm=TransformersLLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            output_mappings={"generation": "output"},
        )
        rename_columns.connect(generate_response_mini)

    pipeline.run(use_fs_to_pass_data=False, use_cache=False)

    mock_telemetry.track_add_step_data.assert_called()
    mock_telemetry.track_add_edge_data.assert_called()
    mock_telemetry.track_process_batch_data.assert_called()
    mock_telemetry.track_run_data.assert_called()
    mock_telemetry._track_data.assert_called()


def test_pipeline_exception_telemetry(mock_telemetry) -> None:
    with Pipeline(name="integration-test-pipeline") as pipeline:
        load_hub_dataset = LoadDataFromHub(name="load_dataset")
        rename_columns = RenameColumns(name="rename_columns")
        load_hub_dataset.connect(rename_columns)

        os.environ["OPENAI_API_KEY"] = "sk-***"
        generate_response = TextGeneration(
            name="generate_response",
            llm=OpenAILLM(model="gpt-3.5-turbo"),
            output_mappings={"generation": "output"},
        )
        rename_columns.connect(generate_response)

        generate_response_mini = TextGeneration(
            name="generate_response_mini",
            llm=TransformersLLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            output_mappings={"generation": "output"},
        )
        rename_columns.connect(generate_response_mini)

    with pytest.raises:
        pipeline.run(
            use_fs_to_pass_data=False, use_cache=False, parameters={"mock": {"a": "b"}}
        )

    mock_telemetry.track_exception.assert_called()
    mock_telemetry._track_data.assert_called()
