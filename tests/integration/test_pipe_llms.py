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
from typing import TYPE_CHECKING, Any, Dict

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.models.llms.huggingface.transformers import TransformersLLM
from distilabel.models.llms.openai import OpenAILLM
from distilabel.pipeline.local import Pipeline
from distilabel.steps.base import Step, StepInput
from distilabel.steps.generators.huggingface import LoadDataFromHub
from distilabel.steps.tasks.text_generation import TextGeneration

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class RenameColumns(Step):
    rename_mappings: RuntimeParameter[Dict[str, str]] = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.outputs = list(self.rename_mappings.values())  # type: ignore

    def process(self, *inputs: StepInput) -> "StepOutput":
        outputs = []
        for input in inputs:
            outputs = []
            for item in input:
                outputs.append(
                    {self.rename_mappings.get(k, k): v for k, v in item.items()}  # type: ignore
                )
            yield outputs


def test_pipeline_with_llms_serde() -> None:
    with Pipeline(name="unit-test-pipeline") as pipeline:
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
        dump = pipeline.dump()

    with Pipeline(name="unit-test-pipeline") as pipe:
        pipe = pipe.from_dict(dump)

    assert "load_dataset" in pipe.dag.G
    assert "rename_columns" in pipe.dag.G
    assert "generate_response" in pipe.dag.G
    assert "generate_response_mini" in pipe.dag.G
