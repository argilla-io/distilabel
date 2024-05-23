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

# from distilabel.distiset import Distiset
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.pipeline.local import Pipeline
from distilabel.steps.base import Step, StepInput
from distilabel.steps.generators.data import LoadDataFromDicts
from distilabel.steps.globals.answer_batch import AnswerBatch
from distilabel.steps.tasks.typing import ChatItem
from typing_extensions import override

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
    rename_mappings: RuntimeParameter[Dict[str, str]]

    @property
    def inputs(self) -> List[str]:
        return []

    @property
    def outputs(self) -> List[str]:
        return list(self.rename_mappings.values())  # type: ignore

    def process(self, inputs: StepInput) -> Generator[List[Dict[str, Any]], None, None]:
        outputs = []
        for input in inputs:
            outputs.append(
                {self.rename_mappings.get(k, k): v for k, v in input.items()}  # type: ignore
            )
        yield outputs


class AnswerWithBatchAPI(AnswerBatch):
    @override
    def format_input(self, input: Dict[str, Any]) -> List[ChatItem]:
        return [{"role": "user", "content": input["instruction"]}]

    @override
    def format_output(
        self, output: str | None, input: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "response": output,
        }

    @property
    def inputs(self) -> List[str]:
        return ["instruction"]

    @property
    def outputs(self) -> List[str]:
        return ["response"]


def run_pipeline():
    with Pipeline(name="unit-test-pipeline") as pipeline:
        load_dataset = LoadDataFromDicts(name="load_dataset", data=DATA, batch_size=8)
        rename_columns = RenameColumns(name="rename_columns", input_batch_size=12)
        generate_response = AnswerWithBatchAPI(
            name="generate_response",
            model_id="gpt-4o",
            batch_description="Testing OpenAI batch processing",
        )

        load_dataset.connect(rename_columns)
        rename_columns.connect(generate_response)

    return pipeline.run(
        parameters={
            "rename_columns": {
                "rename_mappings": {
                    "prompt": "instruction",
                },
            },
        }
    )


def test_pipeline_cached():
    ds = run_pipeline()
    print()
    print("----- RUNNING PIPELINE AGAIN -----")
    print()
    ds = run_pipeline()
    # assert isinstance(ds, Distiset)
    assert len(ds["default"]["train"]) == 80


if __name__ == "__main__":
    test_pipeline_cached()
