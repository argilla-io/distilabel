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

from pathlib import Path
from typing import Any, Dict, Generator, List

from distilabel.pipeline.local import Pipeline
from distilabel.pipeline.step.base import RuntimeParameter, Step
from distilabel.pipeline.step.generators.huggingface import LoadHubDataset
from distilabel.pipeline.step.typing import StepInput


class RenameColumns(Step):
    rename_mappings: RuntimeParameter[Dict[str, str]]

    @property
    def inputs(self) -> List[str]:
        return []

    @property
    def outputs(self) -> List[str]:
        return list(self.rename_mappings.values())

    def process(self, inputs: StepInput) -> Generator[List[Dict[str, Any]], None, None]:
        outputs = []
        for input in inputs:
            outputs.append(
                {self.rename_mappings.get(k, k): v for k, v in input.items()}
            )
        yield outputs


class GenerateResponse(Step):
    @property
    def inputs(self) -> List[str]:
        return ["instruction"]

    def process(self, inputs: StepInput) -> Generator[List[Dict[str, Any]], None, None]:
        import time

        for input in inputs:
            time.sleep(0.8)
            print("*** processing ***")
            input["response"] = "I don't know"
        yield inputs

    @property
    def outputs(self) -> List[str]:
        return ["response"]


def test_pipeline_from_dump():
    with Pipeline() as pipeline:
        load_hub_dataset = LoadHubDataset(name="load_dataset")
        rename_columns = RenameColumns(name="rename_columns")
        generate_response = GenerateResponse(name="generate_response")

        load_hub_dataset.connect(rename_columns)
        rename_columns.connect(generate_response)
        dump = pipeline.dump()

    # Recreate the pipeline from the dump
    with Pipeline() as pipe:
        pipe = pipe.from_dict(dump)

    pipe.run(
        parameters={
            "load_dataset": {
                "repo_id": "alvarobartt/test",
                "split": "train",
            },
            "rename_columns": {
                "rename_mappings": {
                    "prompt": "instruction",
                }
            },
        }
    )
    data_buffered_file = pipe._cache_dir / "data.jsonl"
    assert Path(data_buffered_file).exists()
    data_buffered_file.unlink()


def test_pipeline_cached():
    with Pipeline() as pipeline:
        load_hub_dataset = LoadHubDataset(name="load_dataset")
        rename_columns = RenameColumns(name="rename_columns")
        generate_response = GenerateResponse(name="generate_response")

        load_hub_dataset.connect(rename_columns)
        rename_columns.connect(generate_response)

        pipeline.run(
            parameters={
                "load_dataset": {
                    "repo_id": "alvarobartt/test",
                    "split": "train",
                },
                "rename_columns": {
                    "rename_mappings": {
                        "prompt": "instruction",
                    }
                },
            }
        )

    # Recreate the pipeline from the dump
    with Pipeline() as pipe:
        load_hub_dataset = LoadHubDataset(name="load_dataset")
        rename_columns = RenameColumns(name="rename_columns")
        generate_response = GenerateResponse(name="generate_response")

        load_hub_dataset.connect(rename_columns)
        rename_columns.connect(generate_response)
        assert pipe._cache_filename.exists()

        pipe.run(
            parameters={
                "load_dataset": {
                    "repo_id": "alvarobartt/test",
                    "split": "train",
                },
                "rename_columns": {
                    "rename_mappings": {
                        "prompt": "instruction",
                    }
                },
            }
        )
    # TODO: Add checks on the steps from the pipeline at different stages.


if __name__ == "__main__":
    with Pipeline() as pipeline:
        load_hub_dataset = LoadHubDataset(name="load_dataset")
        rename_columns = RenameColumns(name="rename_columns")
        generate_response = GenerateResponse(name="generate_response")

        load_hub_dataset.connect(rename_columns)
        rename_columns.connect(generate_response)

        pipeline.run(
            parameters={
                "load_dataset": {
                    "repo_id": "alvarobartt/test",
                    "split": "train",
                },
                "rename_columns": {
                    "rename_mappings": {
                        "prompt": "instruction",
                    }
                },
            }
        )

    # Recreate the pipeline from the dump
    with Pipeline() as pipe:
        load_hub_dataset = LoadHubDataset(name="load_dataset")
        rename_columns = RenameColumns(name="rename_columns")
        generate_response = GenerateResponse(name="generate_response")

        load_hub_dataset.connect(rename_columns)
        rename_columns.connect(generate_response)
        assert pipe._cache_filename.exists()

        pipe.run(
            parameters={
                "load_dataset": {
                    "repo_id": "alvarobartt/test",
                    "split": "train",
                },
                "rename_columns": {
                    "rename_mappings": {
                        "prompt": "instruction",
                    }
                },
            }
        )
