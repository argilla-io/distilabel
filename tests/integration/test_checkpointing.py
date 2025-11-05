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

import pytest
from datasets import Dataset

from distilabel.pipeline import Pipeline
from distilabel.steps import HuggingFaceHubCheckpointer
from distilabel.steps.base import Step, StepInput

dataset = Dataset.from_dict({"a": [1, 2] * 50, "b": [5, 6] * 50})


if TYPE_CHECKING:
    from distilabel.typing import StepOutput


class DoNothing(Step):
    def process(self, *inputs: StepInput) -> "StepOutput":
        for input in inputs:
            yield input


@pytest.mark.skip(reason="Currently cannot obtain the correct HF_TOKEN from the CI")
def test_checkpointing() -> None:
    with Pipeline(name="simple-text-generation-pipeline") as pipeline:
        text_generation = DoNothing(input_batch_size=60)
        checkpoint = HuggingFaceHubCheckpointer(
            repo_id="distilabel-internal-testing/__streaming_test_1",
            private=False,
            input_batch_size=50,
        )
        text_generation >> checkpoint
    pipeline.run(dataset=dataset, use_cache=False)

    from huggingface_hub import HfFileSystem

    dataset_name = "distilabel-internal-testing/__streaming_test_1"
    fs = HfFileSystem()
    filenames = fs.glob(f"datasets/{dataset_name}/**/*.jsonl")
    assert len(filenames) == 2


if __name__ == "__main__":
    test_checkpointing()
