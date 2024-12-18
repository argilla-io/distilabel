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

from typing import TYPE_CHECKING, Any

from distilabel.models.llms.base import AsyncLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import CombineOutputs, LoadDataFromDicts
from distilabel.steps.generators.data_sampler import DataSampler
from distilabel.steps.tasks import TextGeneration

if TYPE_CHECKING:
    from distilabel.typing import FormattedInput, GenerateOutput


class DummyAsyncLLM(AsyncLLM):
    structured_output: Any = None

    def load(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return "test"

    async def agenerate(  # type: ignore
        self, input: "FormattedInput", num_generations: int = 1
    ) -> "GenerateOutput":
        return {
            "generations": ["output" for _ in range(num_generations)],
            "statistics": {},
        }


def get_pipeline():
    with Pipeline() as pipe:
        size_dataset_1 = 10
        loader_1 = LoadDataFromDicts(
            data=[{"instruction": f"instruction {i}"} for i in range(size_dataset_1)]
        )
        sampler = DataSampler(
            data=[{"sample": f"sample {i}"} for i in range(30)],
            size=2,
            samples=size_dataset_1,
            batch_size=8,
        )
        text_generation = TextGeneration(llm=DummyAsyncLLM(), input_batch_size=8)

        combine = CombineOutputs()
        [loader_1, sampler] >> combine >> text_generation
    return pipe


def test_sampler():
    pipe = get_pipeline()
    distiset = pipe.run(use_cache=False)
    assert len(distiset["default"]["train"]) == 10
    row = distiset["default"]["train"][0]
    assert isinstance(row["sample"], list)
    assert len(row["sample"]) == 2
    assert isinstance(row["instruction"], str)


if __name__ == "__main__":
    pipe = get_pipeline()
    distiset = pipe.run(use_cache=False)
    print(distiset)
    print(distiset["default"]["train"][0])
