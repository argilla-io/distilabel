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

from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, List, Union

from distilabel.exceptions import DistilabelOfflineBatchGenerationNotFinishedException
from distilabel.models.llms import LLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration

if TYPE_CHECKING:
    from distilabel.models.llms.typing import GenerateOutput
    from distilabel.steps.tasks.typing import FormattedInput


class DummyOfflineBatchGenerateLLM(LLM):
    def load(self) -> None:
        super().load()

    @property
    def model_name(self) -> str:
        return "test"

    def generate(  # type: ignore
        self, inputs: "FormattedInput", num_generations: int = 1
    ) -> "GenerateOutput":
        return ["output" for _ in range(num_generations)]

    def offline_batch_generate(
        self,
        inputs: Union[List["FormattedInput"], None] = None,
        num_generations: int = 1,
        **kwargs: Any,
    ) -> List["GenerateOutput"]:
        # Simulate that the first time we create the jobs
        if not self.jobs_ids:
            self.jobs_ids = ("1234", "5678")
            raise DistilabelOfflineBatchGenerationNotFinishedException(
                jobs_ids=self.jobs_ids  # type: ignore
            )
        return [
            {
                "generations": [f"output {i}" for i in range(num_generations)],
                "statistics": {
                    "input_tokens": [12] * num_generations,
                    "output_tokens": [12] * num_generations,
                },
            }
        ] * len(inputs)


def test_offline_batch_generation() -> None:
    with TemporaryDirectory() as tmp_dir:
        with Pipeline(cache_dir=tmp_dir) as pipeline:
            load_data = LoadDataFromDicts(
                data=[{"instruction": f"{i} instruction"} for i in range(100)]
            )

            text_generation = TextGeneration(
                llm=DummyOfflineBatchGenerateLLM(use_offline_batch_generation=True)
            )

            load_data >> text_generation

        distiset = pipeline.run()

        # First call no results
        assert len(distiset) == 0

        distiset = pipeline.run(use_cache=True)
        assert len(distiset) == 1
