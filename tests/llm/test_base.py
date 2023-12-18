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

from typing import TYPE_CHECKING, Any, Dict, List, Set

import pytest
from distilabel.llm.base import LLM, LLMPool, ProcessLLM
from distilabel.llm.utils import LLMOutput
from distilabel.tasks.preference.ultrafeedback import UltraFeedbackTask
from distilabel.tasks.text_generation.base import TextGenerationTask

if TYPE_CHECKING:
    pass


class DummyLLM(LLM):
    @property
    def model_name(self) -> str:
        return "dummy"

    def _generate(
        self, inputs: List[Dict[str, Any]], num_generations: int = 1
    ) -> List[List["LLMOutput"]]:
        outputs = []
        for _ in range(len(inputs)):
            row_outputs = []
            for _ in range(num_generations):
                row_outputs.append(
                    LLMOutput(
                        model_name=self.model_name,
                        prompt_used="dummy",
                        raw_output="dummy",
                        parsed_output="dummy",
                    )
                )
        return outputs


def test_llmpool_errors_if_llms_less_than_two() -> None:
    with pytest.raises(ValueError, match="The `llms` argument must contain at least 2"):
        LLMPool(llms=[None])  # type: ignore


def test_llmpool_errors_if_llm_not_instance_of_processllm() -> None:
    with pytest.raises(
        ValueError, match="The `llms` argument must contain only `ProcessLLM`s."
    ):
        LLMPool(llms=[None, None])  # type: ignore


def test_llmpool_errors_if_llms_do_not_have_same_task() -> None:
    llm1 = ProcessLLM(
        task=TextGenerationTask(), load_llm_fn=lambda task: DummyLLM(task=task)
    )
    llm2 = ProcessLLM(
        task=UltraFeedbackTask.for_honesty(),
        load_llm_fn=lambda task: DummyLLM(task=task),
    )
    with pytest.raises(
        ValueError,
        match="The `llms` argument must contain `ProcessLLM`s with the same task.",
    ):
        LLMPool(llms=[llm1, llm2])


@pytest.mark.parametrize(
    "num_generations, num_llms, expected", [(2, 4, {0, 1}), (4, 4, {1}), (9, 4, {2, 3})]
)
def test_llmpool_get_num_generations_per_llm(
    num_generations: int, num_llms: int, expected: Set[int]
) -> None:
    llms = []
    for _ in range(num_llms):
        llms.append(
            ProcessLLM(
                task=TextGenerationTask(), load_llm_fn=lambda task: DummyLLM(task=task)
            )
        )

    pool = LLMPool(llms=llms)

    num_generations_per_llm = pool._get_num_generations_per_llm(
        num_generations=num_generations
    )

    assert set(num_generations_per_llm.values()) == expected
