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

import re
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest
from distilabel.llm.base import LLM, LLMPool, ProcessLLM
from distilabel.llm.utils import LLMOutput
from distilabel.tasks.base import Task
from distilabel.tasks.preference.judgelm import JudgeLMTask
from distilabel.tasks.preference.ultrafeedback import UltraFeedbackTask
from distilabel.tasks.prompt import Prompt
from distilabel.tasks.text_generation.base import TextGenerationTask
from distilabel.tasks.text_generation.self_instruct import SelfInstructTask


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


class DummySubtask(TextGenerationTask):
    system_prompt: str = "You are a helpful assistant."

    def generate_prompt(self, input: str) -> "Prompt":
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt="Instruction {instruction}\nResponse".format(
                instruction=input
            ),
        )


class SubtaskOne(TextGenerationTask):
    pass


class SubtaskTwo(TextGenerationTask):
    pass


class TestLLM:
    def test_get_valid_inputs(self) -> None:
        llm = DummyLLM(task=TextGenerationTask())

        inputs = [
            {"input": "I'm valid for text generation task"},
            {"random": "I'm not valid"},
        ]
        valid_inputs, invalid_inputs_indices = llm._get_valid_inputs(inputs=inputs)
        assert valid_inputs == [{"input": "I'm valid for text generation task"}]
        assert invalid_inputs_indices == [1]

    def test_fill_missing_inputs(self) -> None:
        llm = DummyLLM(task=TextGenerationTask())

        generations = [
            [
                LLMOutput(
                    model_name=llm.model_name,
                    prompt_used=llm.task.generate_prompt(
                        input="I'm valid for text generation task"
                    ).format_as("default"),
                    raw_output="dummy",
                    parsed_output="dummy",
                ),
                LLMOutput(
                    model_name=llm.model_name,
                    prompt_used=llm.task.generate_prompt(
                        input="I'm valid too"
                    ).format_as("default"),
                    raw_output="dummy",
                    parsed_output="dummy",
                ),
            ]
        ]

        filled_generations = llm._fill_missing_inputs(
            generations=generations,
            invalid_inputs_indices=[1],
            num_generations=2,
        )

        assert filled_generations == generations + [
            [
                LLMOutput(
                    model_name=llm.model_name,
                    prompt_used=None,
                    raw_output=None,
                    parsed_output=None,
                )
                for _ in range(2)
            ]
        ]


class TestLLMPool:
    def test_llmpool_errors_if_llms_less_than_two(self) -> None:
        with pytest.raises(
            ValueError, match="The `llms` argument must contain at least 2"
        ):
            LLMPool(llms=[None])  # type: ignore

    def test_llmpool_errors_if_llm_not_instance_of_processllm(self) -> None:
        with pytest.raises(
            ValueError, match="The `llms` argument must contain only `ProcessLLM`s."
        ):
            LLMPool(llms=[None, None])  # type: ignore

    @pytest.mark.parametrize(
        "tasks",
        [
            (TextGenerationTask(), TextGenerationTask()),
            (TextGenerationTask(), DummySubtask()),
            (TextGenerationTask(), TextGenerationTask(), DummySubtask()),
            (TextGenerationTask(), DummySubtask(), DummySubtask()),
            (SubtaskOne(), SubtaskOne(), SubtaskTwo()),
        ],
    )
    def test_llmpool_with_subclass_of_tasks(self, tasks: Tuple[Task]) -> None:
        LLMPool(
            llms=[
                ProcessLLM(task=t, load_llm_fn=lambda task: DummyLLM(task=task))
                for t in tasks
            ]
        )

    def test_llmpool_errors_if_llms_do_not_have_same_task(self) -> None:
        llm1 = ProcessLLM(
            task=TextGenerationTask(), load_llm_fn=lambda task: DummyLLM(task=task)
        )
        llm2 = ProcessLLM(
            task=UltraFeedbackTask.for_honesty(),
            load_llm_fn=lambda task: DummyLLM(task=task),
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "All the `ProcessLLM` in `llms` must share the same task (either as the instance or the parent class)."
            ),
        ):
            LLMPool(llms=[llm1, llm2])

    @pytest.mark.parametrize(
        "num_generations, num_llms, expected",
        [(2, 4, {0, 1}), (4, 4, {1}), (9, 4, {2, 3})],
    )
    def test_llmpool_get_num_generations_per_llm(
        self, num_generations: int, num_llms: int, expected: Set[int]
    ) -> None:
        llms = []
        for _ in range(num_llms):
            llms.append(
                ProcessLLM(
                    task=TextGenerationTask(),
                    load_llm_fn=lambda task: DummyLLM(task=task),
                )
            )

        pool = LLMPool(llms=llms)

        num_generations_per_llm = pool._get_num_generations_per_llm(
            num_generations=num_generations
        )

        assert set(num_generations_per_llm.values()) == expected


@pytest.mark.parametrize(
    "task,prompt_format,input,expected",
    [
        (
            TextGenerationTask(),
            None,
            {"input": "Your input"},
            "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\nYour input",
        ),
        (
            TextGenerationTask(),
            "notus",
            {"input": "Your input"},
            "<|system|>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.</s>\n<|user|>\nYour input</s>\n<|assistant|>\n",
        ),
        (
            SelfInstructTask(),
            None,
            {"input": "Your input"},
            'You are an expert prompt writer, writing the best and most diverse prompts for a variety of tasks. You are given a task description and a set of instructions for how to write the prompts for an specific AI application.\n# Task Description\nDevelop 5 user queries that can be received by the given AI application and applicable to the provided context. Emphasize diversity in verbs and linguistic structures within the model\'s textual capabilities.\n\n# Criteria for Queries\nIncorporate a diverse range of verbs, avoiding repetition.\nEnsure queries are compatible with AI model\'s text generation functions and are limited to 1-2 sentences.\nDesign queries to be self-contained and standalone.\nBlend interrogative (e.g., "What is the significance of x?") and imperative (e.g., "Detail the process of x.") styles.\nWrite each query on a separate line and avoid using numbered lists or bullet points.\n\n# AI Application\nAI assistant\n\n# Context\nYour input\n\n# Output\n',
        ),
        (
            JudgeLMTask(),
            None,
            {"input": "Your input", "generations": ["Your output"]},
            "You are a helpful and precise assistant for checking the quality of the answer.\n[Question]\nYour input\n\n\n[The Start of Assistant 1's Answer>\nYour output\n[The End of Assistant 1's Answer>\n\n[System]\nWe would like to request your feedback on the performance of 1 AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only 1 values indicating the scores for Assistants 1 to 1, respectively. The 1 scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
        ),
    ],
)
def test_validate_prompts(
    task: Task, prompt_format: Optional[str], input: Dict[str, Any], expected: str
) -> None:
    llm = DummyLLM(task=task, prompt_format=prompt_format)
    result = llm.validate_prompts([input])

    assert result[0] == expected
