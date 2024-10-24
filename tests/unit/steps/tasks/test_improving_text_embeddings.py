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

import json
from typing import Any, List

import pytest

from distilabel.llms import LLM
from distilabel.llms.typing import GenerateOutput
from distilabel.pipeline.local import Pipeline
from distilabel.steps.tasks.improving_text_embeddings import (
    BitextRetrievalGenerator,
    EmbeddingTaskGenerator,
    GenerateLongTextMatchingData,
    GenerateShortTextMatchingData,
    GenerateTextClassificationData,
    GenerateTextRetrievalData,
    MonolingualTripletGenerator,
)
from distilabel.steps.tasks.typing import ChatType


class MockLLM(LLM):
    output: str

    def load(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return "test"

    def generate(  # type: ignore
        self, inputs: List[ChatType], num_generations: int = 1
    ) -> List[GenerateOutput]:
        return [
            {
                "generations": [self.output for _ in range(num_generations)],
                "statistics": {
                    "input_tokens": [12] * num_generations,
                    "output_tokens": [12] * num_generations,
                },
            }
        ] * len(inputs)


class TestEmbeddingTaskGenerator:
    @pytest.mark.parametrize(
        "category",
        [
            "text-retrieval",
            "text-matching-short",
            "text-matching-long",
            "text-classification",
        ],
    )
    @pytest.mark.parametrize("flatten_tasks", [True, False])
    def test_process(self, category: str, flatten_tasks: bool) -> None:
        task = EmbeddingTaskGenerator(
            name="embedding_task_generator",
            category=category,  # type: ignore
            flatten_tasks=flatten_tasks,
            add_raw_output=False,
            llm=MockLLM(output="[ 'A', 'B', 'C' ]"),
            pipeline=Pipeline(name="unit-test-pipeline"),
            add_raw_input=False,
        )
        task.load()

        assert task.outputs == ["tasks" if not flatten_tasks else "task", "model_name"]

        result = (
            (
                [
                    {
                        "tasks": ["A", "B", "C"],
                        "model_name": "test",
                        "distilabel_metadata": {
                            "statistics_embedding_task_generator": {
                                "input_tokens": 12,
                                "output_tokens": 12,
                            }
                        },
                    }
                ],
                True,
            )
            if not flatten_tasks
            else (
                [
                    {
                        "task": "A",
                        "model_name": "test",
                        "distilabel_metadata": {
                            "statistics_embedding_task_generator": {
                                "input_tokens": 12,
                                "output_tokens": 12,
                            }
                        },
                    },
                    {
                        "task": "B",
                        "model_name": "test",
                        "distilabel_metadata": {
                            "statistics_embedding_task_generator": {
                                "input_tokens": 12,
                                "output_tokens": 12,
                            }
                        },
                    },
                    {
                        "task": "C",
                        "model_name": "test",
                        "distilabel_metadata": {
                            "statistics_embedding_task_generator": {
                                "input_tokens": 12,
                                "output_tokens": 12,
                            }
                        },
                    },
                ],
                True,
            )
        )
        assert next(task.process()) == result


class TestBitextRetrievalGenerator:
    @pytest.mark.parametrize(
        "task_kwargs",
        [
            {
                "source_language": "English",
                "target_language": "French",
                "unit": "sentence",
                "difficulty": "elementary school",
                "high_score": "4",
                "low_score": "2.5",
            }
        ],
    )
    def test_prompt(self, task_kwargs: Any) -> None:
        task = BitextRetrievalGenerator(
            name="bitext_retrieval_generator",
            **task_kwargs,
            add_raw_output=False,
            llm=MockLLM(output=json.dumps({"S1": "A", "S2": "B", "S3": "C"})),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        assert all(
            task.prompt[-1]["content"].__contains__(v) for _, v in task_kwargs.items()
        )

    def test_process(self) -> None:
        task = BitextRetrievalGenerator(
            name="bitext_retrieval_generator",
            source_language="English",
            target_language="French",
            add_raw_output=False,
            llm=MockLLM(output=json.dumps({"S1": "A", "S2": "B", "S3": "C"})),
            pipeline=Pipeline(name="unit-test-pipeline"),
            add_raw_input=False,
        )
        task.load()

        assert task.outputs == ["S1", "S2", "S3", "model_name"]

        assert next(task.process()) == (
            [
                {
                    "S1": "A",
                    "S2": "B",
                    "S3": "C",
                    "model_name": "test",
                    "distilabel_metadata": {
                        "statistics_bitext_retrieval_generator": {
                            "input_tokens": 12,
                            "output_tokens": 12,
                        }
                    },
                }
            ],
            True,
        )

    def test_reproducibility(self) -> None:
        unique_prompts = set()
        for _ in range(10):
            task = BitextRetrievalGenerator(
                name="bitext_retrieval_generator",
                source_language="English",
                target_language="French",
                add_raw_output=False,
                seed=42,
                llm=MockLLM(output=json.dumps({"S1": "A", "S2": "B", "S3": "C"})),
                pipeline=Pipeline(name="unit-test-pipeline"),
            )
            task.load()

            unique_prompts.add(task.prompt[-1]["content"])

        assert len(unique_prompts) == 1


class TestMonolingualTripletGenerator:
    @pytest.mark.parametrize(
        "task_kwargs",
        [
            {
                "language": "English",
                "unit": "sentence",
                "difficulty": "elementary school",
                "high_score": "4",
                "low_score": "2.5",
            }
        ],
    )
    def test_prompt(self, task_kwargs: Any) -> None:
        task = MonolingualTripletGenerator(
            name="monolingual_triplet_generator",
            **task_kwargs,
            add_raw_output=False,
            llm=MockLLM(output=json.dumps({"S1": "A", "S2": "B", "S3": "C"})),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()
        assert all(
            task.prompt[-1]["content"].__contains__(v) for _, v in task_kwargs.items()
        )

    def test_process(self) -> None:
        task = MonolingualTripletGenerator(
            name="monolingual_triplet_generator",
            language="English",
            add_raw_output=False,
            llm=MockLLM(output=json.dumps({"S1": "A", "S2": "B", "S3": "C"})),
            pipeline=Pipeline(name="unit-test-pipeline"),
            add_raw_input=False,
        )
        task.load()
        assert task.outputs == ["S1", "S2", "S3", "model_name"]
        assert next(task.process()) == (
            [
                {
                    "S1": "A",
                    "S2": "B",
                    "S3": "C",
                    "model_name": "test",
                    "distilabel_metadata": {
                        "statistics_monolingual_triplet_generator": {
                            "input_tokens": 12,
                            "output_tokens": 12,
                        }
                    },
                }
            ],
            True,
        )

    def test_reproducibility(self) -> None:
        unique_prompts = set()
        for _ in range(10):
            task = MonolingualTripletGenerator(
                name="monolingual_triplet_generator",
                language="English",
                add_raw_output=False,
                seed=42,
                llm=MockLLM(output=json.dumps({"S1": "A", "S2": "B", "S3": "C"})),
                pipeline=Pipeline(name="unit-test-pipeline"),
            )
            task.load()
            unique_prompts.add(task.prompt[-1]["content"])
        assert len(unique_prompts) == 1


class TestGenerateLongTextMatchingData:
    def test_format_input(self) -> None:
        task = GenerateLongTextMatchingData(
            name="generate_long_text_matching_data",
            language="English",
            add_raw_output=False,
            llm=MockLLM(output=json.dumps({"input": "A", "positive_document": "B"})),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        assert task.format_input({"task": "A"})[-1]["content"].startswith(
            "You have been assigned a text matching task: A"
        )

    def test_process(self) -> None:
        task = GenerateLongTextMatchingData(
            name="generate_long_text_matching_data",
            language="English",
            add_raw_output=False,
            llm=MockLLM(output=json.dumps({"input": "A", "positive_document": "B"})),
            pipeline=Pipeline(name="unit-test-pipeline"),
            add_raw_input=False,
        )
        task.load()

        assert task.outputs == ["input", "positive_document", "model_name"]

        assert next(task.process(inputs=[{"task": "A"}])) == [
            {
                "task": "A",
                "input": "A",
                "positive_document": "B",
                "model_name": "test",
                "distilabel_metadata": {
                    "statistics_generate_long_text_matching_data": {
                        "input_tokens": 12,
                        "output_tokens": 12,
                    }
                },
            }
        ]


class TestGenerateShortTextMatchingData:
    def test_format_input(self) -> None:
        task = GenerateShortTextMatchingData(
            name="generate_short_text_matching_data",
            language="English",
            add_raw_output=False,
            llm=MockLLM(output=json.dumps({"input": "A", "positive_document": "B"})),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()
        assert task.format_input({"task": "A"})[-1]["content"].startswith(
            "You have been assigned a text matching task: A"
        )

    def test_process(self) -> None:
        task = GenerateShortTextMatchingData(
            name="generate_short_text_matching_data",
            language="English",
            add_raw_output=False,
            llm=MockLLM(output=json.dumps({"input": "A", "positive_document": "B"})),
            pipeline=Pipeline(name="unit-test-pipeline"),
            add_raw_input=False,
        )
        task.load()
        assert task.outputs == ["input", "positive_document", "model_name"]
        assert next(task.process(inputs=[{"task": "A"}])) == [
            {
                "task": "A",
                "input": "A",
                "positive_document": "B",
                "model_name": "test",
                "distilabel_metadata": {
                    "statistics_generate_short_text_matching_data": {
                        "input_tokens": 12,
                        "output_tokens": 12,
                    }
                },
            }
        ]

    def test_reproducibility(self) -> None:
        unique_prompts = set()
        for _ in range(10):
            task = GenerateShortTextMatchingData(
                name="generate_short_text_matching_data",
                language="English",
                add_raw_output=False,
                seed=42,
                llm=MockLLM(
                    output=json.dumps({"input": "A", "positive_document": "B"})
                ),
                pipeline=Pipeline(name="unit-test-pipeline"),
            )
            task.load()
            unique_prompts.add(task.format_input({"task": "A"})[-1]["content"])

        assert len(unique_prompts) == 1


class TestGenerateTextClassificationData:
    def test_format_input(self) -> None:
        task = GenerateTextClassificationData(
            name="generate_text_classification_data",
            language="English",
            add_raw_output=False,
            llm=MockLLM(
                output=json.dumps(
                    {"input_text": "A", "label": "B", "misleading_label": "C"}
                )
            ),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()
        assert task.format_input({"task": "A"})[-1]["content"].startswith(
            "You have been assigned a text classification task: A"
        )

    def test_process(self) -> None:
        task = GenerateTextClassificationData(
            name="generate_text_classification_data",
            language="English",
            add_raw_output=False,
            llm=MockLLM(
                output=json.dumps(
                    {"input_text": "A", "label": "B", "misleading_label": "C"}
                )
            ),
            pipeline=Pipeline(name="unit-test-pipeline"),
            add_raw_input=False,
        )
        task.load()
        assert task.outputs == ["input_text", "label", "misleading_label", "model_name"]
        assert next(task.process(inputs=[{"task": "A"}])) == [
            {
                "task": "A",
                "input_text": "A",
                "label": "B",
                "misleading_label": "C",
                "model_name": "test",
                "distilabel_metadata": {
                    "statistics_generate_text_classification_data": {
                        "input_tokens": 12,
                        "output_tokens": 12,
                    }
                },
            }
        ]

    def test_reproducibility(self) -> None:
        unique_prompts = set()
        for _ in range(10):
            task = GenerateTextClassificationData(
                name="generate_text_classification_data",
                language="English",
                add_raw_output=False,
                seed=42,
                llm=MockLLM(
                    output=json.dumps(
                        {"input_text": "A", "label": "B", "misleading_label": "C"}
                    )
                ),
                pipeline=Pipeline(name="unit-test-pipeline"),
            )
            task.load()
            unique_prompts.add(task.format_input({"task": "A"})[-1]["content"])

        assert len(unique_prompts) == 1


class TestGenerateTextRetrievalData:
    def test_format_input(self) -> None:
        task = GenerateTextRetrievalData(
            name="generate_text_retrieval_data",
            language="English",
            add_raw_output=False,
            llm=MockLLM(
                output=json.dumps(
                    {
                        "user_query": "A",
                        "positive_document": "B",
                        "hard_negative_document": "C",
                    }
                )
            ),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()
        assert task.format_input({"task": "A"})[-1]["content"].startswith(
            "You have been assigned a retrieval task: A"
        )

    def test_process(self) -> None:
        task = GenerateTextRetrievalData(
            name="generate_text_retrieval_data",
            language="English",
            add_raw_output=False,
            llm=MockLLM(
                output=json.dumps(
                    {
                        "user_query": "A",
                        "positive_document": "B",
                        "hard_negative_document": "C",
                    }
                )
            ),
            pipeline=Pipeline(name="unit-test-pipeline"),
            add_raw_input=False,
        )
        task.load()
        assert task.outputs == [
            "user_query",
            "positive_document",
            "hard_negative_document",
            "model_name",
        ]
        assert next(task.process(inputs=[{"task": "A"}])) == [
            {
                "task": "A",
                "user_query": "A",
                "positive_document": "B",
                "hard_negative_document": "C",
                "model_name": "test",
                "distilabel_metadata": {
                    "statistics_generate_text_retrieval_data": {
                        "input_tokens": 12,
                        "output_tokens": 12,
                    }
                },
            }
        ]
