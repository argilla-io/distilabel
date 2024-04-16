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

from typing import Generator

import pytest
from distilabel.llms.huggingface.transformers import TransformersLLM
from distilabel.pipeline.local import Pipeline
from distilabel.steps.tasks.generate_embeddings import GenerateEmbeddings


@pytest.fixture(scope="module")
def transformers_llm() -> Generator[TransformersLLM, None, None]:
    llm = TransformersLLM(
        model="TaylorAI/bge-micro-v2",
        model_kwargs={"is_decoder": True},
        cuda_devices=[],
    )
    llm.load()

    yield llm


class TestGenerateEmbeddings:
    def test_process(self, transformers_llm: TransformersLLM) -> None:
        task = GenerateEmbeddings(
            name="task",
            llm=transformers_llm,
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        result = next(task.process([{"text": "Hello, how are you?"}]))

        assert "embedding" in result[0]
        assert len(result[0]["embedding"]) == 384
