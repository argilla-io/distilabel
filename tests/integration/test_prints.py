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

from functools import partial
from typing import Union

import pytest

from distilabel.llms.mixins.magpie import MagpieChatTemplateMixin
from distilabel.steps import tasks as tasks_
from tests.unit.conftest import DummyLLM

# The tasks not listed here don't have a print method (or don't have a print method that works)
tasks = [
    tasks_.ComplexityScorer,
    partial(tasks_.EvolInstruct, num_evolutions=1),
    partial(tasks_.EvolComplexity, num_evolutions=1),
    partial(tasks_.EvolComplexityGenerator, num_instructions=1),
    partial(tasks_.EvolInstructGenerator, num_instructions=1),
    partial(tasks_.EvolQuality, num_evolutions=1),
    tasks_.Genstruct,
    partial(
        tasks_.BitextRetrievalGenerator,
        source_language="English",
        target_language="Spanish",
        unit="sentence",
        difficulty="elementary school",
        high_score="4",
        low_score="2.5",
    ),
    partial(tasks_.EmbeddingTaskGenerator, category="text-retrieval"),
    tasks_.GenerateLongTextMatchingData,
    tasks_.GenerateShortTextMatchingData,
    tasks_.GenerateTextClassificationData,
    tasks_.GenerateTextRetrievalData,
    tasks_.MonolingualTripletGenerator,
    tasks_.InstructionBacktranslation,
    tasks_.Magpie,
    tasks_.MagpieGenerator,
    partial(tasks_.PrometheusEval, mode="absolute", rubric="factual-validity"),
    tasks_.QualityScorer,
    tasks_.SelfInstruct,
    partial(tasks_.GenerateSentencePair, action="paraphrase"),
    tasks_.UltraFeedback,
    tasks_.URIAL,
]


class TestLLM(DummyLLM, MagpieChatTemplateMixin):
    magpie_pre_query_template: Union[str, None] = "llama3"


llm = TestLLM()


@pytest.mark.parametrize("task", tasks)
def test_prints(task) -> None:
    t = task(llm=llm)
    t.load()
    t.print()
    t.unload()
