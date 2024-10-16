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

from distilabel.steps.tasks.apigen.execution_checker import APIGenExecutionChecker
from distilabel.steps.tasks.apigen.generator import APIGenGenerator
from distilabel.steps.tasks.apigen.semantic_checker import APIGenSemanticChecker
from distilabel.steps.tasks.argilla_labeller import ArgillaLabeller
from distilabel.steps.tasks.base import GeneratorTask, Task
from distilabel.steps.tasks.clair import CLAIR
from distilabel.steps.tasks.complexity_scorer import ComplexityScorer
from distilabel.steps.tasks.decorator import task
from distilabel.steps.tasks.evol_instruct.base import EvolInstruct
from distilabel.steps.tasks.evol_instruct.evol_complexity.base import EvolComplexity
from distilabel.steps.tasks.evol_instruct.evol_complexity.generator import (
    EvolComplexityGenerator,
)
from distilabel.steps.tasks.evol_instruct.generator import EvolInstructGenerator
from distilabel.steps.tasks.evol_quality.base import EvolQuality
from distilabel.steps.tasks.generate_embeddings import GenerateEmbeddings
from distilabel.steps.tasks.genstruct import Genstruct
from distilabel.steps.tasks.improving_text_embeddings import (
    BitextRetrievalGenerator,
    EmbeddingTaskGenerator,
    GenerateLongTextMatchingData,
    GenerateShortTextMatchingData,
    GenerateTextClassificationData,
    GenerateTextRetrievalData,
    MonolingualTripletGenerator,
)
from distilabel.steps.tasks.instruction_backtranslation import (
    InstructionBacktranslation,
)
from distilabel.steps.tasks.magpie.base import Magpie
from distilabel.steps.tasks.magpie.generator import MagpieGenerator
from distilabel.steps.tasks.pair_rm import PairRM
from distilabel.steps.tasks.prometheus_eval import PrometheusEval
from distilabel.steps.tasks.quality_scorer import QualityScorer
from distilabel.steps.tasks.self_instruct import SelfInstruct
from distilabel.steps.tasks.sentence_transformers import GenerateSentencePair
from distilabel.steps.tasks.structured_generation import StructuredGeneration
from distilabel.steps.tasks.text_classification import TextClassification
from distilabel.steps.tasks.text_generation import ChatGeneration, TextGeneration
from distilabel.steps.tasks.typing import ChatItem, ChatType
from distilabel.steps.tasks.ultrafeedback import UltraFeedback
from distilabel.steps.tasks.urial import URIAL

__all__ = [
    "GeneratorTask",
    "Task",
    "ArgillaLabeller",
    "APIGenExecutionChecker",
    "APIGenGenerator",
    "APIGenSemanticChecker",
    "ComplexityScorer",
    "task",
    "EvolInstruct",
    "EvolComplexity",
    "EvolComplexityGenerator",
    "EvolInstructGenerator",
    "EvolQuality",
    "GenerateEmbeddings",
    "Genstruct",
    "BitextRetrievalGenerator",
    "EmbeddingTaskGenerator",
    "GenerateLongTextMatchingData",
    "GenerateShortTextMatchingData",
    "GenerateTextClassificationData",
    "GenerateTextRetrievalData",
    "MonolingualTripletGenerator",
    "InstructionBacktranslation",
    "Magpie",
    "MagpieGenerator",
    "PairRM",
    "PrometheusEval",
    "QualityScorer",
    "SelfInstruct",
    "GenerateSentencePair",
    "StructuredGeneration",
    "TextClassification",
    "ChatGeneration",
    "TextGeneration",
    "ChatItem",
    "ChatType",
    "CLAIR",
    "UltraFeedback",
    "URIAL",
]
