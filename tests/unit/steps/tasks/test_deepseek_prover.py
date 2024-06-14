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

from typing import Any, List

import pytest
from distilabel.llms.base import LLM
from distilabel.llms.typing import GenerateOutput
from distilabel.steps.tasks.deepseek_prover import (
    DeepSeekProverAutoFormalization,
    DeepSeekProverScorer,
)
from distilabel.steps.tasks.typing import ChatType


class DeepSeekProverLLM(LLM):
    def load(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return "deepseek-prover"

    def generate(
        self, inputs: List[ChatType], num_generations: int = 1, **kwargs: Any
    ) -> List[GenerateOutput]:
        response = "theorem isIntegral_root (hg : g.Monic) : IsIntegral R (root g):="
        return [
            [f"```lean4\n{response}\n```" for _ in range(num_generations)]
            for _ in inputs
        ]


class DeepSeekProverScorerLLM(LLM):
    response: str = (
        "Natural language:\nINFORMAL\nAnalysis:\nANALYSIS\nAssessment:\nASSESSMENT"
    )

    def load(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return "deepseek-prover-scorer"

    def generate(
        self, inputs: List[ChatType], num_generations: int = 1, **kwargs: Any
    ) -> List[GenerateOutput]:
        return [[self.response for _ in range(num_generations)] for _ in inputs]


class TestDeepSeekProverAutoFormalization:
    @pytest.mark.parametrize(
        "examples, expected",
        [
            (
                [],
                "Mathematical Problem in Natural Language:\n{informal_statement}",
            ),
            (
                ["ex1", "ex2"],
                "Mathematical Problem in Natural Language:\n{informal_statement}\n\nPlease use the following examples to guide you with the answer:\n- ex1\n- ex2\n",
            ),
        ],
    )
    def test_format_input(self, examples: List[str], expected: str) -> None:
        task = DeepSeekProverAutoFormalization(
            llm=DeepSeekProverLLM(), examples=examples
        )
        informal_statement = "If a polynomial g is monic, then the root of g is integral over the ring R."
        task.load()
        assert task.format_input({"informal_statement": informal_statement}) == [
            {
                "role": "system",
                "content": "Translate the problem to Lean 4 (only the core declaration):\n```lean4\nformal statement goes here\n```",
            },
            {
                "role": "user",
                "content": expected.format(informal_statement=informal_statement),
            },
        ]

    def test_format_output(self) -> None:
        task = DeepSeekProverAutoFormalization(
            llm=DeepSeekProverLLM(),
        )
        task.load()
        raw_response = "```lean4\ntheorem isIntegral_root (hg : g.Monic) : IsIntegral R (root g):=\n```"
        assert task.format_output(raw_response, {}) == {
            "formal_statement": "theorem isIntegral_root (hg : g.Monic) : IsIntegral R (root g):="
        }

    def test_process(self) -> None:
        task = DeepSeekProverAutoFormalization(
            llm=DeepSeekProverLLM(),
        )
        task.load()

        assert next(
            task.process(
                [
                    {
                        "informal_statement": "If a polynomial g is monic, then the root of g is integral over the ring R."
                    }
                ]
            )
        ) == [
            {
                "informal_statement": "If a polynomial g is monic, then the root of g is integral over the ring R.",
                "formal_statement": "theorem isIntegral_root (hg : g.Monic) : IsIntegral R (root g):=",
                "model_name": "deepseek-prover",
                "distilabel_metadata": {
                    "raw_output_deep_seek_prover_auto_formalization_0": "```lean4\ntheorem isIntegral_root (hg : g.Monic) : IsIntegral R (root g):=\n```"
                },
            }
        ]


class TestDeepSeekProverScorer:
    formal_statement = (
        "theorem isIntegral_root (hg : g.Monic) : IsIntegral R (root g):="
    )

    def test_format_input(self) -> None:
        task = DeepSeekProverScorer(
            llm=DeepSeekProverScorerLLM(),
        )
        task.load()
        assert task.format_input({"formal_statement": self.formal_statement}) == [
            {
                "role": "system",
                "content": task._template.render(),
            },
            {
                "role": "user",
                "content": self.formal_statement,
            },
        ]

    def test_format_output(self) -> None:
        task = DeepSeekProverScorer(
            llm=DeepSeekProverScorerLLM(),
        )
        task.load()
        raw_response = task.llm.response
        assert task.format_output(raw_response, {}) == {
            "informal_statement": "INFORMAL",
            "analysis": "ANALYSIS",
            "assessment": "ASSESSMENT",
        }

    def test_process(self) -> None:
        task = DeepSeekProverScorer(
            llm=DeepSeekProverScorerLLM(),
        )
        task.load()

        assert next(
            task.process(
                [
                    {
                        "formal_statement": "theorem isIntegral_root (hg : g.Monic) : IsIntegral R (root g):="
                    }
                ]
            )
        ) == [
            {
                "formal_statement": "theorem isIntegral_root (hg : g.Monic) : IsIntegral R (root g):=",
                "informal_statement": "INFORMAL",
                "analysis": "ANALYSIS",
                "assessment": "ASSESSMENT",
                "model_name": "deepseek-prover-scorer",
                "distilabel_metadata": {
                    "raw_output_deep_seek_prover_scorer_0": "Natural language:\nINFORMAL\nAnalysis:\nANALYSIS\nAssessment:\nASSESSMENT"
                },
            }
        ]
