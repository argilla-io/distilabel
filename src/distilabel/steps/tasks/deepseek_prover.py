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

import sys

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

import re
from typing import Any, Dict, List, Optional, Union

from jinja2 import Template
from pydantic import PrivateAttr
from typing_extensions import override

from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.typing import ChatType

_PARSE_DEEPSEEK_PROVER_AUTOFORMAL_REGEX = r"```lean4(.*?)```"

_PARSE_DEEPSEEK_PROVER_SCORER_REGEX = (
    r"Natural language:\n(.*)\nAnalysis:\n(.*)\nAssessment:\n(.*)"
)


class DeepSeekProverAutoFormalization(Task):
    """Task to translate a mathematical problem from natural language to Lean 4.

    Note:
        A related dataset (MMA from the paper) can be found in Hugging Face:
        [casey-martin/multilingual-mathematical-autoformalization](https://huggingface.co/datasets/casey-martin/multilingual-mathematical-autoformalization).

    Input columns:
        - informal_statement (`str`): The statement to be formalized using Lean 4.

    Categories:
        - generation

    References:
        - [`DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data`](https://arxiv.org/abs/2405.14333).
        - [`Lean 4`](https://github.com/leanprover/lean4).

    Examples:

        Formalize a mathematical problem from natural language to Lean 4:

        ```python
        from distilabel.steps.tasks import DeepSeekProverAutoFormalization
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        prover_autoformal = DeepSeekProverAutoFormalization(
            llm=InferenceEndpointsLLM(
                model_id="deepseek-ai/deepseek-math-7b-instruct",
                tokenizer_id="deepseek-ai/deepseek-math-7b-instruct",
            ),
        )

        prover_autoformal.load()

        result = next(
            prover_autoformal.process(
                [
                    {"informal_statement": "If a polynomial g is monic, then the root of g is integral over the ring R."},
                ]
            )
        )
        # result
        # [
        #     {
        #         'informal_statement': 'If a polynomial g is monic, then the root of g is integral over the ring R.',
        #         'formal_statement': 'theorem isIntegral_root (hg : g.Monic) : IsIntegral R (root g):=',
        #         'distilabel_metadata': {
        #             'raw_output_deep_seek_prover_auto_formalization_0': '```lean4\ntheorem isIntegral_root (hg : g.Monic) : IsIntegral R (root g):=\n```'
        #         },
        #         'model_name': 'deepseek-prover'
        #     }
        # ]
        ```

        Use a few-shot setting to formalize a mathematical problem from natural language to Lean 4:

        ```python
        from distilabel.steps.tasks import DeepSeekProverAutoFormalization
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        prover_autoformal = DeepSeekProverAutoFormalization(
            llm=InferenceEndpointsLLM(
                model_id="deepseek-ai/deepseek-math-7b-instruct",
                tokenizer_id="deepseek-ai/deepseek-math-7b-instruct",
            ),
            few_shot=True,
            examples=[
                "theorem amc12a_2019_p21 (z : ℂ) (h₀ : z = (1 + Complex.I) / Real.sqrt 2) :\n\n((∑ k : ℤ in Finset.Icc 1 12, z ^ k ^ 2) * (∑ k : ℤ in Finset.Icc 1 12, 1 / z ^ k ^ 2)) = 36 := by\n\nsorry",
                "theorem amc12a_2015_p10 (x y : ℤ) (h₀ : 0 < y) (h₁ : y < x) (h₂ : x + y + x * y = 80) : x = 26 := by\n\nsorry"
            ]
        )

        prover_autoformal.load()

        result = next(
            prover_autoformal.process(
                [
                    {"informal_statement": "If a polynomial g is monic, then the root of g is integral over the ring R."},
                ]
            )
        )
        # result
        # [
        #     {
        #         'informal_statement': 'If a polynomial g is monic, then the root of g is integral over the ring R.',
        #         'formal_statement': 'theorem isIntegral_root (hg : g.Monic) : IsIntegral R (root g):=',
        #         'distilabel_metadata': {
        #             'raw_output_deep_seek_prover_auto_formalization_0': '```lean4\ntheorem isIntegral_root (hg : g.Monic) : IsIntegral R (root g):=\n```'
        #         },
        #         'model_name': 'deepseek-prover'
        #     }
        # ]
        ```
    """

    few_shot: bool = False
    examples: Optional[List[str]] = None
    system_prompt: str = "Translate the problem to Lean 4 (only the core declaration):\n```lean4\nformal statement goes here\n```"
    _template: Union[Template, None] = PrivateAttr(...)

    def load(self) -> None:
        """Loads the Jinja2 template."""
        super().load()

        _path = str(
            importlib_resources.files("distilabel")
            / "steps"
            / "tasks"
            / "templates"
            / "deepseek_prover"
            / "deepseek-prover-autoformal.jinja2"
        )
        with open(_path, "r") as f:
            self._template = Template(f.read())

    @property
    def inputs(self) -> List[str]:
        """The input for the task is the `instruction`."""
        return ["informal_statement"]

    @property
    def outputs(self):
        """The output for the task is a list of `instructions` containing the generated instructions."""
        return ["formal_statement", "model_name"]

    def format_input(self, input: str) -> ChatType:  # type: ignore
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation. And the
        `system_prompt` is added as the first message if it exists."""
        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": self._template.render(
                    informal_statement=input[self.inputs[0]],
                    few_shot=self.few_shot,
                    examples=self.examples,
                ),
            },
        ]

    @override
    def format_output(  # type: ignore
        self, output: Union[str, None], input: Dict[str, Any] = None
    ) -> Dict[str, Any]:  # type: ignore
        """Extracts the formal statement from the Lean 4 output."""
        match = re.match(_PARSE_DEEPSEEK_PROVER_AUTOFORMAL_REGEX, output, re.DOTALL)
        if match:
            match = match.group(1).strip()
        return {"formal_statement": match}


class DeepSeekProverScorer(Task):
    """Task to evaluate the quality of a formalized mathematical problem in Lean 4,
    inspired by the DeepSeek-Prover task for scoring.

    Note:
        A related dataset (MMA from the paper) can be found in Hugging Face:
        [casey-martin/multilingual-mathematical-autoformalization](https://huggingface.co/datasets/casey-martin/multilingual-mathematical-autoformalization).

    Input columns:
        - formal_statement (`str`): The formalized statement using Lean 4, to be analysed.

    Categories:
        - scorer
        - quality
        - response

    References:
        - [`DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data`](https://arxiv.org/abs/2405.14333).
        - [`Lean 4`](https://github.com/leanprover/lean4).

    Examples:

        Analyse a formal statement in Lean 4:

        ```python
        from distilabel.steps.tasks import DeepSeekProverScorer
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        prover_scorer = DeepSeekProverAutoFormalization(
            llm=InferenceEndpointsLLM(
                model_id="deepseek-ai/deepseek-math-7b-instruct",
                tokenizer_id="deepseek-ai/deepseek-math-7b-instruct",
            ),
        )

        prover_scorer.load()

        result = next(
            prover_scorer.process(
                [
                    {"formal_statement": "theorem isIntegral_root (hg : g.Monic) : IsIntegral R (root g):="},
                ]
            )
        )
        # result
        # [
        #     {
        #         'formal_statement': 'theorem isIntegral_root (hg : g.Monic) : IsIntegral R (root g):=',
        #         'informal_statement': 'INFORMAL',
        #         'analysis': 'ANALYSIS',
        #         'assessment': 'ASSESSMENT',
        #         'distilabel_metadata': {
        #             'raw_output_deep_seek_prover_scorer_0': 'Natural language:\nINFORMAL\nAnalysis:\nANALYSIS\nAssessment:\nASSESSMENT'
        #         },
        #         'model_name': 'deepseek-prover-scorer'
        #     }
        # ]
        ```
    """

    _template: Union[Template, None] = PrivateAttr(...)

    def load(self) -> None:
        """Loads the Jinja2 template."""
        super().load()

        _path = str(
            importlib_resources.files("distilabel")
            / "steps"
            / "tasks"
            / "templates"
            / "deepseek_prover"
            / "deepseek-prover-scorer.jinja2"
        )
        with open(_path, "r") as f:
            self._template = Template(f.read())

    @property
    def inputs(self) -> List[str]:
        """The input for the task is the `instruction`."""
        return ["formal_statement"]

    @property
    def outputs(self):
        """The output for the task is a list of `instructions` containing the generated instructions."""
        return ["informal_statement", "analysis", "assessment", "model_name"]

    def format_input(self, input: str) -> ChatType:  # type: ignore
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation. And the
        `system_prompt` is added as the first message if it exists."""
        return [
            {
                "role": "system",
                "content": self._template.render(),
            },
            {
                "role": "user",
                "content": input[self.inputs[0]],
            },
        ]

    @override
    def format_output(  # type: ignore
        self, output: Union[str, None], input: Dict[str, Any] = None
    ) -> Dict[str, Any]:  # type: ignore
        """Analyses the formal statement with Lean 4 output and generates an assessment
        and the corresponding informal assessment."""

        matches = re.match(_PARSE_DEEPSEEK_PROVER_SCORER_REGEX, output)
        informal = analysis = assessment = None
        if matches:
            informal, analysis, assessment = matches.groups()

        return {
            "informal_statement": informal,
            "analysis": analysis,
            "assessment": assessment,
        }
