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
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Union

from jinja2 import Template
from pydantic import PrivateAttr
from typing_extensions import override

from distilabel.models import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub
from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.typing import ChatType

_PARSE_DEEPSEEK_PROVER_AUTOFORMAL_REGEX = r"```lean4(.*?)```"


template_deepseek_prover_auto_formalization = """\
Mathematical Problem in Natural Language:
{{ informal_statement }}
{%- if few_shot %}

Please use the following examples to guide you with the answer:
{%- for example in examples %}
- {{ example }}
{%- endfor %}
{% endif -%}"""


class DeepSeekProverAutoFormalization(Task):
    """Task to translate a mathematical problem from natural language to Lean 4.

    Note:
        A related dataset (MMA from the paper) can be found in Hugging Face:
        [casey-martin/multilingual-mathematical-autoformalization](https://huggingface.co/datasets/casey-martin/multilingual-mathematical-autoformalization).

    Input columns:
        - informal_statement (`str`): The statement to be formalized using Lean 4.

    Output columns:
        - formal_statement (`str`): The formalized statement using Lean 4, to be analysed.

    Categories:
        - generation

    References:
        - [`DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data`](https://arxiv.org/abs/2405.14333).
        - [`Lean 4`](https://github.com/leanprover/lean4).

    Examples:

        Formalize a mathematical problem from natural language to Lean 4:

        ```python
        from distilabel.steps.tasks import DeepSeekProverAutoFormalization
        from distilabel.models import InferenceEndpointsLLM

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
        from distilabel.models import InferenceEndpointsLLM

        # You can gain inspiration from the following examples to create your own few-shot examples:
        # https://github.com/yangky11/miniF2F-lean4/blob/main/MiniF2F/Valid.lean
        # Consider this as a placeholder for your actual LLM.
        prover_autoformal = DeepSeekProverAutoFormalization(
            llm=InferenceEndpointsLLM(
                model_id="deepseek-ai/deepseek-math-7b-instruct",
                tokenizer_id="deepseek-ai/deepseek-math-7b-instruct",
            ),
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

    examples: Optional[List[str]] = None
    system_prompt: str = "Translate the problem to Lean 4 (only the core declaration):\n```lean4\nformal statement goes here\n```"
    _template: Union[Template, None] = PrivateAttr(...)
    _few_shot: bool = PrivateAttr(default=False)

    def load(self) -> None:
        """Loads the Jinja2 template."""
        super().load()

        self._template = Template(template_deepseek_prover_auto_formalization)

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
                    few_shot=bool(self.examples),
                    examples=self.examples,
                ),
            },
        ]

    @override
    def format_output(  # type: ignore
        self, output: Union[str, None], input: Dict[str, Any] = None
    ) -> Dict[str, Any]:  # type: ignore
        """Extracts the formal statement from the Lean 4 output."""
        match = re.search(_PARSE_DEEPSEEK_PROVER_AUTOFORMAL_REGEX, output, re.DOTALL)
        if match:
            match = match.group(1).strip()
        return {"formal_statement": match}


template_deepseek_prover_scorer = """\
To evaluate whether a formal Lean4 statement will be of interest to the community, consider the following criteria:

1. Relevance to Current Research: Does the statement address a problem or concept that is actively being researched in mathematics or related fields? Higher relevance scores indicate greater potential interest.
2. Complexity and Depth: Is the statement complex enough to challenge existing theories and methodologies, yet deep enough to provide significant insights or advancements? Complexity and depth showcase Lean4's capabilities and attract interest.
3. Interdisciplinary Potential: Does the statement offer opportunities for interdisciplinary research, connecting mathematics with other fields such as computer science, physics, or biology? Interdisciplinary projects often garner wide interest.
4. Community Needs and Gaps: Does the statement fill an identified need or gap within the Lean4 community or the broader mathematical community? Addressing these needs directly correlates with interest.
5. Innovativeness: How innovative is the statement? Does it propose new methods, concepts, or applications? Innovation drives interest and engagement.

Customize your evaluation for each problem accordingly, assessing it as 'excellent', 'good', 'above average', 'fair' or 'poor'.

You should respond in the following format for each statement:

'''
Natural language: (Detailed explanation of the informal statement, including any relevant background information, assumptions, and definitions.)
Analysis: (Provide a brief justification for each score, highlighting why the statement scored as it did across the criteria.)
Assessment: (Based on the criteria, rate the statement as 'excellent', 'good', 'above average', 'fair' or 'poor'. JUST the Assessment.)
'''"""


class DeepSeekProverScorer(Task):
    """Task to evaluate the quality of a formalized mathematical problem in Lean 4,
    inspired by the DeepSeek-Prover task for scoring.

    Note:
        A related dataset (MMA from the paper) can be found in Hugging Face:
        [casey-martin/multilingual-mathematical-autoformalization](https://huggingface.co/datasets/casey-martin/multilingual-mathematical-autoformalization).

    Input columns:
        - informal_statement (`str`): The statement to be formalized using Lean 4.
        - formal_statement (`str`): The formalized statement using Lean 4, to be analysed.

    Output columns:
        - natural_language (`str`): Explanation for the problem.
        - analysis (`str`): Analysis of the different points defined in the prompt.
        - assessment (`str`): Result of the assessment.

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
        from distilabel.models import InferenceEndpointsLLM

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

        self._template = Template(template_deepseek_prover_scorer)

    @property
    def inputs(self) -> List[str]:
        """The input for the task is the `instruction`."""
        return ["informal_statement", "formal_statement"]

    @property
    def outputs(self):
        """The output for the task is a list of `instructions` containing the generated instructions."""
        return ["natural_language", "analysis", "assessment", "model_name"]

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
                "content": f"## Informal statement:\n{input[self.inputs[0]]}\n\n ## Formal statement:\n{input[self.inputs[1]]}",
            },
        ]

    @override
    def format_output(  # type: ignore
        self, output: Union[str, None], input: Dict[str, Any] = None
    ) -> Dict[str, Any]:  # type: ignore
        """Analyses the formal statement with Lean 4 output and generates an assessment
        and the corresponding informal assessment."""

        try:
            result = output.split("Natural language:")[1].strip()
            natural_language, analysis = result.split("Analysis:")
            analysis, assessment = analysis.split("Assessment:")
            natural_language = natural_language.strip()
            analysis = analysis.strip()
            assessment = assessment.strip()
        except Exception:
            natural_language = analysis = assessment = None

        return {
            "natural_language": natural_language,
            "analysis": analysis,
            "assessment": assessment,
        }


class DeepSeekProverSolver(Task):
    """Task to generate a proof for a formal statement (theorem) in lean4.

    Input columns:
        - formal_statement (`str`): The formalized statement using Lean 4.

    Output columns:
        - proof (`str`): The proof for the formal statement theorem.

    Categories:
        - scorer
        - quality
        - response

    References:
        - [`DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data`](https://arxiv.org/abs/2405.14333).
    """

    system_prompt: str = (
        "You are an expert in proving mathematical theorems formalized in lean4 language. "
        "Your answers consist just in the proof to the theorem given, and nothing else."
    )

    @property
    def inputs(self) -> List[str]:
        """The input for the task is the `formal_statement`."""
        return ["formal_statement"]

    @property
    def outputs(self):
        """The output for the task is the proof for the formal statement theorem."""
        return ["proof"]

    def format_input(self, input: str) -> ChatType:  # type: ignore
        """The input is formatted as a `ChatType`, with a system prompt to guide our model."""
        prompt = dedent("""
            Give me a proof for the following theorem:
            ```lean4
            {theorem}
            ```""")
        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": prompt.format(theorem=input["formal_statement"]),
            },
        ]

    def format_output(  # type: ignore
        self, output: Union[str, None], input: Dict[str, Any] = None
    ) -> Dict[str, Any]:  # type: ignore
        import re

        match = re.search(_PARSE_DEEPSEEK_PROVER_AUTOFORMAL_REGEX, output, re.DOTALL)
        if match:
            match = match.group(1).strip()
        return {"proof": match}


examples = [
    dedent("""
    ## Statement in natural language:
    For real numbers k and x:
    If x is equal to (13 - √131) / 4, and
    If the equation 2x² - 13x + k = 0 is satisfied,
    Then k must be equal to 19/4.
    ## Formalized:
    theorem mathd_algebra_116 (k x : ℝ) (h₀ : x = (13 - Real.sqrt 131) / 4)
        (h₁ : 2 * x ^ 2 - 13 * x + k = 0) : k = 19 / 4 :="""),
    dedent("""
    ## Statement in natural language:
    The greatest common divisor (GCD) of 20 factorial (20!) and 200,000 is equal to 40,000.
    ## Formalized:
    theorem mathd_algebra_116 (k x : ℝ) (h₀ : x = (13 - Real.sqrt 131) / 4)
        (h₁ : 2 * x ^ 2 - 13 * x + k = 0) : k = 19 / 4 :="""),
    dedent("""
    ## Statement in natural language:
    Given two integers x and y:
    If y is positive (greater than 0),
    And y is less than x,
    And the equation x + y + xy = 80 is true,
    Then x must be equal to 26.
    ## Formalized:
    theorem mathd_algebra_116 (k x : ℝ) (h₀ : x = (13 - Real.sqrt 131) / 4)
        (h₁ : 2 * x ^ 2 - 13 * x + k = 0) : k = 19 / 4 :="""),
]


with Pipeline(name="test_deepseek_prover") as pipeline:
    data_loader = LoadDataFromHub(
        repo_id="plaguss/informal-mathematical-statements-tiny",
        split="val",
        batch_size=8,
    )

    llm = InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3-70B-Instruct",
    )
    auto_formalization = DeepSeekProverAutoFormalization(
        name="auto_formalization", input_batch_size=8, llm=llm, examples=examples
    )
    prover_scorer = DeepSeekProverScorer(
        name="prover_scorer",
        input_batch_size=8,
        llm=llm,
    )
    proof_generator = DeepSeekProverSolver(
        name="proof_generator", input_batch_size=8, llm=llm
    )

    (data_loader >> auto_formalization >> prover_scorer >> proof_generator)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        help="Do a dry run for testing purposes.",
    )
    args = parser.parse_args()

    pipeline_parameters = {
        data_loader.name: {"split": "val"},
        auto_formalization.name: {
            "llm": {
                "generation_kwargs": {
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "max_new_tokens": 512,
                }
            }
        },
        prover_scorer.name: {
            "llm": {
                "generation_kwargs": {
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "max_new_tokens": 512,
                }
            }
        },
    }

    ds_name = "test_deepseek_prover"

    if args.dry_run:
        distiset = pipeline.dry_run(batch_size=1, parameters=pipeline_parameters)
        distiset.save_to_disk(Path.home() / f"Downloads/{ds_name}")

        import pprint

        pprint.pprint(distiset["default"]["train"][0])

    else:
        distiset = pipeline.run(parameters=pipeline_parameters)
        distiset.push_to_hub(ds_name, include_script=True)
