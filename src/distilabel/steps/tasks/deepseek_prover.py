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
from typing import Any, Dict, List, Union

from typing_extensions import override

from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.typing import ChatType

_PARSE_DEEPSEEK_PROVER_AUTOFORMAL_REGEX = r"```lean4(.*?)```"


class DeepSeekProverAutoFormalization(Task):
    """Task to translate a mathematical problem from natural language to Lean 4.

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
    """

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
                "content": "Translate the problem to Lean 4 (only the core declaration):\n```lean4\nformal statement goes here\n```",
            },
            {
                "role": "user",
                "content": f"Mathematical Problem in Natural Language:\n{input['informal_statement']}",
            },
        ]

    @override
    def format_output(  # type: ignore
        self, output: Union[str, None], input: Dict[str, Any] = None
    ) -> Dict[str, Any]:  # type: ignore
        match = re.match(_PARSE_DEEPSEEK_PROVER_AUTOFORMAL_REGEX, output, re.DOTALL)
        if match:
            match = match.group(1).strip()
        return {"formal_statement": match}
