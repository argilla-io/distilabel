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

from distilabel.steps import GlobalStep, StepInput
from distilabel.steps.tasks import TextGeneration
from distilabel.steps.tasks.typing import ChatType


class ArenaHard(TextGeneration):
    @property
    def inputs(self) -> List[str]:
        return ["instruction", "generations"]

    def format_input(self, input: StepInput) -> ChatType:
        return [
            {
                "role": "system",
                "content": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\".",
            },
            {
                "role": "user",
                "content": f"<|User Prompt|>\n{input['instruction']}\n\n<|The Start of Assistant A's Answer|>\n{input['generations'][0]}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{input['generations'][1]}\n<|The End of Assistant B's Answer|>",
            },
        ]

    @property
    def outputs(self) -> List[str]:
        return ["evaluation", "result", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        if output is None:
            return {"evaluation": None, "result": None}
        pattern = re.compile(r"\[\[([AB<>=]+)\]\]")
        match = pattern.search(output)
        if match is None:
            return {"evaluation": output, "result": None}
        return {"evaluation": output, "result": match.group(1)}


class ArenaHardResults(GlobalStep):
    """Reference: https://github.com/lm-sys/arena-hard-auto/blob/main/show_result.py"""

    ...
