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

import hashlib
from typing import TYPE_CHECKING, List

from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class FormatTextGenerationSFT(Step):
    # The formatting presented below will follow the standards defined within the main
    # LLM fine-tuning libraries / frameworks being `axolotl`, and `alignment-handbook`
    # i.e. `trl`

    @property
    def inputs(self) -> List[str]:
        return ["instruction", "generation"]

    @property
    def optional_inputs(self) -> List[str]:
        # Here for the sake of keeping things consistent, while not obscuring any input
        return ["system_prompt"]

    @property
    def outputs(self) -> List[str]:
        # Output format inspired in https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
        return ["prompt", "prompt_id", "messages"]

    def process(self, *inputs: StepInput) -> "StepOutput":  # type: ignore
        for input in inputs:
            for item in input:
                item["prompt"] = item["instruction"]

                item["prompt_id"] = hashlib.sha256(
                    item["prompt"].encode("utf-8")  # type: ignore
                ).hexdigest()

                item["messages"] = [
                    {"role": "user", "content": item["instruction"]},  # type: ignore
                    {"role": "assistant", "content": item["generation"]},  # type: ignore
                ]
                if (
                    "system_prompt" in item
                    and isinstance(item["system_prompt"], str)  # type: ignore
                    and len(item["system_prompt"]) > 0  # type: ignore
                ):
                    item["messages"].insert(
                        0,
                        {"role": "system", "content": item["system_prompt"]},  # type: ignore
                    )

            yield input


class FormatChatGenerationSFT(Step):
    # The formatting presented below will follow the standards defined within the main
    # LLM fine-tuning libraries / frameworks being `axolotl`, and `alignment-handbook`
    # i.e. `trl`

    @property
    def inputs(self) -> List[str]:
        return ["messages", "generation"]

    @property
    def outputs(self) -> List[str]:
        # Output format inspired in https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
        return ["prompt", "prompt_id", "messages"]

    def process(self, *inputs: StepInput) -> "StepOutput":  # type: ignore
        for input in inputs:
            for item in input:
                item["prompt"] = next(
                    (
                        turn["content"]
                        for turn in item["messages"]
                        if turn["role"] == "user"
                    ),
                    None,
                )

                item["prompt_id"] = hashlib.sha256(
                    item["prompt"].encode("utf-8")  # type: ignore
                ).hexdigest()

                item["messages"] = item["messages"] + [
                    {"role": "assistant", "content": item["generation"]},  # type: ignore
                ]

            yield input
