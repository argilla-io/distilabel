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
    from distilabel.steps.typing import StepColumns, StepOutput


class FormatTextGenerationDPO(Step):
    """Format the output of your LLMs for Direct Preference Optimization (DPO).

    `FormatTextGenerationDPO` is a `Step` that formats the output of the combination of a `TextGeneration`
    task with a preference `Task` i.e. a task generating `ratings`, so that those are used to rank the
    existing generations and provide the `chosen` and `rejected` generations based on the `ratings`.
    Use this step to transform the output of a combination of a `TextGeneration` + a preference task such as
    `UltraFeedback` following the standard formatting from frameworks such as `axolotl` or `alignment-handbook`.

    Note:
        The `generations` column should contain at least two generations, the `ratings` column should
        contain the same number of ratings as generations.

    Input columns:
        - system_prompt (`str`, optional): The system prompt used within the `LLM` to generate the
            `generations`, if available.
        - instruction (`str`): The instruction used to generate the `generations` with the `LLM`.
        - generations (`List[str]`): The generations produced by the `LLM`.
        - generation_models (`List[str]`, optional): The model names used to generate the `generations`,
            only available if the `model_name` from the `TextGeneration` task/s is combined into a single
            column named this way, otherwise, it will be ignored.
        - ratings (`List[float]`): The ratings for each of the `generations`, produced by a preference
            task such as `UltraFeedback`.

    Output columns:
        - prompt (`str`): The instruction used to generate the `generations` with the `LLM`.
        - prompt_id (`str`): The `SHA256` hash of the `prompt`.
        - chosen (`List[Dict[str, str]]`): The `chosen` generation based on the `ratings`.
        - chosen_model (`str`, optional): The model name used to generate the `chosen` generation,
            if the `generation_models` are available.
        - chosen_rating (`float`): The rating of the `chosen` generation.
        - rejected (`List[Dict[str, str]]`): The `rejected` generation based on the `ratings`.
        - rejected_model (`str`, optional): The model name used to generate the `rejected` generation,
            if the `generation_models` are available.
        - rejected_rating (`float`): The rating of the `rejected` generation.

    Categories:
        - format
        - text-generation
        - preference
        - instruction
        - generations

    Examples:
        Format your dataset for DPO fine tuning:

        ```python
        from distilabel.steps import FormatTextGenerationDPO

        format_dpo = FormatTextGenerationDPO()
        format_dpo.load()

        # NOTE: Both "system_prompt" and "generation_models" can be added optionally.
        result = next(
            format_dpo.process(
                [
                    {
                        "instruction": "What's 2+2?",
                        "generations": ["4", "5", "6"],
                        "ratings": [1, 0, -1],
                    }
                ]
            )
        )
        # >>> result
        # [
        #    {   'instruction': "What's 2+2?",
        #        'generations': ['4', '5', '6'],
        #        'ratings': [1, 0, -1],
        #        'prompt': "What's 2+2?",
        #        'prompt_id': '7762ecf17ad41479767061a8f4a7bfa3b63d371672af5180872f9b82b4cd4e29',
        #        'chosen': [{'role': 'user', 'content': "What's 2+2?"}, {'role': 'assistant', 'content': '4'}],
        #        'chosen_rating': 1,
        #        'rejected': [{'role': 'user', 'content': "What's 2+2?"}, {'role': 'assistant', 'content': '6'}],
        #        'rejected_rating': -1
        #    }
        # ]
        ```
    """

    @property
    def inputs(self) -> "StepColumns":
        """List of inputs required by the `Step`, which in this case are: `instruction`, `generations`,
        and `ratings`."""
        return {
            "system_prompt": False,
            "instruction": True,
            "generations": True,
            "generation_models": False,
            "ratings": True,
        }

    @property
    def optional_inputs(self) -> List[str]:
        """List of optional inputs, which are not required by the `Step` but used if available,
        which in this case are: `system_prompt`, and `generation_models`."""
        return ["system_prompt", "generation_models"]

    @property
    def outputs(self) -> "StepColumns":
        """List of outputs generated by the `Step`, which are: `prompt`, `prompt_id`, `chosen`,
        `chosen_model`, `chosen_rating`, `rejected`, `rejected_model`, `rejected_rating`. Both
        the `chosen_model` and `rejected_model` being optional and only used if `generation_models`
        is available.

        Reference:
            - Format inspired in https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
        """
        return [
            "prompt",
            "prompt_id",
            "chosen",
            "chosen_model",
            "chosen_rating",
            "rejected",
            "rejected_model",
            "rejected_rating",
        ]

    def process(self, *inputs: StepInput) -> "StepOutput":  # type: ignore
        """The `process` method formats the received `StepInput` or list of `StepInput`
        according to the DPO formatting standard.

        Args:
            *inputs: A list of `StepInput` to be combined.

        Yields:
            A `StepOutput` with batches of formatted `StepInput` following the DPO standard.
        """
        for input in inputs:
            for item in input:
                messages = [
                    {"role": "user", "content": item["instruction"]},  # type: ignore
                ]
                if (
                    "system_prompt" in item
                    and isinstance(item["system_prompt"], str)  # type: ignore
                    and len(item["system_prompt"]) > 0  # type: ignore
                ):
                    messages.insert(
                        0,
                        {"role": "system", "content": item["system_prompt"]},  # type: ignore
                    )

                item["prompt"] = item["instruction"]
                item["prompt_id"] = hashlib.sha256(
                    item["prompt"].encode("utf-8")  # type: ignore
                ).hexdigest()

                chosen_idx = max(enumerate(item["ratings"]), key=lambda x: x[1])[0]
                item["chosen"] = messages + [
                    {
                        "role": "assistant",
                        "content": item["generations"][chosen_idx],
                    }
                ]
                if "generation_models" in item:
                    item["chosen_model"] = item["generation_models"][chosen_idx]
                item["chosen_rating"] = item["ratings"][chosen_idx]

                rejected_idx = min(enumerate(item["ratings"]), key=lambda x: x[1])[0]
                item["rejected"] = messages + [
                    {
                        "role": "assistant",
                        "content": item["generations"][rejected_idx],
                    }
                ]
                if "generation_models" in item:
                    item["rejected_model"] = item["generation_models"][rejected_idx]
                item["rejected_rating"] = item["ratings"][rejected_idx]

            yield input


class FormatChatGenerationDPO(Step):
    """Format the output of a combination of a `ChatGeneration` + a preference task for Direct Preference Optimization (DPO).

    `FormatChatGenerationDPO` is a `Step` that formats the output of the combination of a `ChatGeneration`
    task with a preference `Task` i.e. a task generating `ratings` such as `UltraFeedback` following the standard
    formatting from frameworks such as `axolotl` or `alignment-handbook`., so that those are used to rank the
    existing generations and provide the `chosen` and `rejected` generations based on the `ratings`.

    Note:
        The `messages` column should contain at least one message from the user, the `generations`
        column should contain at least two generations, the `ratings` column should contain the same
        number of ratings as generations.

    Input columns:
        - messages (`List[Dict[str, str]]`): The conversation messages.
        - generations (`List[str]`): The generations produced by the `LLM`.
        - generation_models (`List[str]`, optional): The model names used to generate the `generations`,
            only available if the `model_name` from the `ChatGeneration` task/s is combined into a single
            column named this way, otherwise, it will be ignored.
        - ratings (`List[float]`): The ratings for each of the `generations`, produced by a preference
            task such as `UltraFeedback`.

    Output columns:
        - prompt (`str`): The user message used to generate the `generations` with the `LLM`.
        - prompt_id (`str`): The `SHA256` hash of the `prompt`.
        - chosen (`List[Dict[str, str]]`): The `chosen` generation based on the `ratings`.
        - chosen_model (`str`, optional): The model name used to generate the `chosen` generation,
            if the `generation_models` are available.
        - chosen_rating (`float`): The rating of the `chosen` generation.
        - rejected (`List[Dict[str, str]]`): The `rejected` generation based on the `ratings`.
        - rejected_model (`str`, optional): The model name used to generate the `rejected` generation,
            if the `generation_models` are available.
        - rejected_rating (`float`): The rating of the `rejected` generation.

    Categories:
        - format
        - chat-generation
        - preference
        - messages
        - generations

    Examples:
        Format your dataset for DPO fine tuning:

        ```python
        from distilabel.steps import FormatChatGenerationDPO

        format_dpo = FormatChatGenerationDPO()
        format_dpo.load()

        # NOTE: "generation_models" can be added optionally.
        result = next(
            format_dpo.process(
                [
                    {
                        "messages": [{"role": "user", "content": "What's 2+2?"}],
                        "generations": ["4", "5", "6"],
                        "ratings": [1, 0, -1],
                    }
                ]
            )
        )
        # >>> result
        # [
        #     {
        #         'messages': [{'role': 'user', 'content': "What's 2+2?"}],
        #         'generations': ['4', '5', '6'],
        #         'ratings': [1, 0, -1],
        #         'prompt': "What's 2+2?",
        #         'prompt_id': '7762ecf17ad41479767061a8f4a7bfa3b63d371672af5180872f9b82b4cd4e29',
        #         'chosen': [{'role': 'user', 'content': "What's 2+2?"}, {'role': 'assistant', 'content': '4'}],
        #         'chosen_rating': 1,
        #         'rejected': [{'role': 'user', 'content': "What's 2+2?"}, {'role': 'assistant', 'content': '6'}],
        #         'rejected_rating': -1
        #     }
        # ]
        ```
    """

    @property
    def inputs(self) -> "StepColumns":
        """List of inputs required by the `Step`, which in this case are: `messages`, `generations`,
        and `ratings`."""
        return ["messages", "generations", "ratings"]

    @property
    def optional_inputs(self) -> List[str]:
        """List of optional inputs, which are not required by the `Step` but used if available,
        which in this case is: `generation_models`."""
        return ["generation_models"]

    @property
    def outputs(self) -> "StepColumns":
        """List of outputs generated by the `Step`, which are: `prompt`, `prompt_id`, `chosen`,
        `chosen_model`, `chosen_rating`, `rejected`, `rejected_model`, `rejected_rating`. Both
        the `chosen_model` and `rejected_model` being optional and only used if `generation_models`
        is available.

        Reference:
            - Format inspired in https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
        """
        return [
            "prompt",
            "prompt_id",
            "chosen",
            "chosen_model",
            "chosen_rating",
            "rejected",
            "rejected_model",
            "rejected_rating",
        ]

    def process(self, *inputs: StepInput) -> "StepOutput":  # type: ignore
        """The `process` method formats the received `StepInput` or list of `StepInput`
        according to the DPO formatting standard.

        Args:
            *inputs: A list of `StepInput` to be combined.

        Yields:
            A `StepOutput` with batches of formatted `StepInput` following the DPO standard.
        """
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

                chosen_idx = max(enumerate(item["ratings"]), key=lambda x: x[1])[0]
                item["chosen"] = item["messages"] + [
                    {
                        "role": "assistant",
                        "content": item["generations"][chosen_idx],
                    }
                ]
                if "generation_models" in item:
                    item["chosen_model"] = item["generation_models"][chosen_idx]
                item["chosen_rating"] = item["ratings"][chosen_idx]

                rejected_idx = min(enumerate(item["ratings"]), key=lambda x: x[1])[0]
                item["rejected"] = item["messages"] + [
                    {
                        "role": "assistant",
                        "content": item["generations"][rejected_idx],
                    }
                ]
                if "generation_models" in item:
                    item["rejected_model"] = item["generation_models"][rejected_idx]
                item["rejected_rating"] = item["ratings"][rejected_idx]

            yield input
