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
from typing import TYPE_CHECKING

from typing_extensions import override

from distilabel.steps import Step
from distilabel.steps.base import StepInput
from distilabel.utils.card.dataset_card import get_dataset_use_template

if TYPE_CHECKING:
    from distilabel.steps.typing import DatasetUse, StepColumns, StepOutput


class FormatAPO(Step):
    """Format the output of `CLAIR` task for Anchored Preference Optimization (APO).

    `FormatAPO` is a `Step` that formats the output of a `CLAIR` task for
    Anchored Preference Optimization (APO) following the standard formatting from `TRL`.

    Input columns:
        - prompt (`str`): The instruction used to generate the `generation` with the `LLM`.
        - response (`str`): The generation produced by the `LLM`.
        - revision (`str`): The revised text.

    Output columns:
        - prompt (`str`): The instruction used to generate the `generation` with the `LLM`.
        - chosen (`List[Dict[str, str]]`): The `chosen` generation based on the `ratings`.
        - rejected (`List[Dict[str, str]]`): The `rejected` generation based on the `ratings`.
        - prompt_id (`str`): The `SHA256` hash of the `prompt`.

    Categories:
        - format
        - preference
        - instruction
        - generation

    Examples:
        Format your dataset for APO fine tuning:

        ```python
        from distilabel.steps import FormatAPO

        formatter = FormatAPO()
        formatter.load()

        result = next(
            formatter.process(
                [
                    {
                        "prompt": "How many gaps are there between the earth and the moon?",
                        "response": '''There are no gaps between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon's orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range.\n\nSo, to summarize, there are no gaps between the Earth and the Moon. The Moon is simply a satellite that orbits the Earth, and its distance from our planet varies slightly due to the elliptical shape of its orbit.''',
                        "revision": '''There are no physical gaps or empty spaces between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a significant separation or gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon\'s orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range. This variation in distance is a result of the Moon\'s orbital path, not the presence of any gaps.\n\nIn summary, the Moon\'s orbit is continuous, with no intervening gaps, and its distance from the Earth varies due to the elliptical shape of its orbit.''',
                    }
                ]
            )
        )
        # >>> result
        # [{'prompt': 'How many gaps are there between the earth and the moon?',
        # 'response': 'There are no gaps between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon\'s orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range.\n\nSo, to summarize, there are no gaps between the Earth and the Moon. The Moon is simply a satellite that orbits the Earth, and its distance from our planet varies slightly due to the elliptical shape of its orbit.',
        # 'revision': 'There are no physical gaps or empty spaces between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a significant separation or gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon\'s orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range. This variation in distance is a result of the Moon\'s orbital path, not the presence of any gaps.\n\nIn summary, the Moon\'s orbit is continuous, with no intervening gaps, and its distance from the Earth varies due to the elliptical shape of its orbit.',
        # 'prompt_id': 'd5e8924f2856fe7180c0aef3ec186f7a421b2ba11551b9ebfffeb7638ec5b021',
        # 'chosen': [{'role': 'user',
        #     'content': 'How many gaps are there between the earth and the moon?'},
        # {'role': 'assistant',
        #     'content': 'There are no physical gaps or empty spaces between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a significant separation or gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon\'s orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range. This variation in distance is a result of the Moon\'s orbital path, not the presence of any gaps.\n\nIn summary, the Moon\'s orbit is continuous, with no intervening gaps, and its distance from the Earth varies due to the elliptical shape of its orbit.'}],
        # 'rejected': [{'role': 'user',
        #     'content': 'How many gaps are there between the earth and the moon?'},
        # {'role': 'assistant',
        #     'content': 'There are no gaps between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon\'s orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range.\n\nSo, to summarize, there are no gaps between the Earth and the Moon. The Moon is simply a satellite that orbits the Earth, and its distance from our planet varies slightly due to the elliptical shape of its orbit.'}]}]
        ```
    """

    @property
    def inputs(self) -> "StepColumns":
        return ["prompt", "response", "revision"]

    @property
    def outputs(self) -> "StepColumns":
        return ["prompt", "chosen", "rejected", "prompt_id"]

    def process(self, *inputs: StepInput) -> "StepOutput":  # type: ignore
        """The `process` method formats the received `StepInput` or list of `StepInput`
        according to the APO formatting standard (DPO with loss_type equal to apo_zero
        or apo_down in TRL).

        Args:
            *inputs: A list of `StepInput` to be combined.

        Yields:
            A `StepOutput` with batches of formatted `StepInput` following the APO standard.
        """
        for input in inputs:
            for item in input:
                messages = [
                    {"role": "user", "content": item["prompt"]},  # type: ignore
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

                item["prompt_id"] = hashlib.sha256(
                    item["prompt"].encode("utf-8")  # type: ignore
                ).hexdigest()

                item["chosen"] = messages + [
                    {
                        "role": "assistant",
                        "content": item["revision"],
                    }
                ]
                item["rejected"] = messages + [
                    {
                        "role": "assistant",
                        "content": item["response"],
                    }
                ]
            yield input

    @override
    def _dataset_use(self) -> "DatasetUse":
        with open(get_dataset_use_template("sft")) as f:
            template = f.read()

        return {
            "title": "Anchored Preference Optimization (APO)",
            "template": template,
            "variables": ["dataset_name"],
        }
