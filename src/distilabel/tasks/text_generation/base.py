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

import random
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Union

from distilabel.tasks.base import Task
from distilabel.tasks.prompt import Prompt
from distilabel.tasks.text_generation.principles import UltraFeedbackPrinciples


@dataclass
class TextGenerationTask(Task):
    """A base `Task` definition for text generation using LLMs.

    Args:
        system_prompt (str, optional): the system prompt to be used. Defaults to `None`.
        principles (Dict[str, List[str]], optional): the principles to be used for the system prompt.
            Defaults to `None`.
        principles_distribution (Union[Dict[str, float], Literal["balanced"], None], optional): the
            distribution of principles to be used for the system prompt. Defaults to `None`.

    Examples:
        >>> from distilabel.tasks.text_generation import TextGenerationTask
        >>> task = TextGenerationTask()
    """

    system_prompt: str = (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible,"
        " while being safe. Your answers should not include any harmful, unethical, racist, sexist,"
        " toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased"
        " and positive in nature.\nIf a question does not make any sense, or is not factually coherent,"
        " explain why instead of answering something not correct. If you don't know the answer to a"
        " question, please don't share false information."
    )
    principles: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "harmlessness": UltraFeedbackPrinciples.harmlessness,
            "helpfulness": UltraFeedbackPrinciples.helpfulness,
            "truthfulness": UltraFeedbackPrinciples.truthfulness,
            "honesty": UltraFeedbackPrinciples.honesty,
            "verbalized_calibration": UltraFeedbackPrinciples.verbalized_calibration,
        },
        repr=False,
    )
    principles_distribution: Union[Dict[str, float], Literal["balanced"], None] = None

    def __post_init__(self) -> None:
        """Validates the `principles_distribution` if it is a dict.

        Raises:
            ValueError: if the `principles_distribution` is a dict and it does not sum to 1.0.
            ValueError: if the `principles` are not included in the `principles_distribution`.
        """
        if isinstance(self.principles_distribution, dict):
            not_included_principles = [
                principle
                for principle in self.principles
                if principle not in self.principles_distribution
            ]
            if not_included_principles:
                principles_str = ", ".join(
                    [f"'{principle}'" for principle in not_included_principles]
                )
                raise ValueError(
                    f"Principles {principles_str} included in `principles` is not in"
                    " `principles_distribution`"
                )

            if sum(self.principles_distribution.values()) != 1.0:
                raise ValueError(
                    "`principles_distribution` must sum to 1.0 if it is a dict containing"
                    " the distribution of principles to use."
                )

    def _get_principle(self) -> str:
        """Gets a principle from the `principles` dict respecting the `principal_distribution`.

        Returns:
            str: the principle to be used.
        """
        if isinstance(self.principles_distribution, dict):
            principle_group = random.choices(
                list(self.principles_distribution.keys()),
                weights=list(self.principles_distribution.values()),
                k=1,
            )[0]
        else:
            principle_group = random.choice(list(self.principles.keys()))
        return random.choice(self.principles[principle_group])

    def generate_prompt(self, input: str) -> Prompt:
        """Generates the prompt to be used for generation.

        Args:
            input (str): the input to be used for generation.

        Returns:
            Prompt: the generated prompt.

        Examples:
            >>> from distilabel.tasks.text_generation import TextGenerationTask
            >>> task = TextGenerationTask(system_prompt="You are a helpful assistant.")
            >>> task.generate_prompt("What are the first 5 Fibonacci numbers?")
            Prompt(system_prompt='You are a helpful assistant.', formatted_prompt='What are the first 5 Fibonacci numbers?')
        """
        system_prompt = self.system_prompt
        if self.principles_distribution is not None:
            principle = self._get_principle()
            system_prompt += " " + principle
        return Prompt(system_prompt=system_prompt, formatted_prompt=input)

    def parse_output(self, output: str) -> dict[str, str]:
        """Parses the output of the LLM into the desired format."""
        return {"generations": output}

    @property
    def input_args_names(self) -> list[str]:
        """Returns the input args names for the task."""
        return ["input"]

    @property
    def output_args_names(self) -> list[str]:
        """Returns the output args names for the task."""
        return ["generations"]
