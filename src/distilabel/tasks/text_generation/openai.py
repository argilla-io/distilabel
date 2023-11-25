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

from typing import TYPE_CHECKING, List

from distilabel.tasks.prompt import Prompt
from distilabel.tasks.text_generation.base import TextGenerationTask

if TYPE_CHECKING:
    from distilabel.tasks.prompt import ChatCompletion


class OpenAITextGenerationTask(TextGenerationTask):
    """A `TextGenerationTask` for any chat-completion OpenAI model.

    Args:
        system_prompt (str, optional): the system prompt to be used. Defaults to `None`.
        principles (Dict[str, List[str]], optional): the principles to be used for the system prompt.
            Defaults to `None`.
        principles_distribution (Union[Dict[str, float], Literal["balanced"], None], optional): the
            distribution of principles to be used for the system prompt. Defaults to `None`.
    """

    def generate_prompt(self, input: str) -> List["ChatCompletion"]:
        """Generates a prompt for any chat-completion OpenAI model.

        Args:
            input (str): the input to be used for the prompt.

        Returns:
            List[ChatCompletion]: the generated prompt.

        Examples:
            >>> from distilabel.tasks.text_generation import OpenAITextGenerationTask
            >>> task = OpenAITextGenerationTask(system_prompt="You are a helpful assistant.")
            >>> task.generate_prompt("What are the first 5 Fibonacci numbers?")
            [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'What are the first 5 Fibonacci numbers?'},
            ]
        """
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=input,
        ).format_as("openai")  # type: ignore
