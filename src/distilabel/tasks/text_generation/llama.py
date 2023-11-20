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

from distilabel.tasks.prompt import Prompt
from distilabel.tasks.text_generation.base import TextGenerationTask


class Llama2TextGenerationTask(TextGenerationTask):
    """A `TextGenerationTask` for the Llama2 model.

    Args:
        system_prompt (str, optional): the system prompt to be used. Defaults to `None`.
        principles (Dict[str, List[str]], optional): the principles to be used for the system prompt.
            Defaults to `None`.
        principles_distribution (Union[Dict[str, float], Literal["balanced"], None], optional): the
            distribution of principles to be used for the system prompt. Defaults to `None`.
    """

    def generate_prompt(self, input: str) -> str:
        """Generates a prompt for the Llama2 model.

        Args:
            input (str): the input to be used for the prompt.

        Returns:
            str: the generated prompt.

        Examples:
            >>> from distilabel.tasks.text_generation import Llama2TextGenerationTask
            >>> task = Llama2TextGenerationTask(system_prompt="You are a helpful assistant.")
            >>> task.generate_prompt("What are the first 5 Fibonacci numbers?")
            '<s>[INST] <<SYS>>\nYou are a helpful assistant.<</SYS>>\n\nWhat are the first 5 Fibonacci numbers? [/INST]'
        """
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=input,
        ).format_as("llama2")  # type: ignore
