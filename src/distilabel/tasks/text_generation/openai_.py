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
#
# WARNING: THIS FILE NAME HAS BEEN PREPENDED WITH AN UNDERSCORE TO AVOID
# ANY POTENTIAL CONFLICT / COLLISSION WITH THE `openai` PYTHON PACKAGE.

from typing import TYPE_CHECKING, List

from distilabel.tasks.prompt import Prompt
from distilabel.tasks.text_generation.base import TextGenerationTask

if TYPE_CHECKING:
    from distilabel.tasks.prompt import ChatCompletion


class OpenAITextGenerationTask(TextGenerationTask):
    def generate_prompt(self, input: str) -> List["ChatCompletion"]:
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=input,
        ).format_as("openai")  # type: ignore
