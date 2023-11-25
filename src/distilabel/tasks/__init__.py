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

from distilabel.tasks.base import Task
from distilabel.tasks.preference.judgelm import JudgeLMTask
from distilabel.tasks.preference.ultrafeedback import UltraFeedbackTask
from distilabel.tasks.preference.ultrajudge import UltraJudgeTask
from distilabel.tasks.prompt import Prompt
from distilabel.tasks.text_generation.base import TextGenerationTask
from distilabel.tasks.text_generation.llama import Llama2TextGenerationTask
from distilabel.tasks.text_generation.openai import OpenAITextGenerationTask
from distilabel.tasks.text_generation.self_instruct import SelfInstructTask

__all__ = [
    "Task",
    "JudgeLMTask",
    "UltraFeedbackTask",
    "UltraJudgeTask",
    "Prompt",
    "TextGenerationTask",
    "OpenAITextGenerationTask",
    "Llama2TextGenerationTask",
    "SelfInstructTask"
]
