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

import sys

if sys.version_info < (3, 11):
    from enum import Enum as StrEnum
else:
    from enum import StrEnum


class MutationTemplatesEvolInstruct(StrEnum):
    COMPLICATE = "Rewrite #Given Prompt# to make it slightly more complicated, and create #New Prompt#.\n\n#Given Prompt#:\n<PROMPT>\n"
    ADD_CONSTRAINTS = "Add a few more constraints or requirements to #Given Prompt#, and create #New Prompt#.\n\n#Given Prompt#:\n<PROMPT>\n"
    DEEPEN = "Slightly increase the depth and breadth of #Given Prompt#, and create #New Prompt#.\n\n#Given Prompt#:\n<PROMPT>\n"
    CONCRETIZE = "Make #Given Prompt# slightly more concrete, and create #New Prompt#.\n#Given Prompt#:\n\n<PROMPT>\n"
    INCREASE_REASONING = "If #Given Prompt# can be solved with just a few simple thinking processes, rewrite it to explicitly request multi-step reasoning, and create #New Prompt#.\n\n#Given Prompt#:\n<PROMPT>\n"
    SWITCH_TOPIC = "Rewrite #Given Prompt# by switching the topic, keeping the domain and difficulty level similar, and create #New Prompt#.\n\n#Given Prompt#:\n<PROMPT>\n"


class GenerationMutationTemplatesEvolInstruct(StrEnum):
    FRESH_START = "Write one question or request containing one or more of the following words: <PROMPT>"
    COMPLICATE = "Rewrite #Given Prompt# to make it slightly more complicated, and create #New Prompt#.\n\n#Given Prompt#:\n<PROMPT>\n"
    ADD_CONSTRAINTS = "Add a few more constraints or requirements to #Given Prompt#, and create #New Prompt#.\n\n#Given Prompt#:\n<PROMPT>\n"
    DEEPEN = "Slightly increase the depth and breadth of #Given Prompt#, and create #New Prompt#.\n\n#Given Prompt#:\n<PROMPT>\n"
    CONCRETIZE = "Make #Given Prompt# slightly more concrete, and create #New Prompt#.\n#Given Prompt#:\n\n<PROMPT>\n"
    INCREASE_REASONING = "If #Given Prompt# can be solved with just a few simple thinking processes, rewrite it to explicitly request multi-step reasoning, and create #New Prompt#.\n\n#Given Prompt#:\n<PROMPT>\n"
    SWITCH_TOPIC = "Rewrite #Given Prompt# by switching the topic, keeping the domain and difficulty level similar, and create #New Prompt#.\n\n#Given Prompt#:\n<PROMPT>\n"


class MutationTemplatesEvolComplexity(StrEnum):
    ADD_CONSTRAINTS = MutationTemplatesEvolInstruct.ADD_CONSTRAINTS.value
    DEEPEN = MutationTemplatesEvolInstruct.DEEPEN.value
    CONCRETIZE = MutationTemplatesEvolInstruct.CONCRETIZE.value
    INCREASE_REASONING = MutationTemplatesEvolInstruct.INCREASE_REASONING.value


class GenerationMutationTemplatesEvolComplexity(StrEnum):
    FRESH_START = GenerationMutationTemplatesEvolInstruct.FRESH_START.value
    ADD_CONSTRAINTS = GenerationMutationTemplatesEvolInstruct.ADD_CONSTRAINTS.value
    DEEPEN = GenerationMutationTemplatesEvolInstruct.DEEPEN.value
    CONCRETIZE = GenerationMutationTemplatesEvolInstruct.CONCRETIZE.value
    INCREASE_REASONING = (
        GenerationMutationTemplatesEvolInstruct.INCREASE_REASONING.value
    )
