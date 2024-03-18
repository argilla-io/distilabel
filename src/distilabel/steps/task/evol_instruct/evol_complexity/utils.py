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

from distilabel.steps.task.evol_instruct.utils import (
    GenerationMutationTemplatesEvolInstruct,
    MutationTemplatesEvolInstruct,
)

if sys.version_info < (3, 11):
    from enum import Enum as StrEnum
else:
    from enum import StrEnum


class MutationTemplatesEvolComplexity(StrEnum):
    CONSTRAINTS = MutationTemplatesEvolInstruct.CONSTRAINTS.value
    DEEPENING = MutationTemplatesEvolInstruct.DEEPENING.value
    CONCRETIZING = MutationTemplatesEvolInstruct.CONCRETIZING.value
    INCREASED_REASONING_STEPS = (
        MutationTemplatesEvolInstruct.INCREASED_REASONING_STEPS.value
    )


class GenerationMutationTemplatesEvolComplexity(StrEnum):
    FRESH_START = GenerationMutationTemplatesEvolInstruct.FRESH_START.value
    CONSTRAINTS = GenerationMutationTemplatesEvolInstruct.CONSTRAINTS.value
    DEEPENING = GenerationMutationTemplatesEvolInstruct.DEEPENING.value
    CONCRETIZING = GenerationMutationTemplatesEvolInstruct.CONCRETIZING.value
    INCREASED_REASONING_STEPS = (
        GenerationMutationTemplatesEvolInstruct.INCREASED_REASONING_STEPS.value
    )
