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
    GenerationMutationTemplates as EvolInstructionGenerationMutationTemplates,
)
from distilabel.steps.task.evol_instruct.utils import (
    MutationTemplates as EvolInstructionMutationTemplates,
)

if sys.version_info < (3, 11):
    from enum import Enum as StrEnum
else:
    from enum import StrEnum


class MutationTemplates(StrEnum):
    ADD_CONSTRAINTS = EvolInstructionMutationTemplates.ADD_CONSTRAINTS.value
    DEEPEN = EvolInstructionMutationTemplates.DEEPEN.value
    CONCRETIZE = EvolInstructionMutationTemplates.CONCRETIZE.value
    INCREASE_REASONING = EvolInstructionMutationTemplates.INCREASE_REASONING.value


class GenerationMutationTemplates(StrEnum):
    FRESH_START = EvolInstructionGenerationMutationTemplates.FRESH_START.value
    ADD_CONSTRAINTS = EvolInstructionGenerationMutationTemplates.ADD_CONSTRAINTS.value
    DEEPEN = EvolInstructionGenerationMutationTemplates.DEEPEN.value
    CONCRETIZE = EvolInstructionGenerationMutationTemplates.CONCRETIZE.value
    INCREASE_REASONING = (
        EvolInstructionGenerationMutationTemplates.INCREASE_REASONING.value
    )
