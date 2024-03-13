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
    ADD_CONSTRAINTS = EvolInstructionMutationTemplates.ADD_CONSTRAINTS
    DEEPEN = EvolInstructionMutationTemplates.DEEPEN
    CONCRETIZE = EvolInstructionMutationTemplates.CONCRETIZE
    INCREASE_REASONING = EvolInstructionMutationTemplates.INCREASE_REASONING


class GenerationMutationTemplates(StrEnum):
    FRESH_START = EvolInstructionGenerationMutationTemplates.FRESH_START
    ADD_CONSTRAINTS = EvolInstructionGenerationMutationTemplates.ADD_CONSTRAINTS
    DEEPEN = EvolInstructionGenerationMutationTemplates.DEEPEN
    CONCRETIZE = EvolInstructionGenerationMutationTemplates.CONCRETIZE
    INCREASE_REASONING = EvolInstructionGenerationMutationTemplates.INCREASE_REASONING
