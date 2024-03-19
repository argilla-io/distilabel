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

from pydantic import Field

from distilabel.steps.task.evol_instruct.base import EvolInstruct
from distilabel.steps.task.evol_instruct.evol_complexity.utils import (
    MutationTemplatesEvolComplexity,
)

if sys.version_info < (3, 11):
    from enum import EnumMeta as EnumType
else:
    from enum import EnumType


class EvolComplexity(EvolInstruct):
    """
    What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning
    and
    WizardLM: Empowering Large Language Models to Follow Complex Instructions

    Reference:

        - https://arxiv.org/abs/2312.15685
        - https://arxiv.org/abs/2304.12244

    Runtime parameters:

    - `seed`: The number of evolutions to be run.

    Columns:

    - `input`: instruction
    - `output`: there's multiple scenarios:
        - `store_evolutions=False`, `generate_answers=False` -> (evolved_instruction, model_name)
        - `store_evolutions=True`, `generate_answers=False` -> (evolved_instructions, model_name)
        - `store_evolutions=False`, `generate_answers=True` -> (evolved_instruction, model_name, answer)
        - `store_evolutions=True`, `generate_answers=True` -> (evolved_instructions, model_name, answer)
    """

    mutation_templates: EnumType = Field(default=MutationTemplatesEvolComplexity)
