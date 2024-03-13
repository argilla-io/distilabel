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

from distilabel.steps.task.evol_complexity.utils import MutationTemplates
from distilabel.steps.task.evol_instruct.generator import EvolInstructGenerator
from pydantic import Field

if sys.version_info < (3, 11):
    from enum import EnumMeta as EnumType
else:
    from enum import EnumType


class EvolComplexityGenerator(EvolInstructGenerator):
    """
    What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning
    and
    WizardLM: Empowering Large Language Models to Follow Complex Instructions
    Reference:
        - https://arxiv.org/abs/2312.15685
        - https://arxiv.org/abs/2304.12244
        - https://github.com/h2oai/h2o-wizardlm
    """

    mutation_templates: EnumType = Field(default=MutationTemplates)
