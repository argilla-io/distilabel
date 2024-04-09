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

from typing import Dict

from distilabel.steps.tasks.evol_instruct.base import EvolInstruct
from distilabel.steps.tasks.evol_instruct.evol_complexity.utils import (
    MUTATION_TEMPLATES,
)


class EvolComplexity(EvolInstruct):
    """EvolComplexity is a task that evolves instructions to make them more complex,
    and it is based in the EvolInstruct task, but using slight different prompts, but the
    exact same evolutionary approach.

    Attributes:
        num_instructions: The number of instructions to be generated.
        generate_answers: Whether to generate answers for the instructions or not. Defaults
            to `False`.
        mutation_templates: The mutation templates to be used for the generation of the
            instructions.
        min_length: Defines the length (in bytes) that the generated instruction needs to
            be higher than, to be considered valid. Defaults to `512`.
        max_length: Defines the length (in bytes) that the generated instruction needs to
            be lower than, to be considered valid. Defaults to `1024`.
        seed: The seed to be set for `numpy` in order to randomly pick a mutation method.
            Defaults to `42`.

    Runtime parameters:
        - `min_length`: Defines the length (in bytes) that the generated instruction needs to be higher than, to be considered valid.
        - `max_length`: Defines the length (in bytes) that the generated instruction needs to be lower than, to be considered valid.
        - `seed`: The number of evolutions to be run.

    Input columns:
        - instruction (`str`): The instruction to evolve.

    Output columns:
        - evolved_instruction (`str`): The evolved instruction.
        - answer (`str`, optional): The answer to the instruction if `generate_answers=True`.
        - model_name (`str`): The name of the LLM used to evolve the instructions.

    References:
        - [What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning](https://arxiv.org/abs/2312.15685)
        - [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)
    """

    mutation_templates: Dict[str, str] = MUTATION_TEMPLATES
