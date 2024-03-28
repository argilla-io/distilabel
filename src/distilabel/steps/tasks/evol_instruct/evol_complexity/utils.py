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

from distilabel.steps.tasks.evol_instruct.utils import (
    GENERATION_MUTATION_TEMPLATES as GENERATION_MUTATION_TEMPLATES_EVOL_INSTRUCT,
)
from distilabel.steps.tasks.evol_instruct.utils import (
    MUTATION_TEMPLATES as MUTATION_TEMPLATES_EVOL_INSTRUCT,
)

MUTATION_TEMPLATES = {
    "CONSTRAINTS": MUTATION_TEMPLATES_EVOL_INSTRUCT["CONSTRAINTS"],
    "DEEPENING": MUTATION_TEMPLATES_EVOL_INSTRUCT["DEEPENING"],
    "CONCRETIZING": MUTATION_TEMPLATES_EVOL_INSTRUCT["CONCRETIZING"],
    "INCREASED_REASONING_STEPS": MUTATION_TEMPLATES_EVOL_INSTRUCT[
        "INCREASED_REASONING_STEPS"
    ],
}

GENERATION_MUTATION_TEMPLATES = {
    "FRESH_START": GENERATION_MUTATION_TEMPLATES_EVOL_INSTRUCT["FRESH_START"],
    "CONSTRAINTS": GENERATION_MUTATION_TEMPLATES_EVOL_INSTRUCT["CONSTRAINTS"],
    "DEEPENING": GENERATION_MUTATION_TEMPLATES_EVOL_INSTRUCT["DEEPENING"],
    "CONCRETIZING": GENERATION_MUTATION_TEMPLATES_EVOL_INSTRUCT["CONCRETIZING"],
    "INCREASED_REASONING_STEPS": GENERATION_MUTATION_TEMPLATES_EVOL_INSTRUCT[
        "INCREASED_REASONING_STEPS"
    ],
}
