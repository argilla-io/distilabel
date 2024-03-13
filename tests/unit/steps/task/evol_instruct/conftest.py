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

import pytest
from distilabel.steps.task.evol_instruct.base import (
    EvolComplexity,
    EvolInstruct,
)
from distilabel.steps.task.evol_instruct.generator import (
    EvolComplexityGenerator,
    EvolInstructGenerator,
)
from distilabel.steps.task.evol_instruct.utils import (
    GenerationMutationTemplatesEvolComplexity,
    GenerationMutationTemplatesEvolInstruct,
    MutationTemplatesEvolComplexity,
    MutationTemplatesEvolInstruct,
)


@pytest.fixture(
    params=[
        (EvolInstruct, MutationTemplatesEvolInstruct),
        (EvolComplexity, MutationTemplatesEvolComplexity),
    ]
)
def task_params_base(request):
    return request.param


@pytest.fixture(
    params=[
        (EvolInstructGenerator, GenerationMutationTemplatesEvolInstruct),
        (EvolComplexityGenerator, GenerationMutationTemplatesEvolComplexity),
    ]
)
def task_params_generator(request):
    return request.param


@pytest.fixture(params=[EvolInstruct, EvolComplexity])
def task_class_base(request):
    return request.param


@pytest.fixture(params=[EvolInstructGenerator, EvolComplexityGenerator])
def task_class_generator(request):
    return request.param
