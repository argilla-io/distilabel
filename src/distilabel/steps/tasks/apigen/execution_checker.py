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

# - Try to import the function from a given module
# - If function, try to import it and run it
# - If fails, track the error message, and return it

from typing import TYPE_CHECKING

from distilabel.steps.base import Step

if TYPE_CHECKING:
    pass


class APIGenPythonChecker(Step):
    """
    Implements a CodeAgent?
    # TODO: Maybe the implementation from does the job here?
    # https://huggingface.co/docs/transformers/en/agents#code-agent

    # NOTE: In load() we may need to add 'pip install transformers[agents]'
    """

    pass
