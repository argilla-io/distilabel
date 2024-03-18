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

from typing import Any, Dict, Iterator, List, Tuple

StepOutput = Iterator[List[Dict[str, Any]]]
"""StepOutput is an alias of the typing `Iterator[List[Dict[str, Any]]]`"""

GeneratorStepOutput = Iterator[Tuple[List[Dict[str, Any]], bool]]
"""GeneratorStepOutput is an alias of the typing `Iterator[Tuple[List[Dict[str, Any]], bool]]`"""
