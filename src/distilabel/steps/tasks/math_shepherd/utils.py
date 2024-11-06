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

import re


def split_solution_steps(text):
    """
    Split a step-by-step solution text into individual components.
    Returns a list of steps and the final answer.
    """
    # Pattern to match:
    # 1. Steps starting with "Step N:" and capturing all content until the next step or answer
    # 2. The final answer starting with "The answer is:"
    pattern = r"Step \d+:.*?(?=Step \d+:|The answer is:|$)|The answer is:.*"

    # Find all matches, strip whitespace
    matches = [match.strip() for match in re.findall(pattern, text, re.DOTALL)]

    return matches
