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


def parse_steps(example_string: str) -> list[str]:
    # Extract content inside the first pair of triple backticks
    content_match = re.search(r"```(.*)```", example_string, re.DOTALL)
    if content_match:
        example_string = content_match.group(1).strip()

    # Regular expression to match step instructions
    step_regex = re.compile(
        r"Step (\d+):\s*(?:#([^#]+)#)?\s*(.*?)(?=Step \d+:|$)", re.DOTALL
    )

    steps_list = []
    for match in step_regex.finditer(example_string):
        step_number = int(match.group(1))
        step_name = match.group(2).strip() if match.group(2) else ""
        step_instruction = match.group(3).strip()

        step_dict = {
            "step_number": step_number,
            "step_name": step_name,
            "step_instruction": step_instruction,
        }
        steps_list.append(step_dict)

    return steps_list


def remove_fences(text: str) -> str:
    pattern = r"^```([\s\S]*)\n```$"
    match = re.match(pattern, text, re.MULTILINE)
    if match:
        return match.group(1)
    return text
