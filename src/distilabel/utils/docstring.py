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
from typing import Callable, Dict

from typing_extensions import TypedDict


class Docstring(TypedDict):
    description: str
    args: Dict[str, str]
    returns: str
    raises: Dict[str, str]


def parse_google_docstring(func: Callable) -> Docstring:
    """Parses the Google-style docstring of the given function into a more structured format.

    Parameters:
        func: The function whose docstring will be parsed.

    Returns:
        A dictionary with keys 'description', 'args', 'returns', and 'raises',
        with 'args' and 'raises' being dictionaries themselves, mapping parameter
        and exception names to their descriptions, respectively.
    """
    sections: Docstring = {"description": "", "args": {}, "returns": "", "raises": {}}

    if not func.__doc__:
        return sections

    docstring = func.__doc__
    sections = {"description": "", "args": {}, "returns": "", "raises": {}}

    # Split the docstring into sections
    parts = re.split(r"\n\s*(Args|Returns|Raises):\s*\n", docstring)

    sections["description"] = parts[0].strip()
    for i in range(1, len(parts), 2):
        section_name = parts[i].lower()
        section_content = parts[i + 1].strip()
        if section_name in ("args", "raises"):
            # Parse arguments or exceptions into a dictionary
            items = re.findall(
                r"\s*(\w+):\s*(.*?)\s*(?=\n\s*\w+:|$)", section_content, re.DOTALL
            )
            sections[section_name] = {
                item[0]: re.sub(r"[\t\n]+|[ ]{2,}", " ", item[1]).strip()
                for item in items
            }
        else:
            sections[section_name] = section_content

    return sections
