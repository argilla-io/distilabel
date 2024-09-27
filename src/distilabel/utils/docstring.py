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
from typing import Callable, Dict, List, Tuple

from typing_extensions import TypedDict


class Docstring(TypedDict):
    short_description: str
    description: str
    attributes: Dict[str, str]
    args: Dict[str, str]
    returns: str
    raises: Dict[str, str]
    runtime_parameters: Dict[str, str]
    input_columns: Dict[str, Tuple[str, str]]
    output_columns: Dict[str, Tuple[str, str]]
    categories: List[str]
    icon: str
    references: Dict[str, str]
    examples: Dict[str, str]
    note: str
    citations: List[str]


def parse_google_docstring(func: Callable) -> Docstring:  # noqa: C901
    """Parses a Google-style docstring and returns a dictionary with its sections. It takes
    into account the peculiarities of the docstrings used in `distilabel`.

    Args:
        func: The function or class to parse the docstring from.

    Returns:
        A dictionary with the parsed docstring sections.
    """
    sections: Docstring = {
        "short_description": "",
        "description": "",
        "attributes": {},
        "args": {},
        "returns": "",
        "raises": {},
        "runtime_parameters": {},
        "input_columns": {},
        "output_columns": {},
        "categories": [],
        "icon": "",
        "references": {},
        "examples": {},
        "note": "",
        "citations": [],
    }

    if not func.__doc__:
        return sections

    docstring = func.__doc__.strip()

    # Define the section headers to recognize
    section_headers = [
        "Args",
        "Returns",
        "Raises",
        "Attributes",
        "Runtime parameters",
        "Input columns",
        "Output columns",
        "Categories",
        "Icon",
        "References",
        "Examples",
        "Note",
        "Citations",
    ]

    # Match section headers
    section_pattern = rf"(\s*{'|'.join(section_headers)}):\s*\n"

    # Extract the short description (first line) or identify if it starts with a section header
    first_line_end = docstring.find("\n")
    if first_line_end == -1 or re.match(section_pattern, docstring[first_line_end:]):
        sections["short_description"] = docstring.split("\n", 1)[0].strip()
        remaining_docstring = (
            docstring.split("\n", 1)[1].strip() if "\n" in docstring else ""
        )
    else:
        sections["short_description"] = docstring[:first_line_end].strip()
        remaining_docstring = docstring[first_line_end:].strip()

    # Split the docstring into sections
    parts = re.split(section_pattern, remaining_docstring)

    if parts[0].strip() and not re.match(section_pattern, f"\n{parts[0].strip()}\n"):
        sections["description"] = parts[0].strip()

    for i in range(1, len(parts), 2):
        section_name = parts[i].lower().replace(" ", "_")
        section_content = parts[i + 1].strip()
        if section_name in ("args", "raises", "attributes"):
            # Parse arguments, exceptions, or attributes into a dictionary
            items = re.findall(
                r"\s*(\w+):\s*(.*?)\s*(?=\n\s*\w+:\s*|\n\s*$|$)",
                section_content,
                re.DOTALL,
            )
            sections[section_name] = {
                item[0]: re.sub(r"[\t\n]+| {2,}", " ", item[1]).strip()
                for item in items
            }
        elif section_name == "runtime_parameters":
            # Parse runtime parameters into a dictionary
            items = re.findall(
                r"\s*-\s*`?(\w+)`?:\s*(.*?)\s*(?=\n\s*-\s*`?\w+`?:|$)",
                section_content,
                re.DOTALL,
            )
            sections[section_name] = {
                item[0]: re.sub(r"[\t\n]+| {2,}", " ", item[1]).strip()
                for item in items
            }
        elif section_name in ("input_columns", "output_columns"):
            items = re.findall(
                r"- (?P<name>\w+) \((?P<type>`[^`]+`|[^)]+)\): (?P<description>.+?)(?=\n\s*-\s|\Z)",
                section_content,
                re.DOTALL,
            )
            sections[section_name] = {
                item[0]: (
                    item[1],
                    re.sub(r"[\t\n]+| {2,}", " ", item[2]).strip(),
                )
                for item in items
            }
        elif section_name == "categories":
            # Parse categories as a list of strings without the "- " prefix
            sections[section_name] = [
                cat.replace("-", "", 1).strip()
                for cat in section_content.split("\n")
                if cat.strip()
            ]
        elif section_name == "icon":
            # Parse logo as a single string
            match = re.match(r"`([^`]+)`", section_content)
            if match:
                sections[section_name] = match.group(1)
        elif section_name == "references":
            # Parse references into a dictionary with the name and URL
            items = re.findall(r"\s*-\s*\[([^]]+)\]\(([^)]+)\)", section_content)
            sections[section_name] = {
                item[0].replace("`", ""): item[1] for item in items
            }
        elif section_name == "examples":
            # Parse examples into a dictionary
            example_items = re.findall(
                r"(\w[\w\s]*?):\s*\n\s*```python\n(.*?)\n\s*```",
                section_content,
                re.DOTALL,
            )
            sections[section_name] = {
                item[0].strip(): remove_leading_whitespaces(item[1].strip())
                for item in example_items
            }
        elif section_name == "note":
            sections[section_name] = remove_leading_whitespaces(section_content.strip())
        elif section_name == "citations":
            pattern_citations = r"```(.*?)```"
            citations = re.findall(
                pattern_citations, section_content.strip(), re.DOTALL
            )
            sections[section_name] = [
                remove_leading_whitespaces(citation).strip() for citation in citations
            ]
        else:
            sections[section_name] = section_content

    return sections


def remove_leading_whitespaces(text: str, num_spaces: int = 8) -> str:
    """Removes the specified leading whitespaces from each line of a given string.

    Args:
        text: the string from which the leading whitespaces has to be removed.
        num_spaces: the number of leading whitespaces to remove.

    Returns:
        The string with the leading whitespaces removed.
    """
    lines = text.split("\n")
    trimmed_lines = [
        line[num_spaces:] if line.startswith(" " * num_spaces) else line
        for line in lines
    ]
    return "\n".join(trimmed_lines)


def get_bibtex(ref: str) -> str:
    r"""Get the bibtex citation from an arxiv url.

    Args:
        ref: Url from the arxiv paper.

    Returns:
        The bibtex style citation.

    Examples:

        ```python
        cite = get_bibtex(r"https://arxiv.org/abs/2406.18518")
        @misc{other,
            title={Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing},
            author={Zhangchen Xu and Fengqing Jiang and Luyao Niu and Yuntian Deng and Radha Poovendran and Yejin Choi and Bill Yuchen Lin},
            year={2024},
            eprint={2406.08464},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        }
        ```
    """
    from urllib.parse import quote_plus

    import requests
    from bs4 import BeautifulSoup

    if not ref.startswith("https://arxiv.org"):
        raise ValueError(
            f"The url must start with of `https://arxiv.org`, but got: {ref}"
        )
    response: bytes = requests.get(
        rf"https://arxiv2bibtex.org/?q={quote_plus(ref)}&format=bibtex"
    )
    soup = BeautifulSoup(response.content.decode("utf-8"), "html.parser")
    textarea = soup.find("div", id="bibtex").find("textarea", class_="wikiinfo")
    bibtex_citation = textarea.get_text().lstrip()
    return bibtex_citation
