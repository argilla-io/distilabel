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

import textwrap

from distilabel.utils.docstring import parse_google_docstring


def test_parse_google_docstring() -> None:
    def dummy_function() -> None:
        """This is a dummy function.

        Args:
            dummy_arg1: The first dummy argument.
            dummy_arg2: The second dummy argument.
            dummy_arg3: The third dummy argument.

        Note:
            Some note.

        Returns:
            A dummy return value.

        Raises:
            ValueError: If something goes wrong.
            NotImplementedError: If something goes wrong.
        """
        pass

    assert parse_google_docstring(dummy_function) == {
        "short_description": "This is a dummy function.",
        "description": "",
        "args": {
            "dummy_arg1": "The first dummy argument.",
            "dummy_arg2": "The second dummy argument.",
            "dummy_arg3": "The third dummy argument.",
        },
        "attributes": {},
        "categories": [],
        "examples": {},
        "icon": "",
        "input_columns": {},
        "output_columns": {},
        "references": {},
        "runtime_parameters": {},
        "returns": "A dummy return value.",
        "raises": {
            "ValueError": "If something goes wrong.",
            "NotImplementedError": "If something goes wrong.",
        },
        "note": "Some note.",
        "citations": [],
    }


def test_parse_google_docstring_with_distilabel_peculiarities() -> None:
    class DummyClass:
        """This is a dummy function.

        And this is still a dummy function, but with a longer description.

        Note:
            Some note.

        Attributes:
            dummy_attr1: The first dummy attribute.
            dummy_attr2: The second dummy attribute.
            dummy_attr3: The third dummy attribute.

        Runtime parameters:
            - `runtime_param1`: The first runtime parameter.
            - `runtime_param2`: The second runtime parameter.
            - `runtime_param3`: The third runtime parameter.

        Input columns:
            - input_col1 (`str`): The first input column.
            - input_col2 (`int`): The second input column.

        Output columns:
            - output_col1 (`List[str]`): The first output column.
            - output_col2 (random text): The second output column.

        Categories:
            - category1
            - category2

        Icon:
            `icon_name`

        Examples:

            Example 1:

            ```python
            dummy_function()
            ```

            Example 2:
            ```python
            dummy_function()
            ```

        References:
            - [Argilla](https://argilla.io)

        Citations:

            ```
            @misc{xu2024magpie,
                title={Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing},
                author={Zhangchen Xu and Fengqing Jiang and Luyao Niu and Yuntian Deng and Radha Poovendran and Yejin Choi and Bill Yuchen Lin},
                year={2024},
                eprint={2406.08464},
                archivePrefix={arXiv},
                primaryClass={cs.CL}
            }
            ```

            ```
            @misc{new,
                title={Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing},
                author={Zhangchen Xu and Fengqing Jiang and Luyao Niu and Yuntian Deng and Radha Poovendran and Yejin Choi and Bill Yuchen Lin},
                year={2024},
                eprint={2406.08464},
                archivePrefix={arXiv},
                primaryClass={cs.CL}
            }
            ```

            ```
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

        pass

    assert parse_google_docstring(DummyClass) == {
        "short_description": "This is a dummy function.",
        "description": "And this is still a dummy function, but with a longer description.",
        "args": {},
        "attributes": {
            "dummy_attr1": "The first dummy attribute.",
            "dummy_attr2": "The second dummy attribute.",
            "dummy_attr3": "The third dummy attribute.",
        },
        "runtime_parameters": {
            "runtime_param1": "The first runtime parameter.",
            "runtime_param2": "The second runtime parameter.",
            "runtime_param3": "The third runtime parameter.",
        },
        "input_columns": {
            "input_col1": ("`str`", "The first input column."),
            "input_col2": ("`int`", "The second input column."),
        },
        "output_columns": {
            "output_col1": ("`List[str]`", "The first output column."),
            "output_col2": ("random text", "The second output column."),
        },
        "categories": ["category1", "category2"],
        "icon": "icon_name",
        "references": {
            "Argilla": "https://argilla.io",
        },
        "returns": "",
        "raises": {},
        "examples": {
            "Example 1": "dummy_function()",
            "Example 2": "dummy_function()",
        },
        "note": "Some note.",
        "citations": [
            textwrap.dedent(
                """\
                @misc{xu2024magpie,
                        title={Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing},
                        author={Zhangchen Xu and Fengqing Jiang and Luyao Niu and Yuntian Deng and Radha Poovendran and Yejin Choi and Bill Yuchen Lin},
                        year={2024},
                        eprint={2406.08464},
                        archivePrefix={arXiv},
                        primaryClass={cs.CL}
                    }
                """.rstrip()
            ),
            textwrap.dedent(
                """\
                @misc{new,
                        title={Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing},
                        author={Zhangchen Xu and Fengqing Jiang and Luyao Niu and Yuntian Deng and Radha Poovendran and Yejin Choi and Bill Yuchen Lin},
                        year={2024},
                        eprint={2406.08464},
                        archivePrefix={arXiv},
                        primaryClass={cs.CL}
                    }
                """.rstrip()
            ),
            textwrap.dedent(
                """\
                @misc{other,
                        title={Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing},
                        author={Zhangchen Xu and Fengqing Jiang and Luyao Niu and Yuntian Deng and Radha Poovendran and Yejin Choi and Bill Yuchen Lin},
                        year={2024},
                        eprint={2406.08464},
                        archivePrefix={arXiv},
                        primaryClass={cs.CL}
                    }
                """.rstrip()
            ),
        ],
    }
