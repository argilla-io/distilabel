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

from distilabel.utils.docstring import parse_google_docstring


def test_parse_google_docstring() -> None:
    def dummy_function() -> None:
        """This is a dummy function.

        Args:
            dummy_arg1: The first dummy argument.
            dummy_arg2: The second dummy argument.
            dummy_arg3: The third dummy argument.

        Returns:
            A dummy return value.

        Raises:
            ValueError: If something goes wrong.
            NotImplementedError: If something goes wrong.
        """
        pass

    assert parse_google_docstring(dummy_function) == {
        "description": "This is a dummy function.",
        "args": {
            "dummy_arg1": "The first dummy argument.",
            "dummy_arg2": "The second dummy argument.",
            "dummy_arg3": "The third dummy argument.",
        },
        "returns": "A dummy return value.",
        "raises": {
            "ValueError": "If something goes wrong.",
            "NotImplementedError": "If something goes wrong.",
        },
    }
