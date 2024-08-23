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

import inspect

from typing_extensions import Annotated

from distilabel.utils.typing_ import is_parameter_annotated_with


def test_is_parameter_annotated_with() -> None:
    def dummy_function(arg: Annotated[int, "unit-test"], arg2: int) -> None:
        pass

    signature = inspect.signature(dummy_function)
    arg_parameter = signature.parameters["arg"]
    arg2_parameter = signature.parameters["arg2"]

    assert is_parameter_annotated_with(arg_parameter, "hello") is False
    assert is_parameter_annotated_with(arg_parameter, "unit-test") is True
    assert is_parameter_annotated_with(arg2_parameter, "unit-test") is False
