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

from typing import List, Optional

from pydantic import Field, SecretStr

from distilabel.mixins.runtime_parameters import (
    RuntimeParameter,
    RuntimeParametersMixin,
)


class DummyNestedClass(RuntimeParametersMixin):
    runtime_param1: RuntimeParameter[str] = Field(
        default=None, description="Runtime param 1"
    )
    runtime_param2: Optional[RuntimeParameter[str]] = Field(
        default=None, description="Runtime param 2"
    )


class DummyClass(RuntimeParametersMixin):
    nested_class: DummyNestedClass
    mixins_list: List[DummyNestedClass]

    runtime_param1: RuntimeParameter[SecretStr] = Field(
        default=None, description="Runtime param 1"
    )
    runtime_param2: Optional[RuntimeParameter[SecretStr]] = Field(
        default=None, description="Runtime param 2"
    )


class TestRuntimeParametersMixin:
    def test_runtime_parameters_names(self) -> None:
        dummy = DummyClass(
            nested_class=DummyNestedClass(),
            mixins_list=[DummyNestedClass(), DummyNestedClass(), DummyNestedClass()],
        )

        assert dummy.runtime_parameters_names == {
            "runtime_param1": False,
            "runtime_param2": True,
            "nested_class": {
                "runtime_param1": False,
                "runtime_param2": True,
            },
            "mixins_list": {
                "0": {
                    "runtime_param1": False,
                    "runtime_param2": True,
                },
                "1": {
                    "runtime_param1": False,
                    "runtime_param2": True,
                },
                "2": {
                    "runtime_param1": False,
                    "runtime_param2": True,
                },
            },
        }

    def test_get_runtime_parameters_info(self) -> None:
        dummy = DummyClass(
            nested_class=DummyNestedClass(),
            mixins_list=[DummyNestedClass(), DummyNestedClass(), DummyNestedClass()],
        )

        assert dummy.get_runtime_parameters_info() == [
            {
                "name": "nested_class",
                "runtime_parameters_info": [
                    {
                        "name": "runtime_param1",
                        "description": "Runtime param 1",
                        "optional": False,
                    },
                    {
                        "name": "runtime_param2",
                        "description": "Runtime param 2",
                        "optional": True,
                    },
                ],
            },
            {
                "name": "mixins_list",
                "runtime_parameters_info": {
                    "0": [
                        {
                            "name": "runtime_param1",
                            "description": "Runtime param 1",
                            "optional": False,
                        },
                        {
                            "name": "runtime_param2",
                            "description": "Runtime param 2",
                            "optional": True,
                        },
                    ],
                    "1": [
                        {
                            "name": "runtime_param1",
                            "description": "Runtime param 1",
                            "optional": False,
                        },
                        {
                            "name": "runtime_param2",
                            "description": "Runtime param 2",
                            "optional": True,
                        },
                    ],
                    "2": [
                        {
                            "name": "runtime_param1",
                            "description": "Runtime param 1",
                            "optional": False,
                        },
                        {
                            "name": "runtime_param2",
                            "description": "Runtime param 2",
                            "optional": True,
                        },
                    ],
                },
            },
            {
                "name": "runtime_param1",
                "description": "Runtime param 1",
                "optional": False,
            },
            {
                "name": "runtime_param2",
                "description": "Runtime param 2",
                "optional": True,
            },
        ]

    def test_set_runtime_parameters(self) -> None:
        dummy = DummyClass(
            nested_class=DummyNestedClass(),
            mixins_list=[DummyNestedClass(), DummyNestedClass(), DummyNestedClass()],
        )

        dummy.set_runtime_parameters(
            {
                "runtime_param1": "value1",
                "runtime_param2": "value2",
                "this_one_is_going_to_be_ignored": "for sure",
                "nested_class": {
                    "runtime_param1": "value3",
                    "runtime_param2": "value4",
                },
            }
        )

        assert dummy.runtime_param1.get_secret_value() == "value1"  # type: ignore
        assert dummy.runtime_param2.get_secret_value() == "value2"  # type: ignore
        assert dummy.nested_class.runtime_param1 == "value3"
        assert dummy.nested_class.runtime_param2 == "value4"
