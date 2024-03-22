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

from typing import TYPE_CHECKING, Any, Dict, List, Tuple, TypeVar, Union

from pydantic import Field
from typing_extensions import Annotated, get_args, get_origin

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

_T = TypeVar("_T")
_RUNTIME_PARAMETER_ANNOTATION = "distilabel_step_runtime_parameter"
RuntimeParameter = Annotated[
    Union[_T, None], Field(default=None), _RUNTIME_PARAMETER_ANNOTATION
]
"""Used to mark the attributes of a `Step` as a runtime parameter."""


class RuntimeParametersMixin:
    """Mixin for classes that have `RuntimeParameter`s attributes. Classes inheriting from
    this mixin must inherit from `pydantic.BaseModel` too."""

    @property
    def runtime_parameters_names(self) -> Dict[str, bool]:
        """Returns a dictionary containing the name of the runtime parameters of the class
        as keys and whether the parameter is required or not as values.

        Returns:
            A dictionary containing the name of the runtime parameters of the class as keys
            and whether the parameter is required or not as values.
        """

        runtime_parameters = {}

        for name, info in self.model_fields.items():  # type: ignore
            is_runtime_param, is_optional = _is_runtime_parameter(info)
            if is_runtime_param:
                runtime_parameters[name] = is_optional

        return runtime_parameters

    def get_runtime_parameters_info(self) -> List[Dict[str, Any]]:
        """Gets the information of the runtime parameters of the class such as the name and
        the description. This function is meant to include the information of the runtime
        parameters in the serialized data of the class.

        Returns:
            A list containing the information for each runtime parameter of the class.
        """
        runtime_parameters_info = []
        for name, field_info in self.model_fields.items():  # type: ignore
            if name in self.runtime_parameters_names:
                info = {"name": name, "optional": self.runtime_parameters_names[name]}
                if field_info.description is not None:
                    info["description"] = field_info.description
                runtime_parameters_info.append(info)
        return runtime_parameters_info


def _is_runtime_parameter(field: "FieldInfo") -> Tuple[bool, bool]:
    """Check if a `pydantic.BaseModel` field is a `RuntimeParameter` and if it's optional
    i.e. providing a value for the field in `Pipeline.run` is optional.

    Args:
        field: The info of the field of the `pydantic.BaseModel` to check.

    Returns:
        A tuple with two booleans. The first one indicates if the field is a
        `RuntimeParameter` or not, and the second one indicates if the field is optional
        or not.
    """
    # Case 1: `runtime_param: RuntimeParameter[int]`
    # Mandatory runtime parameter that needs to be provided when running the pipeline
    if _RUNTIME_PARAMETER_ANNOTATION in field.metadata:
        return True, field.default is not None

    # Case 2: `runtime_param: Union[RuntimeParameter[int], None] = None`
    # Optional runtime parameter that doesn't need to be provided when running the pipeline
    type_args = get_args(field.annotation)
    for arg in type_args:
        is_runtime_param = (
            get_origin(arg) is Annotated
            and get_args(arg)[-1] == _RUNTIME_PARAMETER_ANNOTATION
        )
        if is_runtime_param:
            is_optional = (
                get_origin(field.annotation) is Union and type(None) in type_args
            )
            return True, is_optional

    return False, False
