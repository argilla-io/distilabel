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
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, TypeVar, Union

from pydantic import BaseModel, Field, PrivateAttr
from pydantic.types import _SecretField
from typing_extensions import Annotated, get_args, get_origin

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

_T = TypeVar("_T")
_RUNTIME_PARAMETER_ANNOTATION = "distilabel_step_runtime_parameter"
RuntimeParameter = Annotated[
    Union[_T, None], Field(default=None), _RUNTIME_PARAMETER_ANNOTATION
]
"""Used to mark the attributes of a `Step` as a runtime parameter."""

RuntimeParametersNames = Dict[str, Union[bool, "RuntimeParametersNames"]]


class RuntimeParametersMixin(BaseModel):
    """Mixin for classes that have `RuntimeParameter`s attributes.

    Attributes:
        _runtime_parameters: A dictionary containing the values of the runtime parameters
            of the class. This attribute is meant to be used internally and should not be
            accessed directly.
    """

    _runtime_parameters: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def runtime_parameters_names(self) -> RuntimeParametersNames:
        """Returns a dictionary containing the name of the runtime parameters of the class
        as keys and whether the parameter is required or not as values.

        Returns:
            A dictionary containing the name of the runtime parameters of the class as keys
            and whether the parameter is required or not as values.
        """

        runtime_parameters = {}

        for name, field_info in self.model_fields.items():  # type: ignore
            is_runtime_param, is_optional = _is_runtime_parameter(field_info)
            if is_runtime_param:
                runtime_parameters[name] = is_optional
                continue

            attr = getattr(self, name)
            if isinstance(attr, RuntimeParametersMixin):
                runtime_parameters[name] = attr.runtime_parameters_names

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
            if name not in self.runtime_parameters_names:
                continue

            attr = getattr(self, name)
            if isinstance(attr, RuntimeParametersMixin):
                runtime_parameters_info.append(
                    {
                        "name": name,
                        "runtime_parameters_info": attr.get_runtime_parameters_info(),
                    }
                )
                continue

            info = {"name": name, "optional": self.runtime_parameters_names[name]}
            if field_info.description is not None:
                info["description"] = field_info.description
            runtime_parameters_info.append(info)
        return runtime_parameters_info

    def set_runtime_parameters(self, runtime_parameters: Dict[str, Any]) -> None:
        """Sets the runtime parameters of the class using the provided values. If the attr
        to be set is a `RuntimeParametersMixin`, it will call `set_runtime_parameters` on
        the attr.

        Args:
            runtime_parameters: A dictionary containing the values of the runtime parameters
                to set.
        """
        for name, value in runtime_parameters.items():
            if name not in self.runtime_parameters_names:
                continue

            attr = getattr(self, name)
            if isinstance(attr, RuntimeParametersMixin):
                attr.set_runtime_parameters(value)
                self._runtime_parameters[name] = value
                continue

            # Handle settings values for `_SecretField`
            field_info = self.model_fields[name]
            inner_type = _extract_runtime_parameter_inner_type(field_info.annotation)
            if inspect.isclass(inner_type) and issubclass(inner_type, _SecretField):
                value = inner_type(value)

            setattr(self, name, value)
            self._runtime_parameters[name] = value


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


def _extract_runtime_parameter_inner_type(type_hint: Any) -> Any:
    """Extracts the inner type of a `RuntimeParameter` type hint.

    Args:
        type_hint: The type hint to extract the inner type from.

    Returns:
        The inner type of the `RuntimeParameter` type hint.
    """
    type_hint_args = get_args(type_hint)
    if get_origin(type_hint) is Annotated:
        return _extract_runtime_parameter_inner_type(type_hint_args[0])

    if get_origin(type_hint) is Union and type(None) in type_hint_args:
        return _extract_runtime_parameter_inner_type(type_hint_args[0])

    return type_hint
