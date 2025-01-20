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

import difflib
import inspect
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, TypeVar, Union

from pydantic import BaseModel, Field, PrivateAttr
from typing_extensions import Annotated, get_args, get_origin

from distilabel.utils.docstring import parse_google_docstring
from distilabel.utils.typing_ import (
    extract_annotation_inner_type,
    is_type_pydantic_secret_field,
)

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

    from distilabel.utils.docstring import Docstring


_T = TypeVar("_T")
_RUNTIME_PARAMETER_ANNOTATION = "distilabel_step_runtime_parameter"
RuntimeParameter = Annotated[
    Union[_T, None], Field(default=None), _RUNTIME_PARAMETER_ANNOTATION
]
"""Used to mark the attributes of a `Step` as a runtime parameter."""

RuntimeParametersNames = Dict[str, Union[bool, "RuntimeParametersNames"]]
"""Alias for the names of the runtime parameters of a `Step`."""

RuntimeParameterInfo = Dict[str, Any]
"""Alias for the information of the runtime parameters of a `Step`."""


class RuntimeParametersMixin(BaseModel):
    """Mixin for classes that have `RuntimeParameter`s attributes.

    Attributes:
        _runtime_parameters: A dictionary containing the values of the runtime parameters
            of the class. This attribute is meant to be used internally and should not be
            accessed directly.
    """

    _runtime_parameters: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def runtime_parameters_names(self) -> "RuntimeParametersNames":
        """Returns a dictionary containing the name of the runtime parameters of the class
        as keys and whether the parameter is required or not as values.

        Returns:
            A dictionary containing the name of the runtime parameters of the class as keys
            and whether the parameter is required or not as values.
        """

        runtime_parameters = {}

        for name, field_info in self.model_fields.items():  # type: ignore
            # `field: RuntimeParameter[Any]` or `field: Optional[RuntimeParameter[Any]]`
            is_runtime_param, is_optional = _is_runtime_parameter(field_info)
            if is_runtime_param:
                runtime_parameters[name] = is_optional
                continue

            attr = getattr(self, name)

            # `field: RuntimeParametersMixin`
            if isinstance(attr, RuntimeParametersMixin):
                runtime_parameters[name] = attr.runtime_parameters_names

            # `field: List[RuntimeParametersMixin]`
            if (
                isinstance(attr, list)
                and attr
                and isinstance(attr[0], RuntimeParametersMixin)
            ):
                runtime_parameters[name] = {
                    str(i): item.runtime_parameters_names for i, item in enumerate(attr)
                }

        return runtime_parameters

    def get_runtime_parameters_info(self) -> List["RuntimeParameterInfo"]:
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

            # Get runtime parameters info for `RuntimeParametersMixin` field
            if isinstance(attr, RuntimeParametersMixin):
                runtime_parameters_info.append(
                    {
                        "name": name,
                        "runtime_parameters_info": attr.get_runtime_parameters_info(),
                    }
                )
                continue

            # Get runtime parameters info for `List[RuntimeParametersMixin]` field
            if isinstance(attr, list) and isinstance(attr[0], RuntimeParametersMixin):
                runtime_parameters_info.append(
                    {
                        "name": name,
                        "runtime_parameters_info": {
                            str(i): item.get_runtime_parameters_info()
                            for i, item in enumerate(attr)
                        },
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
        runtime_parameters_names = list(self.runtime_parameters_names.keys())
        for name, value in runtime_parameters.items():
            if name not in self.runtime_parameters_names:
                # Check done just to ensure the unit tests for the mixin run
                if getattr(self, "pipeline", None):
                    closest = difflib.get_close_matches(
                        name, runtime_parameters_names, cutoff=0.5
                    )
                    msg = (
                        f"⚠️  Runtime parameter '{name}' unknown in step '{self.name}'."  # type: ignore
                    )
                    if closest:
                        msg += f" Did you mean any of: {closest}"
                    else:
                        msg += f" Available runtime parameters for the step: {runtime_parameters_names}."
                    self.pipeline._logger.warning(msg)  # type: ignore
                continue

            attr = getattr(self, name)

            # Set runtime parameters for `RuntimeParametersMixin` field
            if isinstance(attr, RuntimeParametersMixin):
                attr.set_runtime_parameters(value)
                self._runtime_parameters[name] = value
                continue

            # Set runtime parameters for `List[RuntimeParametersMixin]` field
            if isinstance(attr, list) and isinstance(attr[0], RuntimeParametersMixin):
                for i, item in enumerate(attr):
                    item_value = value.get(str(i), {})
                    item.set_runtime_parameters(item_value)
                self._runtime_parameters[name] = value
                continue

            # Handle settings values for `_SecretField`
            field_info = self.model_fields[name]
            inner_type = extract_annotation_inner_type(field_info.annotation)
            if is_type_pydantic_secret_field(inner_type):
                value = inner_type(value)

            # Set the value of the runtime parameter
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


class RuntimeParametersModelMixin(RuntimeParametersMixin):
    """Specific mixin for RuntimeParameters that affect the model classes, LLM,
    ImageGenerationModel, etc.
    """

    @property
    def generate_parameters(self) -> list["inspect.Parameter"]:
        """Returns the parameters of the `generate` method.

        Returns:
            A list containing the parameters of the `generate` method.
        """
        return list(inspect.signature(self.generate).parameters.values())

    @property
    def runtime_parameters_names(self) -> "RuntimeParametersNames":
        """Returns the runtime parameters of the `ImageGenerationModel`, which are combination of the
        attributes of the `ImageGenerationModel` type hinted with `RuntimeParameter` and the parameters
        of the `generate` method that are not `input` and `num_generations`.

        Returns:
            A dictionary with the name of the runtime parameters as keys and a boolean
            indicating if the parameter is optional or not.
        """
        runtime_parameters = super().runtime_parameters_names
        runtime_parameters["generation_kwargs"] = {}

        # runtime parameters from the `generate` method
        for param in self.generate_parameters:
            if param.name in ["input", "inputs", "num_generations"]:
                continue
            is_optional = param.default != inspect.Parameter.empty
            runtime_parameters["generation_kwargs"][param.name] = is_optional

        return runtime_parameters

    def get_runtime_parameters_info(self) -> List["RuntimeParameterInfo"]:
        """Gets the information of the runtime parameters of the `LLM` such as the name
        and the description. This function is meant to include the information of the runtime
        parameters in the serialized data of the `LLM`.

        Returns:
            A list containing the information for each runtime parameter of the `LLM`.
        """
        runtime_parameters_info = super().get_runtime_parameters_info()

        generation_kwargs_info = next(
            (
                runtime_parameter_info
                for runtime_parameter_info in runtime_parameters_info
                if runtime_parameter_info["name"] == "generation_kwargs"
            ),
            None,
        )

        # If `generation_kwargs` attribute is present, we need to include the `generate`
        # method arguments as the information for this attribute.
        if generation_kwargs_info:
            generate_docstring_args = self.generate_parsed_docstring["args"]
            generation_kwargs_info["keys"] = []

            for key, value in generation_kwargs_info["optional"].items():
                info = {"name": key, "optional": value}
                if description := generate_docstring_args.get(key):
                    info["description"] = description
                generation_kwargs_info["keys"].append(info)

            generation_kwargs_info.pop("optional")

        return runtime_parameters_info

    @cached_property
    def generate_parsed_docstring(self) -> "Docstring":
        """Returns the parsed docstring of the `generate` method.

        Returns:
            The parsed docstring of the `generate` method.
        """
        return parse_google_docstring(self.generate)
