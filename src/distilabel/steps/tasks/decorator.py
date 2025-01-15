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
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, Final, List, Tuple, Type, Union

import yaml

from distilabel.errors import DistilabelUserError
from distilabel.steps.tasks.base import Task
from distilabel.typing import FormattedInput

if TYPE_CHECKING:
    from distilabel.typing import StepColumns


TaskFormattingOutputFunc = Callable[..., Dict[str, Any]]


def task(
    inputs: Union["StepColumns", None] = None,
    outputs: Union["StepColumns", None] = None,
) -> Callable[..., Type["Task"]]:
    """Creates a `Task` from a formatting output function.

    Args:
        inputs: a list containing the name of the inputs columns/keys or a dictionary
            where the keys are the columns and the values are booleans indicating whether
            the column is required or not, that are required by the step. If not provided
            the default will be an empty list `[]` and it will be assumed that the step
            doesn't need any specific columns. Defaults to `None`.
        outputs: a list containing the name of the outputs columns/keys or a dictionary
            where the keys are the columns and the values are booleans indicating whether
            the column will be generated or not. If not provided the default will be an
            empty list `[]` and it will be assumed that the step doesn't need any specific
            columns. Defaults to `None`.
    """

    inputs = inputs or []
    outputs = outputs or []

    def decorator(func: TaskFormattingOutputFunc) -> Type["Task"]:
        doc = inspect.getdoc(func)
        if doc is None:
            raise DistilabelUserError(
                "When using the `task` decorator, including a docstring in the formatting"
                " function is mandatory. The docstring must follow the format described"
                " in the documentation.",
                page="",
            )

        system_prompt, user_message_template = _parse_docstring(doc)
        _validate_templates(inputs, system_prompt, user_message_template)

        def inputs_property(self) -> "StepColumns":
            return inputs

        def outputs_property(self) -> "StepColumns":
            return outputs

        def format_input(self, input: Dict[str, Any]) -> "FormattedInput":
            return [
                {"role": "system", "content": system_prompt.format(**input)},
                {"role": "user", "content": user_message_template.format(**input)},
            ]

        def format_output(
            self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
        ) -> Dict[str, Any]:
            return func(output, input)

        return type(
            func.__name__,
            (Task,),
            {
                "inputs": property(inputs_property),
                "outputs": property(outputs_property),
                "__module__": func.__module__,
                "format_input": format_input,
                "format_output": format_output,
            },
        )

    return decorator


_SYSTEM_PROMPT_YAML_KEY: Final[str] = "system_prompt"
_USER_MESSAGE_TEMPLATE_YAML_KEY: Final[str] = "user_message_template"
_DOCSTRING_FORMATTING_FUNCTION_ERROR: Final[str] = (
    "Formatting function decorated with `task` doesn't follow the expected format. Please,"
    " check the documentation and update the function to include a docstring with the expected"
    " format."
)


def _parse_docstring(docstring: str) -> Tuple[str, str]:
    """Parses the docstring of the formatting function that was built using the `task`
    decorator.

    Args:
        docstring: the docstring of the formatting function.

    Returns:
        A tuple containing the system prompt and the user message template.

    Raises:
        DistilabelUserError: if the docstring doesn't follow the expected format or if
            the expected keys are missing.
    """
    parts = docstring.split("---")

    if len(parts) != 3:
        raise DistilabelUserError(
            _DOCSTRING_FORMATTING_FUNCTION_ERROR,
            page="",
        )

    yaml_content = parts[1]

    try:
        parsed_yaml = yaml.safe_load(yaml_content)
        if not isinstance(parsed_yaml, dict):
            raise DistilabelUserError(
                _DOCSTRING_FORMATTING_FUNCTION_ERROR,
                page="",
            )

        system_prompt = parsed_yaml.get(_SYSTEM_PROMPT_YAML_KEY)
        user_template = parsed_yaml.get(_USER_MESSAGE_TEMPLATE_YAML_KEY)
        if system_prompt is None or user_template is None:
            raise DistilabelUserError(
                "The formatting function decorated with `task` must include both the `system_prompt`"
                " and `user_message_template` keys in the docstring. Please, check the documentation"
                " and update the docstring of the formatting function to include the expected"
                " keys.",
                page="",
            )

        return system_prompt.strip(), user_template.strip()

    except yaml.YAMLError as e:
        raise DistilabelUserError(_DOCSTRING_FORMATTING_FUNCTION_ERROR, page="") from e


TEMPLATE_PLACEHOLDERS_REGEX = re.compile(r"\{(\w+)\}")


def _validate_templates(
    inputs: "StepColumns", system_prompt: str, user_message_template: str
) -> None:
    """Validates the system prompt and user message template to ensure that they only
    contain the allowed placeholders i.e. the columns/keys that are provided as inputs.

    Args:
        inputs: the list of inputs columns/keys.
        system_prompt: the system prompt.
        user_message_template: the user message template.

    Raises:
        DistilabelUserError: if the system prompt or the user message template contain
            invalid placeholders.
    """
    list_inputs = list(inputs.keys()) if isinstance(inputs, dict) else inputs

    valid_system_prompt, invalid_system_prompt_placeholders = _validate_template(
        system_prompt, list_inputs
    )
    if not valid_system_prompt:
        raise DistilabelUserError(
            f"The formatting function decorated with `task` includes invalid placeholders"
            f" in the extracted `system_prompt` from the function docstring. Valid placeholders"
            f" are: {list_inputs}, but the following placeholders were found: {invalid_system_prompt_placeholders}."
            f" Please, update the `system_prompt` to only include the valid placeholders.",
            page="",
        )

    valid_user_message_template, invalid_user_message_template_placeholders = (
        _validate_template(user_message_template, list_inputs)
    )
    if not valid_user_message_template:
        raise DistilabelUserError(
            f"The formatting function decorated with `task` includes invalid placeholders"
            f" in the extracted `user_message_template` from the function docstring. Valid"
            f" placeholders are: {list_inputs}, but the following placeholders were found:"
            f" {invalid_user_message_template_placeholders}. Please, update the `system_prompt`"
            " to only include the valid placeholders.",
            page="",
        )


def _validate_template(
    template: str, allowed_placeholders: List[str]
) -> Tuple[bool, set[str]]:
    """Validates that the template only contains the allowed placeholders.

    Args:
        template: the template to validate.
        allowed_placeholders: the list of allowed placeholders.

    Returns:
        A tuple containing a boolean indicating if the template is valid and a set
        with the invalid placeholders.
    """
    placeholders = set(TEMPLATE_PLACEHOLDERS_REGEX.findall(template))
    allowed_placeholders_set = set(allowed_placeholders)
    are_valid = placeholders.issubset(allowed_placeholders_set)
    invalid_placeholders = placeholders - allowed_placeholders_set
    return are_valid, invalid_placeholders
