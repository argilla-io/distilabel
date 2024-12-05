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

import json
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field, create_model


def schema_as_dict(
    schema: Union[str, Dict[str, Any], Type[BaseModel]],
) -> Dict[str, Any]:
    """Helper function to obtain the schema and simplify serialization."""
    if isinstance(schema, str):
        return json.loads(schema)

    if isinstance(schema, dict):
        return schema

    return schema.model_json_schema()


# NOTE: The following functions were copied from:
# https://github.com/pydantic/pydantic/issues/643#issuecomment-1999755873
# and slightly modified to work with nested models.
# It would be nice to find the original source of this code to give credit.
# Other option would be working with this library: https://github.com/c32168/dyntamic


def json_schema_to_model(json_schema: Dict[str, Any]) -> Type[BaseModel]:
    """Converts a JSON schema to a `pydantic.BaseModel` class.

    Args:
        json_schema: The JSON schema to convert.

    Returns:
        A `pydantic.BaseModel` class.
    """

    # Extract the model name from the schema title.
    model_name = json_schema["title"]
    if defs := json_schema.get("$defs", None):
        # This is done to grab the content of nested classes that need to dereference
        # the objects (those should be in a higher level).
        pass

    # Extract the field definitions from the schema properties.
    field_definitions = {
        name: json_schema_to_pydantic_field(
            name, prop, json_schema.get("required", []), defs=defs
        )
        for name, prop in json_schema.get("properties", {}).items()
    }

    # Create the BaseModel class using create_model().
    return create_model(model_name, **field_definitions)


def json_schema_to_pydantic_field(
    name: str,
    json_schema: Dict[str, Any],
    required: List[str],
    defs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Converts a JSON schema property to a `pydantic.Field`.

    Args:
        name: The field name.
        json_schema: The JSON schema property.
        required: The list of required fields.
        defs: The definitions of the JSON schema. It's used to dereference nested classes,
            so we can grab the original definition from the json schema (it won't
            work out of the box with just the reference).

    Returns:
        A `pydantic.Field`.
    """

    # NOTE(plaguss): This needs more testing, nested classes need extra work to be converted
    # here if we pass a reference to another class it will crash, we have to find the original
    # definition and insert it here
    # This takes into account single items referred to other classes
    if ref := json_schema.get("$ref"):
        json_schema = defs.get(ref.split("/")[-1])

    # This takes into account lists of items referred to other classes
    if "items" in json_schema and (ref := json_schema["items"].get("$ref")):
        json_schema["items"] = defs.get(ref.split("/")[-1])

    # Get the field type.
    type_ = json_schema_to_pydantic_type(json_schema)

    # Get the field description.
    description = json_schema.get("description")

    # Get the field examples.
    examples = json_schema.get("examples")

    # Create a Field object with the type, description, and examples.
    # The "required" flag will be set later when creating the model.
    return (
        type_,
        Field(
            description=description,
            examples=examples,
            default=... if name in required else None,
        ),
    )


def json_schema_to_pydantic_type(json_schema: Dict[str, Any]) -> Any:
    """Converts a JSON schema type to a Pydantic type.

    Args:
        json_schema: The JSON schema to convert.

    Returns:
        A Pydantic type.
    """
    type_ = json_schema.get("type")

    if type_ == "string":
        type_val = str
    elif type_ == "integer":
        type_val = int
    elif type_ == "number":
        type_val = float
    elif type_ == "boolean":
        type_val = bool
    elif type_ == "array":
        items_schema = json_schema.get("items")
        if items_schema:
            item_type = json_schema_to_pydantic_type(items_schema)
            type_val = List[item_type]
        else:
            type_val = List
    elif type_ == "object":
        # Handle nested models.
        properties = json_schema.get("properties")
        if properties:
            nested_model = json_schema_to_model(json_schema)
            type_val = nested_model
        else:
            type_val = Dict
    elif type_ == "null":
        type_val = Optional[Any]  # Use Optional[Any] for nullable fields
    else:
        raise ValueError(f"Unsupported JSON schema type: {type_}")

    return type_val
