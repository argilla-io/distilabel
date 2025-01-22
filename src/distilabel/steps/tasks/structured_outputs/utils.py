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


def resolve_refs(schema_element: Any, defs: Optional[Dict[str, Any]] = None) -> Any:
    """Resolves JSON schema references.

    Args:
        schema_element: The JSON schema or part of it to resolve.
        defs: The definitions of the JSON schema.

    Returns:
        The resolved JSON schema.
    """
    if isinstance(schema_element, dict):
        # If the schema contains a $ref, resolve it
        if "$ref" in schema_element and len(schema_element) == 1:
            ref_key = schema_element["$ref"].split("/")[-1]
            resolved = defs.get(ref_key, {})
            return resolve_refs(resolved, defs)  # Resolve recursively
        else:
            # Recursively resolve properties of the dictionary
            return {
                key: resolve_refs(value, defs) for key, value in schema_element.items()
            }
    elif isinstance(schema_element, list):
        # Resolve each item in the list
        return [resolve_refs(item, defs) for item in schema_element]
    # Return other types as-is
    return schema_element


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
    json_schema = resolve_refs(json_schema, defs)

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


def handle_any_of(json_schema: Dict[str, Any]) -> Any:
    """Handle 'anyOf' in JSON schema."""
    types = [json_schema_to_pydantic_type(schema) for schema in json_schema["anyOf"]]
    return Union[tuple(types)]


def handle_array(json_schema: Dict[str, Any]) -> Any:
    """Handle 'array' type in JSON schema."""
    items_schema = json_schema.get("items")
    if items_schema:
        item_type = json_schema_to_pydantic_type(items_schema)
        return List[item_type]
    return List


def handle_object(json_schema: Dict[str, Any]) -> Any:
    """Handle 'object' type in JSON schema."""
    properties = json_schema.get("properties")
    if properties:
        return json_schema_to_model(json_schema)
    return Dict


def json_schema_to_pydantic_type(json_schema: Dict[str, Any]) -> Any:
    """Converts a JSON schema type to a Pydantic type.

    Args:
        json_schema: The JSON schema to convert.

    Returns:
        A Pydantic type.
    """
    if "anyOf" in json_schema:
        return handle_any_of(json_schema)

    type_ = json_schema.get("type")

    if type_ == "string":
        return str
    elif type_ == "integer":
        return int
    elif type_ == "number":
        return float
    elif type_ == "boolean":
        return bool
    elif type_ == "array":
        return handle_array(json_schema)
    elif type_ == "object":
        # Handle nested models.
        return handle_object(json_schema)
    elif type_ == "null":
        return Optional[Any]  # Use Optional[Any] for nullable fields
    else:
        raise ValueError(f"Unsupported JSON schema type: {type_}")
