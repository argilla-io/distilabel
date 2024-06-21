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
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Set, Type, Union

from pydantic import BaseModel, Field, create_model


def schema_as_dict(schema: Union[str, Type[BaseModel]]) -> Dict[str, Any]:
    """Helper function to obtain the schema and simplify serialization."""
    if type(schema) == type(BaseModel):
        return schema.model_json_schema()
    elif isinstance(schema, str):
        return json.loads(schema)
    return schema


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
    model_name = json_schema.get("title")
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


# NOTE: Code copied from langchain core repo:
# https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/utils/json_schema.py
# These are functions to work with JSON schemas and OpenAI function calls.


def _retrieve_ref(path: str, schema: dict) -> dict:
    components = path.split("/")
    if components[0] != "#":
        raise ValueError(
            "ref paths are expected to be URI fragments, meaning they should start "
            "with #."
        )
    out = schema
    for component in components[1:]:
        if component.isdigit():
            out = out[int(component)]
        else:
            out = out[component]
    return deepcopy(out)


def _dereference_refs_helper(
    obj: Any,
    full_schema: Dict[str, Any],
    skip_keys: Sequence[str],
    processed_refs: Optional[Set[str]] = None,
) -> Any:
    if processed_refs is None:
        processed_refs = set()

    if isinstance(obj, dict):
        obj_out = {}
        for k, v in obj.items():
            if k in skip_keys:
                obj_out[k] = v
            elif k == "$ref":
                if v in processed_refs:
                    continue
                processed_refs.add(v)
                ref = _retrieve_ref(v, full_schema)
                full_ref = _dereference_refs_helper(
                    ref, full_schema, skip_keys, processed_refs
                )
                processed_refs.remove(v)
                return full_ref
            elif isinstance(v, (list, dict)):
                obj_out[k] = _dereference_refs_helper(
                    v, full_schema, skip_keys, processed_refs
                )
            else:
                obj_out[k] = v
        return obj_out
    elif isinstance(obj, list):
        return [
            _dereference_refs_helper(el, full_schema, skip_keys, processed_refs)
            for el in obj
        ]
    else:
        return obj


def _infer_skip_keys(
    obj: Any, full_schema: dict, processed_refs: Optional[Set[str]] = None
) -> List[str]:
    if processed_refs is None:
        processed_refs = set()

    keys = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "$ref":
                if v in processed_refs:
                    continue
                processed_refs.add(v)
                ref = _retrieve_ref(v, full_schema)
                keys.append(v.split("/")[1])
                keys += _infer_skip_keys(ref, full_schema, processed_refs)
            elif isinstance(v, (list, dict)):
                keys += _infer_skip_keys(v, full_schema, processed_refs)
    elif isinstance(obj, list):
        for el in obj:
            keys += _infer_skip_keys(el, full_schema, processed_refs)
    return keys


def dereference_refs(
    schema_obj: dict,
    *,
    full_schema: Optional[dict] = None,
    skip_keys: Optional[Sequence[str]] = None,
) -> dict:
    """Try to substitute $refs in JSON Schema."""

    full_schema = full_schema or schema_obj
    skip_keys = (
        skip_keys
        if skip_keys is not None
        else _infer_skip_keys(schema_obj, full_schema)
    )
    return _dereference_refs_helper(schema_obj, full_schema, skip_keys)


def _rm_titles(kv: dict, prev_key: str = "") -> dict:
    new_kv = {}
    for k, v in kv.items():
        if k == "title":
            if isinstance(v, dict) and prev_key == "properties" and "title" in v.keys():
                new_kv[k] = _rm_titles(v, k)
            else:
                continue
        elif isinstance(v, dict):
            new_kv[k] = _rm_titles(v, k)
        else:
            new_kv[k] = v
    return new_kv


def convert_to_openai_function(
    model_or_schema: Union[Type[BaseModel], Dict[str, Any]],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Converts a Pydantic model or json_schema from a pydantic model
    to a function description for the OpenAI API.

    Args:
        model_or_schema (Union[Type[BaseModel], Dict[str, Any]]): _description_
        name (Optional[str], optional): _description_. Defaults to None.
        description (Optional[str], optional): _description_. Defaults to None.

    Returns:
        Dict[str, Any]: _description_
    """
    schema = dereference_refs(
        model_or_schema.model_json_schema()
        if isinstance(model_or_schema, type(BaseModel))
        else model_or_schema
    )
    schema.pop("definitions", None)
    title = schema.pop("title", "")
    default_description = schema.pop("description", "")
    return {
        "name": name or title,
        "description": description or default_description,
        "parameters": _rm_titles(schema),
    }
