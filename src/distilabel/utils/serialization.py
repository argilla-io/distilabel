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

import importlib
import os
import sys
from enum import Enum

import orjson

from distilabel.mixins.runtime_parameters import RuntimeParametersMixin

if sys.version_info < (3, 11):
    from enum import EnumMeta as EnumType
else:
    from enum import EnumType

from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
)

import yaml
from pydantic import BaseModel
from typing_extensions import Self

T = TypeVar("T")

DISTILABEL_FILENAME = "distilabel-file.json"
TYPE_INFO_KEY = "type_info"


StrOrPath = Union[str, os.PathLike]
SaveFormats = Literal["json", "yaml"]


# Mapping to handle import paths that could have been serialized from previous versions
_OLD_IMPORT_MODULE_ATTR: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("distilabel.pipeline.base", "_Batch"): ("distilabel.pipeline.batch", "_Batch"),
    ("distilabel.pipeline.base", "_BatchManager"): (
        "distilabel.pipeline.batch_manager",
        "_BatchManager",
    ),
    ("distilabel.pipeline.base", "_BatchManagerStep"): (
        "distilabel.pipeline.batch_manager",
        "_BatchManagerStep",
    ),
}


def _get_module_attr(module: str, name: str) -> Type:
    """Gets a class given the module and the name of the class.

    Returns:
        The type of the class.
    """

    if (module, name) in _OLD_IMPORT_MODULE_ATTR:
        module, name = _OLD_IMPORT_MODULE_ATTR[(module, name)]

    mod = importlib.import_module(module)
    return getattr(mod, name)


def load_with_type_info(class_: Any) -> Any:
    """Creates an instance of a class from a dictionary containing the type info and the
    serialized data of the class.

    Args:
        class_: dictionary containing the type info and the serialized data of the class.

    Returns:
        An instance of the class with the data loaded from the dictionary.
    """
    if not isinstance(class_, (list, dict)):
        return class_

    if isinstance(class_, list):
        return [load_with_type_info(x) for x in class_]

    for k, v in class_.items():
        class_[k] = load_with_type_info(v) if isinstance(v, (dict, list)) else v

        if isinstance(v, dict) and "_type" in v and v["_type"] == "enum":
            class_[k] = Enum(v["_name"], v["_values"], type=eval(v["_enum_type"]))

    if TYPE_INFO_KEY not in class_:
        return class_

    type_info = class_.pop(TYPE_INFO_KEY)

    cls = _get_module_attr(type_info["module"], type_info["name"])

    if issubclass(cls, BaseModel):
        # `pop` keys from the dictionary that are not in the model fields
        field_names = cls.model_fields
        keys_to_drop = [k for k in class_.keys() if k not in field_names]
        for k in keys_to_drop:
            class_.pop(k)

    instance = cls(**class_)
    return instance


def write_json(filename: Path, data: Any) -> None:
    """Writes a JSON file to the given path, creating the parent dir if needed.

    Args:
        filename: the path to the file.
        data: the data to write to the file.
    """
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY))


def read_json(filename: StrOrPath) -> Any:
    """Reads a JSON file.

    Args:
        filename: the path to the JSON file.

    Returns:
        The data from the file.
    """
    with open(filename, "rb") as f:
        return orjson.loads(f.read())


def write_yaml(filename: Path, data: Dict[str, Any]) -> None:
    """Writes a YAML file to the given path, creating the parent dir if needed.

    Args:
        filename: the path to the file.
        data: the data to write to the file.
    """
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)


def read_yaml(filename: StrOrPath) -> Dict[str, Any]:
    """Reads a YAML file.

    Args:
        filename: the path to the YAML file.

    Returns:
        The data from the file.
    """
    with open(filename, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


class _Serializable:
    """Base class for serializable classes. It provides the means to serialize and deserialize."""

    _type_info: Dict[str, Any] = {}

    def _model_dump(self, obj: Any, **kwargs: Any) -> Dict[str, Any]:
        """Method in charge of serializing the object.

        This method works for `pydantic.BaseModel`s. Other classes will have to reimplement
        this method, like the `DAG` class.

        The signature must be respected, `obj` represents the object to serialize and `kwargs` are
        the optional parameters for the method used.

        Args:
            obj: the object to be serialized.

        Returns:
            A dictionary containing the serializable content of the class.
        """
        # Any parameter named api_key will be excluded from the dump (those are supposed to be SecretStr anyway,
        # and will remove them afterwards)
        dump = obj.model_dump(exclude="api_key", **kwargs)

        # Check if any attribute in value within the `dump` is an `EnumType`,
        # as it needs a specific serialization.
        for k, v in dump.items():
            if isinstance(v, EnumType):
                dump[k] = {
                    "_type": "enum",
                    "_enum_type": type(next(iter(v)).value).__name__,  # type: ignore
                    "_name": getattr(obj, k).__name__,
                    "_values": {x.name: x.value for x in v},  # type: ignore
                }
            elif isinstance(v, list):
                obj_list = getattr(obj, k)
                if isinstance(obj_list, list) and isinstance(
                    obj_list[0], RuntimeParametersMixin
                ):
                    dump[k] = {str(i): list_v for i, list_v in enumerate(v)}

        # Grab the fields that need extra care (LLMs from inside tasks)
        to_update = _extra_serializable_fields(obj)

        # Update those in the dumped dict
        for field in to_update:
            dump.update(field)

        return dump

    def dump(self, **kwargs: Any) -> Dict[str, Any]:
        """Transforms the class into a dict including the type info to be serialized.

        Args:
            kwargs: optional parameters to be used in the serialization process.

        Returns:
            A dictionary containing the serializable content of the class.
        """
        _dict = self._model_dump(self, **kwargs)

        # Remove private variables from the dump
        _dict = {k: v for k, v in _dict.items() if not k.startswith("_")}
        _dict[TYPE_INFO_KEY] = {
            "module": type(self).__module__,
            "name": type(self).__name__,
        }
        return _dict

    def save(
        self,
        path: Union[StrOrPath, None] = None,
        format: SaveFormats = "json",
        dump: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Serialized the object and saves it to a file.

        Args:
            path: filename of the object to save. If a folder is given, will create the object
                inside. If None is given, the file will be created at the current
                working directory. Defaults to None.
            format: the format to use when saving the file. Valid options are 'json' and
                'yaml'. Defaults to `"json"`.
            dump: the serialized object to save. If None, the object will be serialized using
                the default self.dump. This variable is here to allow extra customization, in
                general should be set as None.

        Raises:
            ValueError: if the provided `format` is not valid.
        """
        if path is None:
            path = Path.cwd() / DISTILABEL_FILENAME
        path = Path(path)
        if path.suffix == "":
            # If the path has no suffix, assume the user just wants a folder to write the task
            path = path / DISTILABEL_FILENAME

        if dump is None:
            dump = self.dump(**kwargs)

        if format == "json":
            write_json(path, dump)
        elif format == "yaml":
            write_yaml(path, dump)
        else:
            raise ValueError(
                f"Invalid format: '{format}', must be one of {get_args(SaveFormats)}."
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """Creates a class from a dict containing the type info and the serialized data
        of the class.

        Args:
            data: dictionary containing the type info and the serialized data of the class.

        Returns:
            An instance of the class with the data loaded from the dictionary.
        """
        return load_with_type_info(data)

    @classmethod
    def from_json(cls, path: StrOrPath) -> Self:
        """Loads a class type info and the serialized data from a JSON file and returns
        an instance of the class.

        Args:
            path: the path to the file containing the serialized class.

        Raises:
            ValueError: if the path is a directory.

        Returns:
            An instance of the class.
        """
        _check_is_dir(path)
        content = read_json(path)
        return cls.from_dict(content)

    @classmethod
    def from_yaml(cls, path: StrOrPath) -> Self:
        """Loads a class type info and the serialized data from a YAML file and returns
        an instance of the class.

        Args:
            path: the path to the file containing the serialized class.

        Raises:
            ValueError: if the path is a directory.

        Returns:
            An instance of the class.
        """
        _check_is_dir(path)
        content = read_yaml(path)
        return cls.from_dict(content)

    @classmethod
    def from_file(cls, path: StrOrPath) -> Self:
        """Loads a class from a file.

        Args:
            path: the path to the file containing the serialized class.

        Returns:
            An instance of the class.
        """
        path = Path(path)
        if path.suffix == ".json":
            return cls.from_json(path)

        if path.suffix == ".yaml" or path.suffix == ".yml":
            return cls.from_yaml(path)

        raise ValueError(
            f"Invalid file format: '{path.suffix}', must be one of {get_args(SaveFormats)}."
        )


def _check_is_dir(path: StrOrPath) -> None:
    if Path(path).is_dir():
        raise ValueError(f"You must provide a file path, not a directory: {path}")


def _extra_serializable_fields(obj: BaseModel) -> List[Dict[str, Dict[str, Any]]]:
    """Gets the information of the nested `_Serializable` attributes within another `_Serializable`
    instance.

    It's mainly used to get the information of the `LLM` objects inside a `Task` object,
    as they are nested and need to be serialized (`type_info`).

    Args:
        obj: the object to extract the information from.

    Returns:
        A list of dictionaries containing the information of the nested `_Serializable`
        attributes.
    """
    from distilabel.pipeline.base import BasePipeline

    to_update = []
    for k in obj.model_fields.keys():
        field = getattr(obj, k)
        # Have to remove the Pipeline as it will be inside the Step objects but is really
        # in a higher level hierarchy.
        if isinstance(field, BasePipeline):
            continue

        if isinstance(field, _Serializable):
            to_update.append({k: getattr(obj, k).dump()})
        elif isinstance(field, list) and field and isinstance(field[0], _Serializable):
            to_update.append({k: {str(i): x.dump() for i, x in enumerate(field)}})

    return to_update
