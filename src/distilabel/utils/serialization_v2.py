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
import json
import os
from pathlib import Path
from typing import Any, Dict, Generic, Literal, Optional, Type, TypeVar, get_args

import yaml

T = TypeVar("T")

DISTILABEL_FILENAME = "distilabel-file.json"


SaveFormat = Literal["json", "yaml"]


def _get_class(module: str = None, name: str = None) -> Type:
    mod = importlib.import_module(module)
    return getattr(mod, name)


def load_from_dict(class_: Dict[str, Any]) -> Generic[T]:
    """Reads a template (a class serialized) and returns the instance
    contained.

    Args:
        template (Dict[str, Any]): Dict containing the template, the dict serialized.

    Returns:
        Generic[T]: Instance contained in the template
    """
    type_info = class_.pop("_type_info_")
    if "_type_info_" in type_info:
        # There is a nested type_info, load the class recursively
        type_info = load_from_dict(type_info)

    cls = _get_class(type_info["module"], type_info["name"])
    instance = cls(**class_)
    return instance


def write_json(filename: Path, data: Dict[str, Any]) -> None:
    """Writes a json file to the given path, creates the parent dir.

    Args:
        filename (Path): Name of the file.
        data (Dict[str, Any]): Dict to be written as json.
    """
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as file:
        json.dump(data, file, indent=2)


def read_json(filename: Path) -> Dict[str, Any]:
    """Read a json file from disk.

    Args:
        filename (Path): Name of the json file.

    Returns:
        Dict[str, Any]: Dict containing the json data.
    """
    with open(filename, "r") as file:
        return json.load(file)


def write_yaml(filename: Path, data: Dict[str, Any]) -> None:
    """Writes a yaml file to the given path, creates the parent dir.

    Args:
        filename (Path): Name of the file.
        data (Dict[str, Any]): Dict to be written as yaml.
    """
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def read_yaml(filename: Path) -> Dict[str, Any]:
    """Read a yaml file from disk.

    Args:
        filename (Path): Name of the yaml file.

    Returns:
        Dict[str, Any]: Dict containing the json data.
    """
    with open(filename, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


class _Serializable:
    """Base class for serializable classes. It provides the means to serialize and deserialize."""

    _type_info_: Dict[str, Any] = {}

    def _model_dump(self, obj: Any, **kwargs: Any) -> Dict[str, Any]:
        """Method in charge of serializing the object.

        This method works for pydantic models (the classes that inherit from pydantic's
        BaseModel).

        The signature must be respected, `obj` represents the object to serialize and `kwargs` are
        the optional parameters for the method used.
        For example in the case of a pydantic model, the method `model_dump` is used,
        but for the DAG class, we networkx to obtain the information.

        Args:
            obj (Any): The object to be serialized.

        Returns:
            Dict[str, Any]: Data to be dumped for the object.
        """
        # NOTE(plaguss): This should be reimplemented for some cases, so maybe we have to enforce it via ABC?.
        # It will work out of the box for pydantic's BaseModels.
        return obj.model_dump(**kwargs)

    def dump(self, **kwargs: Any) -> Dict[str, Any]:
        """Transforms the class into a dict to write to a file.

        Args:
            kwargs: Optional parameters to be used in the serialization process.

        Returns:
            Dict[str, Any]: Serializable content of the class.

        """
        _dict = self._model_dump(self, **kwargs)
        # Remove private variables from the dump
        _dict = {k: v for k, v in _dict.items() if not k.startswith("_")}
        _dict["_type_info_"] = {
            "module": type(self).__module__,
            "name": type(self).__name__,
        }
        return _dict

    def save(
        self,
        path: Optional[os.PathLike] = None,
        format: SaveFormat = "json",
        **kwargs: Any,
    ) -> None:
        """Writes the content to a file.

        Args:
            path (Optional[os.PathLike], optional):
                Filename of the object to save. If a folder is given, will create the object
                inside. If None is given, the file will be created at the current
                working directory. Defaults to None.
            format (SaveFormat, optional): The format to save the file, must be one of `json` or `yaml`.
        """
        if path is None:
            path = Path.cwd() / DISTILABEL_FILENAME
        path = Path(path)
        if path.suffix == "":
            # If the path has no suffix, assume the user just wants a folder to write the task
            path = path / DISTILABEL_FILENAME

        if format == "json":
            write_json(path, self.dump(**kwargs))
        elif format == "yaml":
            write_yaml(path, self.dump(**kwargs))
        else:
            raise ValueError(
                f"Invalid format: '{format}', must be one of {get_args(SaveFormat)}."
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Generic[T]:
        """Creates a class from a dict and returns the instance.

        Args:
            data (Dict[str, Any]): Data needed to create the instance.

        Returns:
            Generic[T]: Instance of the class.
        """
        return load_from_dict(data)

    @classmethod
    def from_json(cls, path: os.PathLike) -> Generic[T]:
        """Loads a class from a file and returns the instance contained.

        Args:
            path (os.PathLike): Path to the file containing the serialized class.

        Raises:
            ValueError: If the path is a directory.

        Returns:
            Generic[T]: Instance of the class.
        """
        _check_is_dir(path)
        content = read_json(path)
        return cls.from_dict(content)

    @classmethod
    def from_yaml(cls, path: os.PathLike) -> Generic[T]:
        """Loads a class from a yaml file and returns the instance contained.

        Args:
            path (os.PathLike): Path to the file containing the serialized class.

        Raises:
            ValueError: If the path is a directory.

        Returns:
            Generic[T]: Instance of the class.
        """
        _check_is_dir(path)
        content = read_yaml(path)
        return cls.from_dict(content)


def _check_is_dir(path: os.PathLike) -> None:
    if Path(path).is_dir():
        raise ValueError(f"You must provide a file path, not a directory: {path}")
