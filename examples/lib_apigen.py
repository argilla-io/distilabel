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

from typing import Any, Dict, Optional


def final_velocity(initial_velocity: float, acceleration: float, time: float) -> int:
    """Calculates the final velocity of an object given its initial velocity, acceleration, and time.

    Args:
        initial_velocity: The initial velocity of the object.
        acceleration: The acceleration of the object.
        time: The time elapsed.

    Returns:
        The final velocity
    """
    # Tool:
    # {"name": "final_velocity", "description": "Calculates the final velocity of an object given its initial velocity, acceleration, and time.", "parameters": {"initial_velocity": {"description": "The initial velocity of the object.", "type": "float"}, "acceleration": {"description": "The acceleration of the object.", "type": "float"}, "time": {"description": "The time elapsed.", "type": "float"}}}
    # Answer:
    # {"name": "final_velocity", "arguments": {"initial_velocity": 5, "acceleration": 1.5, "time": 40}}
    return initial_velocity + acceleration * time


def permutation_count(n: int, k: int) -> int:
    """Calculates the number of permutations of k elements from a set of n elements.

    Args:
        n: The total number of elements in the set.
        k: The number of elements to choose for the permutation.

    Returns:
        The number of permutations.
    """
    # Tool:
    # {"name": "permutation_count", "description": "Calculates the number of permutations of k elements from a set of n elements.", "parameters": {"n": {"description": "The total number of elements in the set.", "type": "int"}, "k": {"description": "The number of elements to choose for the permutation.", "type": "int"}}}
    # Answer:
    # {"name": "permutation_count", "arguments": {"n": 10, "k": 3}}
    import math

    return math.factorial(n) / math.factorial(n - k)


def getdivision(dividend: int, divisor: int) -> float:
    """Divides two numbers by making an API call to a division service.

    Args:
        dividend: The dividend in the division operation.
        divisor: The divisor in the division operation.

    Returns:
        Division of the 2 numbers.
    """
    # Tool:
    # {"name": "getdivision", "description": "Divides two numbers by making an API call to a division service.", "parameters": {"divisor": {"description": "The divisor in the division operation.", "type": "int", "default": ""}, "dividend": {"description": "The dividend in the division operation.", "type": "int", "default": ""}}}
    # Answer:
    # {"name": "getdivision", "arguments": {"divisor": 25, "dividend": 100}}
    return dividend / divisor


def binary_addition(a: str, b: str) -> str:
    """Adds two binary numbers and returns the result as a binary string.

    Args:
        a: The first binary number.
        b: The second binary number.

    Raises:
        ValueError: On invalid binary number.

    Returns:
        Binary string of the sum of the two numbers.
    """
    # Tool:
    # {"name": "binary_addition", "description": "Adds two binary numbers and returns the result as a binary string.", "parameters": {"a": {"description": "The first binary number.", "type": "str"}, "b": {"description": "The second binary number.", "type": "str"}}}
    # Answer:
    # {"name": "binary_addition", "arguments": {"a": "1010", "b": "1101"}}
    if not set(a).issubset("01") or not set(b).issubset("01"):
        raise ValueError("Invalid binary number")

    return bin(int(a, 2) + int(b, 2))[2:]


def _make_request(url: str, params: Optional[Dict[str, Any]] = None):
    import requests

    req = requests.get(url, params=params)
    return req.json()


def swapi_planet_resource(id: str) -> Dict[str, Any]:
    """get a specific planets resource

    Args:
        id: identifier of the planet

    Returns:
        Information about the planet.
    """
    # url = "https://swapi.dev/api/planets/1"
    return _make_request(r"https://swapi.dev/api/planets/", params={"id": id})


def disney_character(name: str) -> Dict[str, Any]:
    """Find a specific character using this endpoint

    Args:
        name: Name of the character to look for.

    Returns:
        Infrmation about the character.
    """
    # Example:
    # url = "https://api.disneyapi.dev/character"
    # params = {"name": "mulan"}
    return _make_request(r"https://api.disneyapi.dev/character", params={"name": name})


def get_lib():
    return {
        "swapi_planet_resource": swapi_planet_resource,
        "disney_character": disney_character,
        "final_velocity": final_velocity,
        "permutation_count": permutation_count,
        "getdivision": getdivision,
        "binary_addition": binary_addition,
    }


def get_tools() -> Dict[str, Dict[str, Any]]:
    """Returns the tool representation of the functions in the library."""
    # TODO: Improve the `get_json_schema`, it fails on a lot of examples.
    from transformers.utils import get_json_schema

    return {name: get_json_schema(func) for name, func in get_lib().items()}
