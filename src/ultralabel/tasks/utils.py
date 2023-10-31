from typing import Literal

from typing_extensions import TypedDict


class ChatCompletion(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str
