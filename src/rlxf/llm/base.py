from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rlxf.prompts.base import PromptTemplate


class LLM(ABC):
    def __init__(self, prompt_template: PromptTemplate) -> None:
        self.prompt_template = prompt_template

    @abstractmethod
    def generate(self, prompts: list[dict[str, Any]]) -> Any:
        pass
