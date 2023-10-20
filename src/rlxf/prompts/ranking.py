from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, TypedDict

from rlxf.prompts.base import PromptTemplate


class Rank(TypedDict):
    rank: int
    description: str


class RankOutput(TypedDict):
    score: int
    rationale: str


@dataclass
class RankingPromptTemplate(PromptTemplate, ABC):
    ranks: List[Rank]
    ranks_description: str

    __type__: str = "ranking"

    @abstractmethod
    def generate_prompt(self, **kwargs: Any) -> str:
        ...

    def parse_output(self, output: str) -> RankOutput:
        ...
