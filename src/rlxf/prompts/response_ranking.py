from abc import ABC, abstractmethod
from typing import Any, List
from typing_extensions import TypedDict

from rlxf.prompts.base import PromptTemplate


class Rank(TypedDict):
    rank: int
    description: str


class RankOutput(TypedDict):
    score: int
    rationale: str


class ResponseRankingPromptTemplate(PromptTemplate, ABC):
    task_description: str
    ranks: List[Rank]
    ranks_description: str

    __type__: str = "ranking"

    @abstractmethod
    def generate_prompt(self, **kwargs: Any) -> str:
        ...

    @abstractmethod
    def parse_output(self, output: str) -> RankOutput | List[RankOutput]:
        ...
