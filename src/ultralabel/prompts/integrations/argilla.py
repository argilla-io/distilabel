from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Type, Union

if TYPE_CHECKING:
    from argilla.client.feedback.schemas.records import FeedbackRecord
    from argilla.client.feedback.schemas.types import (
        AllowedFieldTypes,
        AllowedQuestionTypes,
    )


class ArgillaTemplate(ABC):
    @property
    @abstractmethod
    def argilla_fields_typedargs(self) -> Dict[str, Union[Type[str], Type[list]]]:
        pass

    @abstractmethod
    def to_argilla_fields(
        self, dataset_row: Dict[str, Any]
    ) -> List["AllowedFieldTypes"]:
        pass

    @property
    @abstractmethod
    def argilla_questions_typedargs(self) -> Dict[str, Type[list]]:
        pass

    @abstractmethod
    def to_argilla_questions(
        self, dataset_row: Dict[str, Any]
    ) -> List["AllowedQuestionTypes"]:
        pass

    @abstractmethod
    def to_argilla_record(self, dataset_row: Dict[str, Any]) -> "FeedbackRecord":
        pass
