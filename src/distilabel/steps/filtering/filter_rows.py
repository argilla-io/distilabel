from typing import TYPE_CHECKING, Callable
from distilabel.steps import Step, StepInput
from pydantic import Field

if TYPE_CHECKING:
    from distilabel.typing import (
        StepColumns,
        StepOutput,
    )

class FilterRows(Step):
    '''
    For each row, check if the condition is met for all columns in `cols`.

    If the condition is met for all columns, the row is kept, otherwise it is dropped.

    Example:
    ---
    ```python
    drop_none_questions = FilterRows(
        cols=['question'],
        condition=lambda question: question is not None
    )
    ```
    '''
    cols: list[str]
    condition: Callable = Field(default=lambda **kwargs: True, exclude=True)

    @property
    def inputs(self) -> 'StepColumns':
        return self.cols

    @property
    def outputs(self) -> 'StepColumns':
        return self.cols

    def process(self, *inputs: StepInput) -> 'StepOutput':
        for step_input in inputs:
            yield [
                row for row in step_input
                if all([self.condition(row[col]) for col in self.cols])
            ]
