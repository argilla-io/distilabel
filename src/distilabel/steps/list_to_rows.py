from typing import TYPE_CHECKING
from distilabel.steps import Step, StepInput

if TYPE_CHECKING:
    from distilabel.typing import (
        StepColumns,
        StepOutput,
    )

class ListToRows(Step):
    '''
    Takes a list from column `input_col` (expected to be a list of any Python base type) and splits it 
    into separate rows, replacing the `input_col` and passing through `other_fields` 
    (any field not referenced is dropped, so account for that).
    '''
    input_col: str

    @property
    def inputs(self) -> 'StepColumns':
        return [self.input_col]

    @property
    def outputs(self) -> 'StepColumns':
        return [self.input_col]

    def process(self, *inputs: StepInput) -> 'StepOutput':
        for step_input in inputs:
            expanded_fields = [
                {k: row[k] for k in row.keys()} | {self.input_col: field}
                for row in step_input
                    for field in (row[self.input_col] if row[self.input_col] else [None])
            ]
            yield expanded_fields
