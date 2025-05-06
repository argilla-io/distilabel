from typing import TYPE_CHECKING
from distilabel.steps import Step, StepInput
from functools import partial
from pydantic import BaseModel

if TYPE_CHECKING:
    from distilabel.typing import (
        StepColumns,
        StepOutput,
    )

class LoadPydanticAsColumns(Step):
    '''
    Converts inputs into `pydantic_model` and then unpacks the fields as separate columns.
    '''
    pydantic_model: type[BaseModel]

    @property
    def inputs(self) -> 'StepColumns':
        return ['input']

    @property
    def outputs(self) -> 'StepColumns':
        return list(self.pydantic_model.model_fields.keys())

    def process(self, *inputs: StepInput) -> 'StepOutput':
        none_dict = dict.fromkeys(self.pydantic_model.model_fields.keys(), None)
        load_pydantic = partial(self.pydantic_model.model_validate_json, strict=True)
        for step_input in inputs:
            converted_input = [
                load_pydantic(input).model_dump() if input is not None else none_dict
                for input in step_input
            ]
            yield converted_input
