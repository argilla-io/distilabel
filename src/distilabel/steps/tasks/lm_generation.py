from typing import TYPE_CHECKING, Callable
from functools import partial
from pydantic import Field

from distilabel.steps.tasks import Task
from distilabel.pydantics import Stage, LMConfig

if TYPE_CHECKING:
    from distilabel.typing import (
        StepColumns,
        ChatType,
    )

class LMGenerationTask(Task):
    '''
    Task for running LM/VLM generation with a sampled system prompt and structured output.

    The pydantic model will be unpacked into separate columns for output or will be none if 
    structured output fails within `STRUCTURED_OUTPUT_RETRIES`.

    Args:
    ---
        in_cols: extra columns to include in the messages to the LM, postfixed in order 
    '''
    stage: Stage = Field(default_factory=Stage, exclude=True)
    lm_config: LMConfig = Field(default_factory=LMConfig, exclude=True)
    in_cols: list[str] = []
    input_formatter: Callable = Field(default=lambda **kwargs: kwargs, exclude=True)

    # note that Task.unload() will unload the llm, so we don't need to do that ourselves
    def load(self):
        # add_raw_input is set to False because if it has image type messages, they can't be formatted in a pytable
        super().load()
        self.add_raw_input = False
    
    @property
    def pydantic_fields(self) -> list[str]:
        return list(self.lm_config.out_model.model_fields.keys())

    @property
    def inputs(self) -> 'StepColumns':
        return ['source'] + self.in_cols

    @property
    def outputs(self) -> 'StepColumns':
        return ['source', 'model_name', *self.pydantic_fields, 'system']

    def format_input(self, input: dict) -> 'ChatType':
        return self.input_formatter(input, self.in_cols)

    def format_output(self, output: str | None, input: dict) -> dict:
        none_dict = dict.fromkeys(self.pydantic_fields)
        load_pydantic = partial(self.lm_config.out_model.model_validate_json, strict=True)

        pydantic_output = load_pydantic(output).model_dump() if output is not None else none_dict
        return {**pydantic_output, 'source': input['source'], 'system': input['system']}
