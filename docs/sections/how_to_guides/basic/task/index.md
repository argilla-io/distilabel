# Define Tasks as Steps that rely on LLMs

## Working with Tasks

The [`Task`][distilabel.steps.tasks.Task] is a special kind of [`Step`][distilabel.steps.Step] that includes the [`LLM`][distilabel.llms.LLM] as a mandatory argument. As with a [`Step`][distilabel.steps.Step], it is normally used within a [`Pipeline`][distilabel.pipeline.Pipeline] but can also be used standalone.

For example, the most basic task is the [`TextGeneration`][distilabel.steps.tasks.TextGeneration] task, which generates text based on a given instruction.

```python
from distilabel.steps.tasks import TextGeneration

task = TextGeneration(
    name="text-generation",
    llm=OpenAILLM(model="gpt-4"),
)
task.load()

next(task.process([{"instruction": "What's the capital of Spain?"}]))
# [
#     {
#         "instruction": "What's the capital of Spain?",
#         "generation": "The capital of Spain is Madrid.",
#         "model_name": "gpt-4",
#         "distilabel_metadata": {
#             "raw_output_text-generation": "The capital of Spain is Madrid"
#         }
#     }
# ]
```

!!! NOTE
    The `Step.load()` always needs to be executed when being used as a standalone. Within a pipeline, this will be done automatically during pipeline execution.

As shown above, the [`TextGeneration`][distilabel.steps.tasks.TextGeneration] task adds a `generation` based on the `instruction`. Additionally, it provides some metadata about the LLM call through `distilabel_metadata`. This can be disabled by setting the `add_raw_output` attribute to `False` when creating the task.

## Defining custom Tasks

We can define a custom step by creating a new subclass of the [`Task`][distilabel.steps.tasks.Task] and defining the following:

- `inputs`: is a property that returns a list of strings with the names of the required input fields.

- `format_input`: is a method that receives a dictionary with the input data and returns a [`ChatType`][distilabel.steps.tasks.ChatType] following [the chat-completion OpenAI message formatting](https://platform.openai.com/docs/guides/text-generation).

- `outputs`: is a property that returns a list of strings with the names of the output fields, this property should always include `model_name` as one of the outputs since that's automatically injected from the LLM.

- `format_output`: is a method that receives the output from the [`LLM`][distilabel.llms.LLM] and optionally also the input data (which may be useful to build the output in some scenarios), and returns a dictionary with the output data formatted as needed i.e. with the values for the columns in `outputs`. Note that there's no need to include the `model_name` in the output.

```python
from typing import Any, Dict, List, Union

from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.typing import ChatType


class MyCustomTask(Task):
    @property
    def inputs(self) -> List[str]:
        return ["input_field"]

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        return [
            {
                "role": "user",
                "content": input["input_field"],
            },
        ]

    @property
    def outputs(self) -> List[str]:
        return ["output_field", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"output_field": output}
```
