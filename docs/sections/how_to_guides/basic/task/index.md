# Tasks for generating and judging with LLMs

## Working with Tasks

The [`Task`][distilabel.steps.tasks.Task] is a special kind of [`Step`][distilabel.steps.Step] that includes the [`LLM`][distilabel.llms.LLM] as a mandatory argument. As with a [`Step`][distilabel.steps.Step], it is normally used within a [`Pipeline`][distilabel.pipeline.Pipeline] but can also be used standalone.

For example, the most basic task is the [`TextGeneration`][distilabel.steps.tasks.TextGeneration] task, which generates text based on a given instruction.

```python
from distilabel.llms import InferenceEndpointsLLM
from distilabel.steps.tasks import TextGeneration

task = TextGeneration(
    name="text-generation",
    llm=InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3-70B-Instruct",
        tokenizer_id="meta-llama/Meta-Llama-3-70B-Instruct",
    ),
)
task.load()

next(task.process([{"instruction": "What's the capital of Spain?"}]))
# [
#     {
#         'instruction': "What's the capital of Spain?",
#         'generation': 'The capital of Spain is Madrid.',
#         'distilabel_metadata': {
#               'raw_output_text-generation': 'The capital of Spain is Madrid.',
#               'raw_input_text-generation': [
#                   {'role': 'user', 'content': "What's the capital of Spain?"}
#               ]
#         },
#         'model_name': 'meta-llama/Meta-Llama-3-70B-Instruct'
#     }
# ]
```

!!! NOTE
    The `Step.load()` always needs to be executed when being used as a standalone. Within a pipeline, this will be done automatically during pipeline execution.

As shown above, the [`TextGeneration`][distilabel.steps.tasks.TextGeneration] task adds a `generation` based on the `instruction`.

!!! Tip
    Since version `1.2.0`, we provide some metadata about the LLM call through `distilabel_metadata`. This can be disabled by setting the `add_raw_output` attribute to `False` when creating the task.

    Additionally, since version `1.4.0`, the formatted input can also be included, which can be helpful when testing
    custom templates (testing the pipeline using the [`dry_run`][distilabel.pipeline.local.Pipeline.dry_run] method).

    ```python title="disable raw input and output"
    task = TextGeneration(
        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        ),
        add_raw_output=False,
        add_raw_input=False
    )
    ```

## Specifying the number of generations and grouping generations

All the `Task`s have a `num_generations` attribute that allows defining the number of generations that we want to have per input. We can update the example above to generate 3 completions per input:

```python
from distilabel.llms import InferenceEndpointsLLM
from distilabel.steps.tasks import TextGeneration

task = TextGeneration(
    name="text-generation",
    llm=InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3-70B-Instruct",
        tokenizer_id="meta-llama/Meta-Llama-3-70B-Instruct",
    ),
    num_generations=3,
)
task.load()

next(task.process([{"instruction": "What's the capital of Spain?"}]))
# [
#     {
#         'instruction': "What's the capital of Spain?",
#         'generation': 'The capital of Spain is Madrid.',
#         'distilabel_metadata': {'raw_output_text-generation': 'The capital of Spain is Madrid.'},
#         'model_name': 'meta-llama/Meta-Llama-3-70B-Instruct'
#     },
#     {
#         'instruction': "What's the capital of Spain?",
#         'generation': 'The capital of Spain is Madrid.',
#         'distilabel_metadata': {'raw_output_text-generation': 'The capital of Spain is Madrid.'},
#         'model_name': 'meta-llama/Meta-Llama-3-70B-Instruct'
#     },
#     {
#         'instruction': "What's the capital of Spain?",
#         'generation': 'The capital of Spain is Madrid.',
#         'distilabel_metadata': {'raw_output_text-generation': 'The capital of Spain is Madrid.'},
#         'model_name': 'meta-llama/Meta-Llama-3-70B-Instruct'
#     }
# ]
```

In addition, we might want to group the generations in a single output row as maybe one downstream step expects a single row with multiple generations. We can achieve this by setting the `group_generations` attribute to `True`:

```python
from distilabel.llms import InferenceEndpointsLLM
from distilabel.steps.tasks import TextGeneration

task = TextGeneration(
    name="text-generation",
    llm=InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3-70B-Instruct",
        tokenizer_id="meta-llama/Meta-Llama-3-70B-Instruct",
    ),
    num_generations=3,
    group_generations=True
)
task.load()

next(task.process([{"instruction": "What's the capital of Spain?"}]))
# [
#     {
#         'instruction': "What's the capital of Spain?",
#         'generation': ['The capital of Spain is Madrid.', 'The capital of Spain is Madrid.', 'The capital of Spain is Madrid.'],
#         'distilabel_metadata': [
#             {'raw_output_text-generation': 'The capital of Spain is Madrid.'},
#             {'raw_output_text-generation': 'The capital of Spain is Madrid.'},
#             {'raw_output_text-generation': 'The capital of Spain is Madrid.'}
#         ],
#         'model_name': 'meta-llama/Meta-Llama-3-70B-Instruct'
#     }
# ]
```

## Defining custom Tasks

We can define a custom step by creating a new subclass of the [`Task`][distilabel.steps.tasks.Task] and defining the following:

- `inputs`: is a property that returns a list of strings with the names of the required input fields or a dictionary in which the keys are the names of the columns and the values are boolean indicating whether the column is required or not.

- `format_input`: is a method that receives a dictionary with the input data and returns a [`ChatType`][distilabel.steps.tasks.ChatType] following [the chat-completion OpenAI message formatting](https://platform.openai.com/docs/guides/text-generation).

- `outputs`: is a property that returns a list of strings with the names of the output fields or a dictionary in which the keys are the names of the columns and the values are boolean indicating whether the column is required or not. This property should always include `model_name` as one of the outputs since that's automatically injected from the LLM.

- `format_output`: is a method that receives the output from the [`LLM`][distilabel.llms.LLM] and optionally also the input data (which may be useful to build the output in some scenarios), and returns a dictionary with the output data formatted as needed i.e. with the values for the columns in `outputs`. Note that there's no need to include the `model_name` in the output.

```python
from typing import Any, Dict, List, Union, TYPE_CHECKING

from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.steps.typing import StepColumns
    from distilabel.steps.tasks.typing import ChatType


class MyCustomTask(Task):
    @property
    def inputs(self) -> "StepColumns":
        return ["input_field"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        return [
            {
                "role": "user",
                "content": input["input_field"],
            },
        ]

    @property
    def outputs(self) -> "StepColumns":
        return ["output_field", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"output_field": output}
```
