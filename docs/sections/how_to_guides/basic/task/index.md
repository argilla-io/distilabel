# Define Tasks as Steps that rely on LLMs

The [`Task`][distilabel.steps.tasks.Task] is an implementation on top of [`Step`][distilabel.steps.Step] that includes the [`LLM`][distilabel.llms.LLM] as a mandatory argument, so that the [`Task`][distilabel.steps.tasks.Task] defines both the input and output format via the `format_input` and `format_output` abstract methods, respectively; and calls the [`LLM`][distilabel.llms.LLM] to generate the text. We can see the [`Task`][distilabel.steps.tasks.Task] as an [`LLM`][distilabel.llms.LLM] powered [`Step`][distilabel.steps.Step].

## Working with Tasks

The subclasses of [`Task`][distilabel.steps.tasks.Task] are intended to be used within the scope of a [`Pipeline`][distilabel.pipeline.Pipeline], which will orchestrate the different tasks defined; but nonetheless, they can be used standalone if needed too.

For example, the most basic task is the [`TextGeneration`][distilabel.steps.tasks.TextGeneration] task, which generates text based on a given instruction, and it can be used standalone as well as within a [`Pipeline`][distilabel.pipeline.Pipeline].

```python
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
    The `load` method needs to be called ALWAYS if using the tasks as standalone, otherwise, if the [`Pipeline`][distilabel.pipeline.Pipeline] context manager is used, there's no need to call that method, since it will be automatically called on `Pipeline.run`; but in any other case the method `load` needs to be called from the parent class e.g. a [`Task`][distilabel.steps.tasks.Task] with an [`LLM`][distilabel.llms.LLM] will need to call `Task.load` to load both the task and the LLM.

As we can see in the comment of the code snippet above, the task has enriched the input dictionaries adding the `generation`, the `model_name` that was used to generate, and finally the `distilabel_metadata` dictionary that contains the raw output (without post-processing) from the LLM. In this case, the `TextGeneration` task does no post-processing, so the `generation` and the raw output is the same, but some other tasks do post-processing, which in some situations it can fail. That's why is useful to have the raw output available in the `distilabel_metadata` dictionary. If this default behaviour is not desired, then all the `Task`s has a `add_raw_output` attribute that we can set to `False` when creating the instance of the task or at run time.

## Defining custom Tasks

In order to define custom tasks, we need to inherit from the [`Task`][distilabel.steps.tasks.Task] class and implement the `format_input` and `format_output` methods, as well as setting the properties `inputs` and `outputs`, as for [`Step`][distilabel.steps.Step] subclasses.

So on, the following will need to be defined:

- `inputs`: is a property that returns a list of strings with the names of the required input fields.

- `format_input`: is a method that receives a dictionary with the input data and returns a [`ChatType`][distilabel.steps.tasks.ChatType], which is basically a list of dictionaries with the input data formatted for the [`LLM`][distilabel.llms.LLM] following [the chat-completion OpenAI formatting](https://platform.openai.com/docs/guides/text-generation). It's important to note that the [`ChatType`][distilabel.steps.tasks.ChatType] is a list of dictionaries, where each dictionary represents a turn in the conversation, and it must contain the keys `role` and `content`, and this is done like this since the [`LLM`][distilabel.llms.LLM] subclasses will format that according to the LLM used, since it's the most standard formatting.

- `outputs`: is a property that returns a list of strings with the names of the output fields. Note that since all the [`Task`][distilabel.steps.tasks.Task] subclasses are designed to work with a single [`LLM`][distilabel.llms.LLM], this property should always include `model_name` as one of the outputs, since that's automatically injected from the LLM.

- `format_output`: is a method that receives the output from the [`LLM`][distilabel.llms.LLM] and optionally also the input data (which may be useful to build the output in some scenarios), and returns a dictionary with the output data formatted as needed i.e. with the values for the columns in `outputs`. Note that there's no need to include the `model_name` in the output, since that's automatically injected from the LLM in the `process` method of the [`Task`][distilabel.steps.tasks.Task].

Once those methods have been implemented, the task can be used as any other task, and it will be able to generate text based on the input data.

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
