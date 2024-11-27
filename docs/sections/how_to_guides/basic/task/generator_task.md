# GeneratorTask that produces output

## Working with GeneratorTasks

The [`GeneratorTask`][distilabel.steps.tasks.GeneratorTask] is a custom implementation of a [`Task`][distilabel.steps.tasks.Task] based on the [`GeneratorStep`][distilabel.steps.GeneratorStep]. As with a [`Task`][distilabel.steps.tasks.Task], it is normally used within a [`Pipeline`][distilabel.pipeline.Pipeline] but can also be used standalone.

!!! WARNING
    This task is still experimental and may be subject to changes in the future.

```python
from typing import Any, Dict, List, Union
from typing_extensions import override

from distilabel.steps.tasks.base import GeneratorTask
from distilabel.typing import ChatType, GeneratorOutput, StepColumns


class MyCustomTask(GeneratorTask):
    instruction: str
    outputs: StepColumns = ["output_field", "model_name"]

    @override
    def process(self, offset: int = 0) -> GeneratorOutput:
        output = self.llm.generate(
            inputs=[
                [
                    {"role": "user", "content": self.instruction},
                ],
            ],
        )
        output = {"model_name": self.llm.model_name}
        output.update(
            self.format_output(output=output, input=None)
        )
        yield output

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"output_field": output}
```

We can then use it as follows:

```python
task = MyCustomTask(
    name="custom-generation",
    instruction="Tell me a joke.",
    llm=OpenAILLM(model="gpt-4"),
)
task.load()

next(task.process())
# [{'output_field": "Why did the scarecrow win an award? Because he was outstanding!", "model_name": "gpt-4"}]
```

!!! NOTE
    Most of the times you would need to override the default `process` method, as it's suited for the standard [`Task`][distilabel.steps.tasks.Task] and not for the [`GeneratorTask`][distilabel.steps.tasks.GeneratorTask]. But within the context of the `process` function you can freely use the `llm` to generate data in any way.

!!! NOTE
    The `Step.load()` always needs to be executed when being used as a standalone. Within a pipeline, this will be done automatically during pipeline execution.

## Defining custom GeneratorTasks

We can define a custom generator task by creating a new subclass of the [`GeneratorTask`][distilabel.steps.tasks.Task] and defining the following:

- `process`: is a method that generates the data based on the [`LLM`][distilabel.models.llms.LLM] and the `instruction` provided within the class instance, and returns a dictionary with the output data formatted as needed i.e. with the values for the columns in `outputs`. Note that the `inputs` argument is not allowed in this function since this is a [`GeneratorTask`][distilabel.steps.tasks.GeneratorTask]. The signature only expects the `offset` argument, which is used to keep track of the current iteration in the generator.

- `outputs`: is an attribute that returns a list of strings with the names of the output fields, this attribute should always include `model_name` as one of the outputs since that's automatically injected from the LLM.

- `format_output`: is a method that receives the output from the [`LLM`][distilabel.models.llms.LLM] and optionally also the input data (which may be useful to build the output in some scenarios), and returns a dictionary with the output data formatted as needed i.e. with the values for the columns in `outputs`. Note that there's no need to include the `model_name` in the output.

```python
from typing import Any, Dict, List, Union

from distilabel.steps.tasks.base import GeneratorTask
from distilabel.typing import ChatType, StepColumns


class MyCustomTask(GeneratorTask):
    outputs: StepColumns = ["output_field", "model_name"]

    @override
    def process(self, offset: int = 0) -> GeneratorOutput:
        output = self.llm.generate(
            inputs=[
                [{"role": "user", "content": "Tell me a joke."}],
            ],
        )
        output = {"model_name": self.llm.model_name}
        output.update(
            self.format_output(output=output, input=None)
        )
        yield output

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"output_field": output}
```
