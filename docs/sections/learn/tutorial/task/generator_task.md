# GeneratorTask

The [`GeneratorTask`][distilabel.steps.tasks.GeneratorTask] is a custom implementation of a [`Task`][distilabel.steps.tasks.Task], but based on [`GeneratorStep`][distilabel.steps.GeneratorStep]; which means that will essentially be similar to the standard [`Task`][distilabel.steps.tasks.Task], but without the need of providing any input data, as the data will be generated as part of the [`GeneratorTask`][distilabel.steps.tasks.GeneratorTask] execution.

!!! WARNING
    This task is still experimental and may be subject to changes in the future, since apparently it's not the most efficient way to generate data, but it's a good way to generate data on the fly without the need of providing any input data.

## Working with GeneratorTasks

The subclasses of [`GeneratorTask`][distilabel.steps.tasks.GeneratorTask] are intended to be used within the scope of a [`Pipeline`][distilabel.pipeline.Pipeline], which will orchestrate the different tasks defined; but nonetheless, they can be used standalone if needed too.

These tasks will basically expect no input data, but generate data as part of the `process` method of the parent class. Say you have a [`GeneratorTask`][distilabel.steps.tasks.GeneratorTask] that generates text from a pre-defined instruction:

```python
from typing import Any, Dict, List, Union
from typing_extensions import override

from distilabel.steps.tasks.base import GeneratorTask
from distilabel.steps.tasks.typing import ChatType
from distilabel.steps.typing import GeneratorOutput


class MyCustomTask(GeneratorTask):
    instruction: str

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

    @property
    def outputs(self) -> List[str]:
        return ["output_field", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"output_field": output}
```

To then use it as:

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
    The `load` method needs to be called ALWAYS if using the tasks as standalone, otherwise, if the [`Pipeline`][distilabel.pipeline.Pipeline] context manager is used, there's no need to call that method, since it will be automatically called on `Pipeline.run`; but in any other case the method `load` needs to be called from the parent class e.g. a [`GeneratorTask`][distilabel.steps.tasks.Task] with an [`LLM`][distilabel.llms.LLM] will need to call `GeneratorTask.load` to load both the task and the LLM.

## Defining custom GeneratorTasks

In order to define custom tasks, we need to inherit from the [`Task`][distilabel.steps.tasks.Task] class and implement the `format_input` and `format_output` methods, as well as setting the properties `inputs` and `outputs`, as for [`Step`][distilabel.steps.Step] subclasses.

So on, the following will need to be defined:

- `process`: is a method that generates the data based on the [`LLM`][distilabel.llms.LLM] and the `instruction` provided within the class instance, and returns a dictionary with the output data formatted as needed i.e. with the values for the columns in `outputs`. Note that the `inputs` argument is not allowed in this function since this is not a [`Task`][distilabel.steps.tasks.Task] but a [`GeneratorTask`][distilabel.steps.tasks.GeneratorTask], so no input data is expected; so the signature only expects the `offset` argument, which is used to keep track of the current iteration in the generator.

- `outputs`: is a property that returns a list of strings with the names of the output fields. Note that since all the [`Task`][distilabel.steps.tasks.Task] subclasses are designed to work with a single [`LLM`][distilabel.llms.LLM], this property should always include `model_name` as one of the outputs, since that's automatically injected from the LLM.

- `format_output`: is a method that receives the output from the [`LLM`][distilabel.llms.LLM] and optionally also the input data (which may be useful to build the output in some scenarios), and returns a dictionary with the output data formatted as needed i.e. with the values for the columns in `outputs`. Note that there's no need to include the `model_name` in the output, since that's automatically injected from the LLM in the `process` method of the [`Task`][distilabel.steps.tasks.Task].

Once those methods have been implemented, the task can be used as any other task, and it will be able to generate text based on the input data.

```python
from typing import Any, Dict, List, Union

from distilabel.steps.tasks.base import GeneratorTask
from distilabel.steps.tasks.typing import ChatType


class MyCustomTask(GeneratorTask):
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

    @property
    def outputs(self) -> List[str]:
        return ["output_field", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"output_field": output}
```
