# Steps for processing data

## Working with Steps

The [`Step`][distilabel.steps.Step] is intended to be used within the scope of a [`Pipeline`][distilabel.pipeline.Pipeline], which will orchestrate the different steps defined but can also be used standalone.

Assuming that we have a [`Step`][distilabel.steps.Step] already defined as follows:

```python
from typing import TYPE_CHECKING
from distilabel.steps import Step, StepInput
from distilabel.typing import StepColumns

if TYPE_CHECKING:
    from distilabel.typing import StepOutput

class MyStep(Step):
    inputs: StepColumns = ["input_field"]
    outputs: StepColumns = ["output_field"]

    def process(self, inputs: StepInput) -> "StepOutput":
        for input in inputs:
            input["output_field"] = input["input_field"]
        yield inputs
```

Then we can use it as follows:

```python
step = MyStep(name="my-step")
step.load()

next(step.process([{"input_field": "value"}]))
# [{'input_field': 'value', 'output_field': 'value'}]
```
!!! NOTE
    The `Step.load()` always needs to be executed when being used as a standalone. Within a pipeline, this will be done automatically during pipeline execution.

### Arguments

- `input_mappings`, is a dictionary that maps keys from the input dictionaries to the keys expected by the step. For example, if `input_mappings={"instruction": "prompt"}`, means that the input key `prompt` will be used as the key `instruction` for current step.

- `output_mappings`, is a dictionary that can be used to map the outputs of the step to other names. For example, if `output_mappings={"conversation": "prompt"}`, means that output key `conversation` will be renamed to `prompt` for the next step.

- `input_batch_size` (by default set to 50), is independent for every step and will determine how many input dictionaries will process at once.

### Runtime parameters

`Step`s can also have `RuntimeParameter`, which are parameters that can only be used after the pipeline initialisation when calling the `Pipeline.run`.

```python
from distilabel.mixins.runtime_parameters import RuntimeParameter

class Step(...):
    input_batch_size: RuntimeParameter[PositiveInt] = Field(
        default=DEFAULT_INPUT_BATCH_SIZE,
        description="The number of rows that will contain the batches processed by the"
        " step.",
    )
```

## Types of Steps

There are two special types of [`Step`][distilabel.steps.Step] in `distilabel`:

* [`GeneratorStep`][distilabel.steps.GeneratorStep]: is a step that only generates data, and it doesn't need any input data from previous steps and normally is the first node in a [`Pipeline`][distilabel.pipeline.Pipeline]. More information: [Components -> Step - GeneratorStep](./generator_step.md).

* [`GlobalStep`][distilabel.steps.GlobalStep]: is a step with the standard interface i.e. receives inputs and generates outputs, but it processes all the data at once, and often is the final step in the [`Pipeline`][distilabel.pipeline.Pipeline]. The fact that a [`GlobalStep`][distilabel.steps.GlobalStep] requires the previous steps  to finish before being able to start. More information: [Components - Step - GlobalStep](global_step.md).

* [`Task`][distilabel.steps.tasks.Task], is essentially the same as a default [`Step`][distilabel.steps.Step], but it relies on an [`LLM`][distilabel.llms.LLM] as an attribute, and the `process` method will be in charge of calling that LLM. More information: [Components - Task](../task/index.md).

## Defining custom Steps

We can define a custom step by creating a new subclass of the [`Step`][distilabel.steps.Step] and defining the following:

- `inputs`: is an attribute that returns a list of strings with the names of the required input fields or a dictionary in which the keys are the names of the columns and the values are boolean indicating whether the column is required or not.

- `outputs`: is an attribute that returns a list of strings with the names of the output fields or a dictionary in which the keys are the names of the columns and the values are boolean indicating whether the column is required or not.

- `process`: is a method that receives the input data and returns the output data, and it should be a generator, meaning that it should `yield` the output data.

!!! NOTE
    The default signature for the `process` method is `process(self, *inputs: StepInput) -> StepOutput`. The argument `inputs` should be respected, no more arguments can be provided, and the type-hints and return type-hints should be respected too because it should be able to receive any number of inputs by default i.e. more than one [`Step`][distilabel.steps.Step] at a time could be connected to the current one.

!!! WARNING
    For the custom [`Step`][distilabel.steps.Step] subclasses to work properly with `distilabel` and with the validation and serialization performed by default over each [`Step`][distilabel.steps.Step] in the [`Pipeline`][distilabel.pipeline.Pipeline], the type-hint for both [`StepInput`][distilabel.steps.StepInput] and [`StepOutput`][distilabel.steps.typing.StepOutput] should be used and not surrounded with double-quotes or imported under `typing.TYPE_CHECKING`, otherwise, the validation and/or serialization will fail.

=== "Inherit from `Step`"

    We can inherit from the `Step` class and define the `inputs`, `outputs`, and `process` methods as follows:

    ```python
    from typing import TYPE_CHECKING
    from distilabel.steps import Step, StepInput
    from distilabel.typing import StepColumns
    
    if TYPE_CHECKING:
        from distilabel.typing import StepOutput

    class CustomStep(Step):
        inputs: StepColumns = ...
        outputs: StepColumns = ...

        def process(self, *inputs: StepInput) -> "StepOutput":
            for upstream_step_inputs in inputs:
                ...
                yield item

        # When overridden (ideally under the `typing_extensions.override` decorator)
        # @typing_extensions.override
        # def process(self, inputs: StepInput) -> StepOutput:
        #     for input in inputs:
        #         ...
        #     yield inputs
    ```

=== "Using the `@step` decorator"

    The `@step` decorator will take care of the boilerplate code, and will allow to define the `inputs`, `outputs`, and `process` methods in a more straightforward way. One downside is that it won't let you access the `self` attributes if any, neither set those, so if you need to access or set any attribute, you should go with the first approach of defining the custom [`Step`][distilabel.steps.Step] subclass.


    ```python
    from typing import TYPE_CHECKING
    from distilabel.steps import StepInput, step

    if TYPE_CHECKING:
        from distilabel.typing import StepOutput

    @step(inputs=[...], outputs=[...])
    def CustomStep(inputs: StepInput) -> "StepOutput":
        for input in inputs:
            ...
        yield inputs

    step = CustomStep(name="my-step")
    ```
