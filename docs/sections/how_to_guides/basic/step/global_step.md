# GlobalStep

The [`GlobalStep`][distilabel.steps.GlobalStep] is a subclass of [`Step`][distilabel.steps.Step] that is used to define a step that requires the previous steps to be completed to run, since it will wait until all the input batches are received before running. This step is useful when you need to run a step that requires all the input data to be processed before running. Alternatively, it can also be used as a standalone.

## Defining custom GlobalSteps

We can define a custom step by creating a new subclass of the [`GlobalStep`][distilabel.steps.GlobalStep] and defining the following:

- `inputs`: is a property that returns a list of strings with the names of the required input fields or a dictionary in which the keys are the names of the columns and the values are boolean indicating whether the column is required or not.

- `outputs`: is a property that returns a list of strings with the names of the output fields or a dictionary in which the keys are the names of the columns and the values are boolean indicating whether the column is required or not.

- `process`: is a method that receives the input data and returns the output data, and it should be a generator, meaning that it should `yield` the output data.

!!! NOTE
    The default signature for the `process` method is `process(self, *inputs: StepInput) -> StepOutput`. The argument `inputs` should be respected, no more arguments can be provided, and the type-hints and return type-hints should be respected too because it should be able to receive any number of inputs by default i.e. more than one [`Step`][distilabel.steps.Step] at a time could be connected to the current one.

!!! WARNING
    For the custom [`GlobalStep`][distilabel.steps.GlobalStep] subclasses to work properly with `distilabel` and with the validation and serialization performed by default over each [`Step`][distilabel.steps.Step] in the [`Pipeline`][distilabel.pipeline.Pipeline], the type-hint for both [`StepInput`][distilabel.steps.StepInput] and [`StepOutput`][distilabel.typing.StepOutput] should be used and not surrounded with double-quotes or imported under `typing.TYPE_CHECKING`, otherwise, the validation and/or serialization will fail.

=== "Inherit from `GlobalStep`"

    We can inherit from the `GlobalStep` class and define the `inputs`, `outputs`, and `process` methods as follows:

    ```python
    from typing import TYPE_CHECKING
    from distilabel.steps import GlobalStep, StepInput

    if TYPE_CHECKING:
        from distilabel.typing import StepColumns, StepOutput

    class CustomStep(Step):
        @property
        def inputs(self) -> "StepColumns":
            ...

        @property
        def outputs(self) -> "StepColumns":
            ...

        def process(self, *inputs: StepInput) -> StepOutput:
            for upstream_step_inputs in inputs:
                for item in input:
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

    The `@step` decorator will take care of the boilerplate code, and will allow to define the `inputs`, `outputs`, and `process` methods in a more straightforward way. One downside is that it won't let you access the `self` attributes if any, neither set those, so if you need to access or set any attribute, you should go with the first approach of defining the custom [`GlobalStep`][distilabel.steps.GlobalStep] subclass.

    ```python
    from typing import TYPE_CHECKING
    from distilabel.steps import StepInput, step

    if TYPE_CHECKING:
        from distilabel.typing import StepOutput

    @step(inputs=[...], outputs=[...], step_type="global")
    def CustomStep(inputs: StepInput) -> "StepOutput":
        for input in inputs:
            ...
        yield inputs

    step = CustomStep(name="my-step")
    ```
