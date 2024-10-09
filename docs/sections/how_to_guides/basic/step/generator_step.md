# GeneratorStep

The [`GeneratorStep`][distilabel.steps.GeneratorStep] is a subclass of [`Step`][distilabel.steps.Step] that is intended to be used as the first step within a [`Pipeline`][distilabel.pipeline.Pipeline], because it doesn't require input and generates data that can be used by other steps. Alternatively, it can also be used as a standalone.

```python
from typing import List, TYPE_CHECKING
from typing_extensions import override

from distilabel.steps import GeneratorStep
from distilabel.steps.typing import StepColumns

if TYPE_CHECKING:
    from distilabel.steps.typing import GeneratorStepOutput

class MyGeneratorStep(GeneratorStep):
    instructions: List[str]
    outputs: StepColumns = ["instruction"]

    @override
    def process(self, offset: int = 0) -> "GeneratorStepOutput":
        if offset:
            self.instructions = self.instructions[offset:]

        while self.instructions:
            batch = [
                {
                    "instruction": instruction
                } for instruction in self.instructions[: self.batch_size]
            ]
            self.instructions = self.instructions[self.batch_size :]
            yield (
                batch,
                True if len(self.instructions) == 0 else False,
            )
```

Then we can use it as follows:

```python
step = MyGeneratorStep(
    name="my-generator-step",
    instructions=["Tell me a joke.", "Tell me a story."],
    batch_size=1,
)
step.load()

next(step.process(offset=0))
# ([{'instruction': 'Tell me a joke.'}], False)
next(step.process(offset=1))
# ([{'instruction': 'Tell me a story.'}], True)
```

!!! NOTE
    The `Step.load()` always needs to be executed when being used as a standalone. Within a pipeline, this will be done automatically during pipeline execution.

## Defining custom GeneratorSteps

We can define a custom generator step by creating a new subclass of the [`GeneratorStep`][distilabel.steps.GeneratorStep] and defining the following:

- `outputs`: is an attribute that returns a list of strings with the names of the output fields or a dictionary in which the keys are the names of the columns and the values are boolean indicating whether the column is required or not.

- `process`: is a method that yields output data and a boolean flag indicating whether that's the last batch to be generated.

!!! NOTE
    The default signature for the `process` method is `process(self, offset: int = 0) -> GeneratorStepOutput`. The argument `offset` should be respected, no more arguments can be provided, and the type-hints and return type-hints should be respected too because it should be able to receive any number of inputs by default i.e. more than one [`Step`][distilabel.steps.Step] at a time could be connected to the current one.

!!! WARNING
    For the custom [`Step`][distilabel.steps.Step] subclasses to work properly with `distilabel` and with the validation and serialization performed by default over each [`Step`][distilabel.steps.Step] in the [`Pipeline`][distilabel.pipeline.Pipeline], the type-hint for both [`StepInput`][distilabel.steps.StepInput] and [`StepOutput`][distilabel.steps.typing.StepOutput] should be used and not surrounded with double-quotes or imported under `typing.TYPE_CHECKING`, otherwise, the validation and/or serialization will fail.

=== "Inherit from `GeneratorStep`"

    We can inherit from the `GeneratorStep` class and define the `outputs`, and `process` methods as follows:


    ```python
    from typing import List, TYPE_CHECKING
    from typing_extensions import override

    from distilabel.steps import GeneratorStep
    from distilabel.steps.typing import StepColumns

    if TYPE_CHECKING:
        from distilabel.steps.typing import GeneratorStepOutput

    class MyGeneratorStep(GeneratorStep):
        instructions: List[str]
        outputs: StepColumns = ...

        @override
        def process(self, offset: int = 0) -> "GeneratorStepOutput":
            ...
    ```

=== "Using the `@step` decorator"

    The `@step` decorator will take care of the boilerplate code, and will allow to define the `outputs`, and `process` methods in a more straightforward way. One downside is that it won't let you access the `self` attributes if any, neither set those, so if you need to access or set any attribute, you should go with the first approach of defining the custom [`GeneratorStep`][distilabel.steps.GeneratorStep] subclass.

    ```python
    from typing import TYPE_CHECKING
    from distilabel.steps import step

    if TYPE_CHECKING:
        from distilabel.steps.typing import GeneratorStepOutput

    @step(outputs=[...], step_type="generator")
    def CustomGeneratorStep(offset: int = 0) -> "GeneratorStepOutput":
        yield (
            ...,
            True if offset == 10 else False,
        )

    step = CustomGeneratorStep(name="my-step")
    ```
