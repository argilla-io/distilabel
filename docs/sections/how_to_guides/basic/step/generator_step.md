# GeneratorStep

The [`GeneratorStep`][distilabel.steps.GeneratorStep] is a subclass of [`Step`][distilabel.steps.Step] that is intended to be used the first step within a [`Pipeline`][distilabel.pipeline.Pipeline], because it doesn't require input and generates data that can be used by other steps. Alternatively, in can also be used as a standalone [`Step`][distilabel.steps.Step] outside a [`Pipeline`][distilabel.pipeline.Pipeline].

For example, the following code snippet shows how to use the [`GeneratorStep`][distilabel.steps.GeneratorStep] as a standalone [`Step`][distilabel.steps.Step], to generate data out of a provided list of strings.

```python
from typing import List
from typing_extensions import override

from distilabel.steps import GeneratorStep
from distilabel.steps.typing import GeneratorStepOutput

class MyGeneratorStep(GeneratorStep):
    instructions: List[str]

    @override
    def process(self, offset: int = 0) -> GeneratorStepOutput:
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

    @property
    def outputs(self) -> List[str]:
        return ["instruction"]
```

Then we can use / instantiate it as follows:

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
    The `load` method needs to be called ALWAYS if using the steps and any [`Step`][distilabel.steps.Step] subclass as standalone, unless the [`Pipeline`][distilabel.pipeline.Pipeline] context manager is used, meaning that there will be no need to call the `load` method, since it will be automatically called on `Pipeline.run`; but in any other case the method `load` needs to be called from the parent class.

Anyway, most of the times we'll end up using pre-defined steps in `distilabel`, so that there's no need to create custom steps, but anyway, we'll cover that later in this page.

## Defining custom GeneratorSteps

In order to define a custom [`GeneratorStep`][distilabel.steps.GeneratorStep], we need to subclass it, and set the `outputs` property, and define the `process` method. In this case, the `process` method signature differs from the `process` method signature of the [`Step`][distilabel.steps.Step], since it won't receive any `inputs` but generate those, so the only argument of `process` is `offset` which is automatically handled by the [`Pipeline`][distilabel.pipeline.Pipeline] shifting it until all the batches are generated.

So on, the following will need to be defined:

- `outputs`: is a property that returns a list of strings with the names of the output fields.

- `process`: is a method that yields output data and a boolean flag indicating whether that's the last batch to be generated. It's important to override the default signature of the [`Step.process`][distilabel.steps.Step] method `def process(self, *inputs: StepInput) -> StepOutput`, to be set to `def process(self, offset: int = 0) -> GeneratorStepOutput` instead, since that's the one that will be used by the [`Pipeline`][distilabel.pipeline.Pipeline] to orchestrate the steps, meaning that the argument `offset` should be respected, no more arguments can be provided, and the type-hints and return type-hints should be respected too.

!!! NOTE
    The default signature for the `process` method is `process(self, *inputs: StepInput) -> StepOutput`, but since in this case we're defining a [`GeneratorStep`][distilabel.steps.GeneratorStep], we will need to override that (ideally under the `typing_extensions.override` decorator) with `process(self, offset: int = 0) -> GeneratorStepOutput`, so that the `process` method only receives the `offset` argument, and the return type-hints should be respected too. The `offset` argument is automatically handled by the [`Pipeline`][distilabel.pipeline.Pipeline] shifting it until all the batches are generated, and there's no need to default it to 0, since it will be set to 0 by default anyway.

!!! WARNING
    For the custom [`GeneratorStep`][distilabel.steps.GeneratorStep] subclasses to work properly with `distilabel` and with the validation and serialization performed by default over each [`Step`][distilabel.steps.Step] in the [`Pipeline`][distilabel.pipeline.Pipeline], the type-hint for [`GeneratorStepOutput`][distilabel.steps.typing.GeneratorStepOutput] should be used and not surrounded with double-quotes or imported under `typing.TYPE_CHECKING`, otherwise, the validation and/or serialization will fail.

```python
from typing import List
from typing_extensions import override

from distilabel.steps import GeneratorStep
from distilabel.steps.typing import GeneratorStepOutput

class MyGeneratorStep(GeneratorStep):
    instructions: List[str]

    @override
    def process(self, offset: int = 0) -> GeneratorStepOutput:
        ...

    @property
    def outputs(self) -> List[str]:
        ...
```

Alternatively, a simpler and more suitable way of defining custom [`GeneratorStep`][distilabel.steps.GeneratorStep] subclasses is via the `@step` decorator with the `step_type="generator"`, which will take care of the boilerplate code, and will allow to define the `outputs`, and `process` methods in a more straightforward way.

```python
from distilabel.steps import step
from distilabel.steps.typing import GeneratorStepOutput

@step(outputs=[...], step_type="generator")
def CustomGeneratorStep(offset: int = 0) -> GeneratorStepOutput:
    yield (
        ...,
        True if offset == 10 else False,
    )

step = CustomGeneratorStep(name="my-step")
```

!!! WARNING
    One downside of the `@step` decorator is that it won't let you access the `self` attributes if any, neither set those, so if you need to access or set any attribute, you should go with the first approach of defining the custom [`GeneratorStep`][distilabel.steps.GeneratorStep] subclass.

