# GlobalStep

The [`GlobalStep`][distilabel.steps.GlobalStep] is a subclass of [`Step`][distilabel.steps.Step] that is used to define a step that requires the previous steps to be completed to run, since it will wait until all the input batches are received before running. This step is useful when you need to run a step that requires all the input data to be processed before running.

## Working with GlobalSteps

The [`GlobalStep`][distilabel.steps.GlobalStep] is intended to be used within the scope of a [`Pipeline`][distilabel.pipeline.Pipeline] and after some previous steps have been defined. Alternatively, it can also be used as a standalone [`Step`][distilabel.steps.Step] if needed, but then using [`Step`][distilabel.steps.Step] instead would be more appropriate.

## Defining custom GlobalSteps

In order to define custom steps, we need to create a new subclass of the [`GlobalStep`][distilabel.steps.GlobalStep] class, and set both the `inputs` and `outputs` property, as well as the `process` method.

So on, the following will need to be defined:

- `inputs`: is a property that returns a list of strings with the names of the required input fields.

- `outputs`: is a property that returns a list of strings with the names of the output fields.

- `process`: is a method that receives the input data and returns the output data, and it should be a generator, meaning that it should `yield` the output data. It's important to preserve the default signature within the method `def process(self, *inputs: StepInput) -> StepOutput`, since that's the one that will be used by the [`Pipeline`][distilabel.pipeline.Pipeline] to orchestrate the steps, meaning that the argument `inputs` should be respected, no more arguments can be provided, and the type-hints and return type-hints should be respected too.

!!! NOTE
    The default signature for the `process` method is `process(self, *inputs: StepInput) -> StepOutput`, meaning that it should be able to receive any number of inputs by default i.e. more than one [`Step`][distilabel.steps.Step] at a time could be connected to the current one. Anyway, when defining custom steps, that can be overridden with `process(self, inputs: StepInput) -> StepOutput`, so that the `process` method only receives the outputs from one previous [`Step`][distilabel.steps.Step] connected to it.

!!! WARNING
    For the custom [`GlobalStep`][distilabel.steps.GlobalStep] subclasses to work properly with `distilabel` and with the validation and serialization performed by default over each [`Step`][distilabel.steps.Step] in the [`Pipeline`][distilabel.pipeline.Pipeline], the type-hint for both [`StepInput`][distilabel.steps.StepInput] and [`StepOutput`][distilabel.steps.typing.StepOutput] should be used and not surrounded with double-quotes or imported under `typing.TYPE_CHECKING`, otherwise, the validation and/or serialization will fail.

```python
from distilabel.steps import GlobalStep, StepInput
from distilabel.steps.typing import StepOutput

class CustomStep(Step):
    @property
    def inputs(self) -> List[str]:
        ...

    @property
    def outputs(self) -> List[str]:
        ...

    def process(self, *inputs: StepInput) -> StepOutput:
        for input in inputs:
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

Alternatively, a simpler and more suitable way of defining custom [`GlobalStep`][distilabel.steps.GlobalStep] subclasses is via the `@step` decorator with the `step_type="global"`, which will take care of the boilerplate code, and will allow to define the `inputs`, `outputs`, and `process` methods in a more straightforward way.

```python
from distilabel.steps import StepInput, step
from distilabel.steps.typing import StepOutput

@step(inputs=[...], outputs=[...], step_type="global")
def CustomStep(inputs: StepInput) -> StepOutput:
    for input in inputs:
        ...
    yield inputs

step = CustomStep(name="my-step")
```

!!! WARNING
    One downside of the `@step` decorator is that it won't let you access the `self` attributes if any, neither set those, so if you need to access or set any attribute, you should go with the first approach of defining the custom [`GlobalStep`][distilabel.steps.GlobalStep] subclass.

## Available GlobalSteps

These have already been mentioned in [Components - Step (Available Steps)](../step/index.md#available-steps) section, but only the following are available:

#### [`PushToHub`][distilabel.steps.PushToHub]

This is a [`GlobalStep`][distilabel.steps.GlobalStep] (i.e. needs to wait for all the incoming steps to finish before running) that will push all the data generated from the previous steps to the Hugging Face Hub.

```python
from distilabel.steps import PushToHub, LoadDataFromDicts
from distilabel.pipeline import Pipeline

with Pipeline(name="my-pipeline") as pipeline:
    load_data = LoadDataFromDicts(
        name="load_data",
        data=[
            {"instruction": "Tell me a joke."},
            ...,
        ],
    )
    push_to_hub = PushToHub(
        name="push_to_hub",
        repo_id="my-distilabel-dataset",
        split="train",
    )
    load_data >> push_to_hub
```

!!! WARNING
    The `PushToHub` step will only work when the `Pipeline.run` method is called, since it will need the data generated from the previous steps to push it to the Hugging Face Hub.

#### [`DeitaFiltering`][distilabel.steps.DeitaFiltering]

This is a step created for the [`DEITA`](../../examples/papers/deita.md) implementation, so as to filter a dataset based on the DEITA score and the cosine distance between the generated embeddings.

To see a fully working example, please check the [Examples - Papers - DEITA](../../examples/papers/deita.md).
