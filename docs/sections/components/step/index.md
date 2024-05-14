# Step

The [`Step`][distilabel.steps.Step] is an abstract class which defines the interface for the building blocks to be defined within the context of a [`Pipeline`][distilabel.pipeline.Pipeline], a [`Step`][distilabel.steps.Step] can be seen as a node within a Direct Acyclic Graph (DAG) which execution is orchestrated by the [`Pipeline`][distilabel.pipeline.Pipeline].

## Working with Steps

The [`Step`][distilabel.steps.Step] is intended to be used within the scope of a [`Pipeline`][distilabel.pipeline.Pipeline], which will orchestrate the different steps defined; but nonetheless, they can be used standalone if needed too.

Assuming that we have a [`Step`][distilabel.steps.Step] already defined as it follows:

```python
class MyStep(Step):
    @property
    def inputs(self) -> List[str]:
        return ["input_field"]

    @property
    def outputs(self) -> List[str]:
        return ["output_field"]

    def process(self, inputs: StepInput) -> "StepOutput":
        for input in inputs:
            input["output_field"] = input["input_field"]
        yield inputs
```

Then we can use / instantiate it as follows:

```python
step = MyStep(name="my-step")
step.load()

next(step.process([{"input_field": "value"}]))
# [{'input_field': 'value', 'output_field': 'value'}]
```
!!! NOTE
    The `load` method needs to be called ALWAYS if using the steps and any [`Step`][distilabel.steps.Step] subclass as standalone, unless the [`Pipeline`][distilabel.pipeline.Pipeline] context manager is used, meaning that there will be no need to call the `load` method, since it will be automatically called on `Pipeline.run`; but in any other case the method `load` needs to be called from the parent class.

Anyway, most of the times we'll end up using pre-defined steps in `distilabel`, so that there's no need to create custom steps, but anyway, we'll cover that later in this page.

## Types of Steps

Besides the default [`Step`][distilabel.steps.Step] already described, in `distilabel` we find the following abstract subclasses on top of the [`Step`][distilabel.steps.Step].

* [`GeneratorStep`][distilabel.steps.GeneratorStep]: is a step that only produces / generates data, and it doesn't need any input data from previous steps, is in most of the cases a parent node of the graph i.e. the first [`Step`][distilabel.steps.Step] in the [`Pipeline`][distilabel.pipeline.Pipeline].

    More information about it at [Components -> Step -> GeneratorStep](/components/step/generator-step).

* [`GlobalStep`][distilabel.steps.GlobalStep]: is a step with the standard interface i.e. receives inputs and generates outputs, but it processes all the data at once, is in most of the cases a leaf node of the graph i.e. the last [`Step`][distilabel.steps.Step] in the [`Pipeline`][distilabel.pipeline.Pipeline]. The fact that a [`GlobalStep`](distilabel.steps.GlobalStep) requires the outputs from the previous steps, means that the previous steps needs to finish for this step to start, and the connected outputs steps, if any, will need to wait until this step is done.

    More information about it at [Components -> Step -> GlobalStep](/components/step/global-step).

Additionally, `distilabel` also defines another type of [`Step`][distilabel.steps.Step], which is the [`Task`][distilabel.steps.tasks.Task], which is essentially the same, besides the fact that the task will expect an [`LLM`][distilabel.llms.LLM] as an attribute, and the `process` method will be in charge of calling that LLM. So one could say that the [`Task`][distilabel.steps.tasks.Task] is a [`Step`][distilabel.steps.Step] to work with an [`LLM`][distilabel.llms.LLM].

More information about it at [Components -> Task](/components/task).

## Defining custom Steps

...

Alternatively, a simpler and more suitable way of defining custom [`Step`][distilabel.steps.Step] subclasses is via the `@step` decorator, which will take care of the boilerplate code, and will allow to define the `inputs`, `outputs`, and `process` methods in a more straightforward way.

```python
from distilabel.steps import step

@step(inputs=["input_field"], outputs=["output_field"])
def CustomStep(inputs: StepInput) -> StepOutput:
    for input in inputs:
        input["output_field"] = input["input_field"]
    yield inputs

step = CustomStep(name="my-step")
```

## Available Steps

...
