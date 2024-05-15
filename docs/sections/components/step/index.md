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

In order to define custom steps, we need to create a new subclass of the [`Step`][distilabel.steps.Step] class, and set both the `inputs` and `outputs` property, as well as the `process` method.

So on, the following will need to be defined:

- `inputs`: is a property that returns a list of strings with the names of the required input fields.

- `outputs`: is a property that returns a list of strings with the names of the output fields.

- `process`: is a method that receives the input data and returns the output data, and it should be a generator, meaning that it should `yield` the output data. It's important to preserve the default signature within the method `def process(self, *inputs: StepInput) -> StepOutput`, since that's the one that will be used by the [`Pipeline`][distilabel.pipeline.Pipeline] to orchestrate the steps, meaning that the argument `inputs` should be respected, no more arguments can be provided, and the type-hints and return type-hints should be respected too.

!!! NOTE
    The default signature for the `process` method is `process(self, *inputs: StepInput) -> StepOutput`, meaning that it should be able to receive any number of inputs by default i.e. more than one [`Step`][distilabel.steps.Step] at a time could be connected to the current one. Anyway, when defining custom steps, that can be overridden with `process(self, inputs: StepInput) -> StepOutput`, so that the `process` method only receives the outputs from one previous [`Step`][distilabel.steps.Step] connected to it.

!!! WARNING
    For the custom [`Step`][distilabel.steps.Step] subclasses to work properly with `distilabel` and with the validation and serialization performed by default over each [`Step`][distilabel.steps.Step] in the [`Pipeline`][distilabel.pipeline.Pipeline], the type-hint for both [`StepInput`][distilabel.steps.StepInput] and [`StepOutput`][distilabel.steps.typing.StepOutput] should be used and not surrounded with double-quotes or imported under `typing.TYPE_CHECKING`, otherwise, the validation and/or serialization will fail.

```python
from distilabel.steps import Step, StepInput
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
            ...
            yield item

    # When overridden (ideally under the `typing_extensions.override` decorator)
    # @typing_extensions.override
    # def process(self, inputs: StepInput) -> StepOutput:
    #     for input in inputs:
    #         ...
    #     yield inputs
```

Alternatively, a simpler and more suitable way of defining custom [`Step`][distilabel.steps.Step] subclasses is via the `@step` decorator, which will take care of the boilerplate code, and will allow to define the `inputs`, `outputs`, and `process` methods in a more straightforward way.

```python
from distilabel.steps import StepInput, step
from distilabel.steps.typing import StepOutput

@step(inputs=[...], outputs=[...])
def CustomStep(inputs: StepInput) -> StepOutput:
    for input in inputs:
        ...
    yield inputs

step = CustomStep(name="my-step")
```

!!! WARNING
    One downside of the `@step` decorator is that it won't let you access the `self` attributes if any, neither set those, so if you need to access or set any attribute, you should go with the first approach of defining the custom [`Step`][distilabel.steps.Step] subclass.

## Available Steps

### Data Loading

For loading data as a first step of the [`Pipeline`][distilabel.pipeline.Pipeline] i.e. as a [`GeneratorStep`][distilabel.steps.GeneratorStep], the following steps are available:

#### [`LoadDataFromDicts`][distilabel.steps.LoadDataFromDicts]

Loads data from a list of dictionaries, where each dictionary represents a row in the dataset.

```python
from distilabel.steps import LoadDataFromDicts
from distilabel.pipeline import Pipeline

with Pipeline(name="my-pipeline") as pipeline:
    load_data = LoadDataFromDicts(
        name="load_data",
        data=[
            {"instruction": "Tell me a joke."},
            ...,
        ],
    )

    ...

    load_data >> ...
```

#### [`LoadHubDataset`][distilabel.steps.LoadHubDataset]

Loads data from a Hugging Face Hub dataset, and then streams the batches to the follow up steps.

```python
from distilabel.steps import LoadHubDataset
from distilabel.pipeline import Pipeline

with Pipeline(name="my-pipeline") as pipeline:
    load_data = LoadHubDataset(
        name="load_data",
        repo_id="distilabel-internal-testing/instruction-dataset-mini",
        split="train",
    )

    ...

    load_data >> ...
```

### Columns

Also, `distilabel` provides a collection of light steps that can be used to manipulate the columns in the batches in an easy way, since the following operations may be common, so we remove the hussle of defining new steps for that:

#### [`KeepColumns`][distilabel.steps.KeepColumns]

Keeps only the columns specified in the `columns` attribute.

For example, the example below will receive a batch with the columns `instruction` and `generation`, and will only keep the `instruction` column, which means the `generation` column will be dumped.

```python
from distilabel.steps import KeepColumns, LoadDataFromDicts
from distilabel.pipeline import Pipeline

with Pipeline(name="my-pipeline") as pipeline:
    load_data = LoadDataFromDicts(
        name="load_data",
        data=[
            {"instruction": "Tell me a joke.", "generation": "Why did the chicken cross the road?"},
            ...,
        ],
    )
    keep_columns = KeepColumns(
        name="keep_columns",
        columns=["instruction"],
    )
    load_data >> keep_columns
```

This [`Step`][distilabel.steps.Step] is really useful whenever either the dataset loaded contains columns we want to discard or whenever we generate intermediate columns that are not useful neither for the next steps nor for the final output.

#### [`CombineColumns`][distilabel.steps.CombineColumns]

Combines the columns specified in the `columns` attribute into a new column specified in the `output_columns` attribute. But note that this step will only work when multiple steps are connected to it, since it will need the outputs from the previous steps to combine the columns.

For example, the example below will receive a batch with data from two previous steps, both containing the column `instruction`, and this step will combine both into a column named `instructions`, so that the content of that column is not a string, but a list of strings.

```python
from distilabel.steps import CombineColumns, LoadDataFromDicts
from distilabel.pipeline import Pipeline

with Pipeline(name="my-pipeline") as pipeline:
    load_data_a = LoadDataFromDicts(
        name="load_data_a",
        data=[
            {"instruction": "Tell me a joke."},
            ...,
        ],
    )
    load_data_b = LoadDataFromDicts(
        name="load_data_b",
        data=[
            {"instruction": "Tell me another joke."},
            ...,
        ],
    )
    combine_columns = CombineColumns(
        name="combine_columns",
        columns=["instruction"],
        new_column=["instructions"],
    )
    [load_data_a, load_data_b] >> combine_columns
```

This [`Step`][distilabel.steps.Step] always needs to be connected to multiple previous steps, since it will need the outputs from the previous steps to combine the columns.

#### [`ExpandColumns`][distilabel.steps.ExpandColumns]

Expands the columns that contain a list within a batch into multiple rows, so that if the batch contains a column with 10 items, then 10 new rows will be returned as output, assuming a batch size of 1.

For example, the example below will receive a batch with the column `instructions` containing a list of strings, and will return a batch with multiple rows, one for each item in the list.

```python
from distilabel.steps import ExpandColumns, LoadDataFromDicts
from distilabel.pipeline import Pipeline

with Pipeline(name="my-pipeline") as pipeline:
    load_data = LoadDataFromDicts(
        name="load_data",
        data=[
            {"instructions": ["Tell me a joke.", "Tell me another joke."]},
            ...,
        ],
    )
    expand_columns = ExpandColumns(
        name="expand_columns",
        columns={"instructions": "instruction"},
    )
    load_data >> expand_columns
```

!!! WARNING
    This [`Step`][distilabel.steps.Step] will only work when the column to be expanded contains a list, otherwise, it will raise an error. And if multiple columns are provided but the length of those columns differs, then the length of the longest column will be used as the length of the batch.

---

!!! NOTE
    Also bear in mind that for column renaming, which may also be a common operation, every [`Step`][distilabel.steps.Step] (and so on, every subclass i.e. [`GeneratorStep`][distilabel.steps.GeneratorStep], [`GlobalStep`][distilabel.steps.GlobalStep], and [`Task`][distilabel.steps.tasks.Task]) has both `input_mappings` and `output_mappings` attributes, that can be used to rename the columns in each batch.

### Pushing Data

At the moment, the intermediate data and the final data is stored in disk under the `.cache/distilabel/pipeline/<UUID>` directory generated at the beginning of each `Pipeline.run` call, and the data is stored in the form of `.parquet` files, which are the most suitable format for storing tabular data.

Anyway, the `Pipeline.run` method will return a [`Distiset`][distilabel.distiset.Distiset] which is a custom `datasets.DatasetDict` from Hugging Face, that can be pushed to the Hugging Face Hub via the `push_to_hub` method.

Alternatively, we offer the following step to push the data to the Hugging Face Hub which can be used to store intermediate datasets without having to wait for the `Pipeline.run` to finish:

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

### Miscellaneous

These steps are intended to be used in specific scenarios, and they are not intended to be used in general cases, but they are provided as examples of how to create custom steps.

#### [`DeitaFiltering`][distilabel.steps.DeitaFiltering]

This is a step created for the [`DEITA`](/sections/papers/deita) implementation, so as to filter a dataset based on the DEITA score and the cosine distance between the generated embeddings.

To see a fully working example, please check the [Examples -> Papers -> DEITA](/sections/papers/deita).
