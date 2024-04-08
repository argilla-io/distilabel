# Steps

The [`Step`][distilabel.steps.base.Step] is the base class in `distilabel`, every unit of work in a `Pipeline` will inherit from it.

## What's a Step in distilabel?

This a base class is in charge of processing data, which in the end will be lists of dictionaries. In order to process, it will define the data, it defines two properties: `inputs` and `outputs`, which are a list of strings that represent the names of the columns that the step needs as input or output respectively.

Every `Step` is connected to a `Pipeline`, which in practice means that we will build them in the context of a `Pipeline`.

Lastly, these steps inherit from `pydantic.BaseModel`, so all the attributes of a step will be validated upon definition.

## An example: ConversationTemplate

Let's see one simple type of step as an example, the [`ConversationTemplate`][distilabel.steps.conversation.ConversationTemplate]. Let's take a look at it's definition (the docstrings are removed for clarity, but it can be reviewd in the API reference):

```python
class ConversationTemplate(Step):

    @property
    def inputs(self) -> List[str]:
        return ["instruction", "response"]

    @property
    def outputs(self) -> List[str]:
        return ["conversation"]

    def process(self, inputs: StepInput) -> "StepOutput":
        for input in inputs:
            input["conversation"] = [
                {"role": "user", "content": input["instruction"]},
                {"role": "assistant", "content": input["response"]},
            ]
        yield inputs
```

At the very minimal, we need to define the `inputs` and `outputs` properties with the column names required as input, and returned as output respectively, and the processing logic of the step in the `process` method.

In this example, we see that it takes `inputs` as argument, annotated as `StepInput`, which is a list of dictionaries with the data, and *yields* a `StepOutput`.

### Working with the step

Let's see how to instantiate this `Step` outside of a `Pipeline`:

```python
from distilabel.pipeline.local import Pipeline
from distilabel.steps.conversation import ConversationTemplate

conversation_template = ConversationTemplate(
    name="my-conversation-template",
    pipeline=Pipeline(name="my-pipeline"),
)
```

As we mentioned, every `Step` must be defined in the context of a `Pipeline`, which means we need to pass it as an argument if we decide to create a standalone step. If we take a look at the `conversation_template` step, we see the following fields:

```python
conversation_template
# ConversationTemplate(name='my-conversation-template', input_mappings={}, output_mappings={}, input_batch_size=50)
```

The `name` of the `Step`, a mandatory field to identify the `Step` within the `Pipeline`. `input_mappings`, which is a dictionary that can be useful when the names of the output columns of the current step differ from the names of the input columns of the following step/s, and `output_mappings` which can be used to verify the output columns of this step exist after processing the data. If this two fields are left to the default (without content), the different steps will assume the column names along the different steps will be present, and raise an error if that's not the case. And `input_batch_size` (by default set to 50), which is independent for every step and will determine how often it can be executed. If won't matter that much in this step, but as we will see later, other types of steps will come with an `LLM`, so having this flexibility will be really useful.

### Processing data

Internally, the `Pipeline` will call the `process` method when appropriate, but we can see it in action with some dummy data:

```python
next(conversation_template.process([{"instruction": "Hello", "response": "Hi"}]))
# [
#   {
#     "instruction": "Hello",
#     "response": "Hi",
#     "conversation": [
#       {
#         "role": "user",
#         "content": "Hello"
#       },
#       {
#         "role": "assistant",
#         "content": "Hi"
#       }
#     ]
#   }
# ]
```

It takes the dictionary with data, adds another `conversation` with the data formatted as a conversation template, and passes this data to the following step.

This is a small type step that shows what to expect when we are creating our `Step` objects, which can start from something as simple as generating a conversation template from some columns on a dataset.

## step decorator

If all that we want to apply in a step is some simple processing, it can be easier to just create a plain function, and decorate it. We can find more examples in the [API reference][distilabel.steps.decorator], but let's see how we could define the previous step as a function and use the decorator:

```python
from distilabel.steps.decorator import step
from distilabel.steps.typing import StepOutput
from distilabel.steps.base import StepInput

# Normal step
@step(inputs=["instruction", "response"], outputs=["conversation"])
def ConversationTemplate(inputs: StepInput) -> StepOutput:
    for input in inputs:
        input["conversation"] = [
            {"role": "user", "content": input["instruction"]},
            {"role": "assistant", "content": input["response"]},
        ]
    yield inputs
```

Which can be instantiated exactly the same as the `ConversationTemplate` class:

```python
conversation_template = ConversationTemplate(
    name="my-conversation-template",
    pipeline=Pipeline(name="my-pipeline"),
)
```

This `@step` decorator has a special type depending `step_type` which will be better understood once we see the different types of steps.

## Runtime Parameters

There is one extra thing to keep in mind related to how we can interact with the step's parameters, the [Runtime paramaters][distilabel.mixins.runtime_parameters]. Let's inspect them using the previous example class:

```python
print(conversation_template.runtime_parameters_names)
# {'input_batch_size': True}
```

The `ConversationTemplate` only has one `runtime_parameter`, which comes defined from the `Step` class, and can be defined as such:

```python
from distilabel.mixins.runtime_parameters import RuntimeParameter

class Step(...):
    ...
    input_batch_size: RuntimeParameter[PositiveInt] = Field(
        default=DEFAULT_INPUT_BATCH_SIZE,
        description="The number of rows that will contain the batches processed by the"
        " step.",
    )
```

When we define the `input_batch_size` as a `RuntimeParameter`, the most direct effect we can see is we have some access to some extra information, thanks to the [RuntimeParamatersMixin][distilabel.mixins.runtime_parameters.RuntimeParametersMixin]:

```python
print(conversation_template.get_runtime_parameters_info())
# [{'name': 'input_batch_size', 'optional': True, 'description': 'The number of rows that will contain the batches processed by the step.'}]
```

But other than accessing some extra information internally, we can directly interact with these parameters when we interacting or modifying the arguments of our `Steps` while running them in the context of a `Pipeline`. We will see them in action once we interact with the `Steps` inside of a `Pipeline`.

## Types of steps

Other than the general or normal steps we have seen, there are special types of steps that have a restricted behaviour compared to the general `Step`.

### Generator steps

These are steps that are able to generate data, and don't need to receive any input from previous step, as it's implied in the normal steps. The typical use for these steps will be loading data for example, as can be seen in [`LoadDataFromDicts`][distilabel.steps.generators.data]. For this type of steps we will only need to define the `process` method, and we can optionally pass a `batch_size` argument, that will determine the batch size of the generated batches.

### Global steps

Other special type of step are the global steps. These steps don't have any `inputs` or `outputs`, and their `process` method receives all the data at once instead of using batches. This kind of behavior is necessary for example to push a dataset to a specific place, or doing some filtering on the whole data before continuing with the pipeline.
