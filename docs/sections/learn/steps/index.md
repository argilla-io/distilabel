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

## Types of steps

In the next section we will see specific types of steps and how to use them.
