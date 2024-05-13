# LLMs

The LLMs are implemented as subclasses of either [`LLM`][distilabel.llms.LLM] or [`AsyncLLM`][distilabel.llms.AsyncLLM], and are only in charge of running the text generation for a given prompt or conversation. The LLMs are intended to be used together with the [`Task`][distilabel.steps.tasks.Task] and any of its subclasses, via the `llm` argument, this means that any of the implemented LLMs can be easily plugged seamlessly into any task.

## Working with LLMs

The subclasses of both [`LLM`][distilabel.llms.LLM] or [`AsyncLLM`][distilabel.llms.AsyncLLM] are intended to be used within the scope of a [`Task`][distilabel.steps.tasks.Task], since those are seamlessly integrated within the different tasks; but nonetheless, they can be used standalone if needed.

```python
from distilabel.llms import OpenAILLM

llm = OpenAILLM(model="gpt-4")
llm.load()

llm.generate(
    inputs=[
        [
            {"role": "user", "content": "What's the capital of Spain?"},
        ],
    ],
)
# "The capital of Spain is Madrid."
```

!!! NOTE
    The `load` method needs to be called ALWAYS if using the LLMs as standalone or as part of a task, otherwise, if the `Pipeline` context manager is used, there's no need to call that method, since it will be automatically called on `Pipeline.run`; but in any other case the method `load` needs to be called from the parent class e.g. a `Task` with an `LLM` will need to call `Task.load` to load both the task and the LLM.

### Within a Task

Now, in order to use the LLM within a [`Task`][distilabel.steps.tasks.Task], we need to pass it as an argument to the task, and the task will take care of the rest.

```python
from distilabel.llms import OpenAILLM
from distilabel.steps.tasks import TextGeneration


llm = OpenAILLM(model="gpt-4")
task = TextGeneration(name="text_generation", llm=llm)

task.load()

next(task.process(inputs=[{"instruction": "What's the capital of Spain?"}]))
# [{'instruction': "What's the capital of Spain?", "generation": "The capital of Spain is Madrid."}]
```

### Runtime Parameters

Additionally, besides the runtime parameters that can / need to be provided to the [`Task`][distilabel.steps.tasks], the LLMs can also define their own runtime parameters such as the `generation_kwargs`, and those need to be provided within the `Pipeline.run` method via the argument `params`.

!!! NOTE
    Each LLM subclass may have its own runtime parameters and those can differ between the different implementations, as those are not aligned, since the LLM engines offer different functionalities.

```python
from distilabel.pipeline import Pipeline
from distilabel.llms import OpenAILLM
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration


with Pipeline(name="text-generation-pipeline") as pipeline:
    load_dataset = LoadDataFromDicts(
        name="load_dataset",
        data=[
            {
                "instruction": "Write a short story about a dragon that saves a princess from a tower.",
            },
        ],
    )

    text_generation = TextGeneration(
        name="text_generation",
        llm=OpenAILLM(model="gpt-4"),
    )

    load_dataset >> text_generation

if __name__ == "__main__":
    pipeline.run(params={text_generation.name: {"llm": {"generation_kwargs": {"temperature": 0.3}}}})
```

## Defining custom LLMs

In order to define custom LLMs, one must subclass either [`LLM`][distilabel.llms.LLM] or [`AsyncLLM`][distilabel.llms.AsyncLLM], to define a synchronous or asynchronous LLM, respectively.

One can either extend any of the existing LLMs to override the default behaviour if needed, but also to define a new one from scratch, that could be potentially contributed to the `distilabel` codebase.

In order to define a new LLM, one must define the following methods:

* `model_name`: is a property that contains the name of the model to be used, which means that it needs to be retrieved from the LLM using the LLM-specific approach i.e. for [`TransformersLLM`][distilabel.llms.TransformersLLM] the `model_name` will be the `model_name_or_path` provided as an argument, or in [`OpenAILLM`][distilabel.llms.OpenAILLM] the `model_name` will be the `model` provided as an argument.

* `generate`: is a method that will take a list of prompts and return a list of generated texts. This method will be called by the [`Task`][distilabel.steps.tasks.Task] to generate the texts, so it's the most important method to define. This method will be implemented in the subclass of the [`LLM`][distilabel.llms.LLM] i.e. the synchronous LLM.

* `agenerate`: is a method that will take a single prompt and return a list of generated texts, since the rest of the behaviour will be controlled by the `generate` method that cannot be overwritten when subclassing [`AsyncLLM`][distilabel.llms.AsyncLLM]. This method will be called by the [`Task`][distilabel.steps.tasks.Task] to generate the texts, so it's the most important method to define. This method will be implemented in the subclass of the [`AsyncLLM`][distilabel.llms.AsyncLLM] i.e. the asynchronous LLM.

* (optional) `get_last_hidden_state`: is a method that will take a list of prompts and return a list of hidden states. This method is optional and will be used by some tasks such as the [`GenerateEmbeddings`][distilabel.steps.tasks.GenerateEmbeddings] task.

Once those methods have been implemented, then the custom LLM will be ready to be integrated within either any of the existing or a new task.

```python
from typing import Any

from pydantic import validate_call

from distilabel.llms import AsyncLLM, LLM
from distilabel.llms.typing import GenerateOutput, HiddenState
from distilabel.steps.tasks.typing import ChatType


class CustomLLM(LLM):
    @property
    def model_name(self) -> str:
        return "my-model"

    @validate_call
    def generate(self, inputs: List[ChatType], num_generations: int = 1, **kwargs: Any) -> List[GenerateOutput]:
        for _ in range(num_generations):
            ...

    def get_last_hidden_state(self, inputs: List[ChatType]) -> List[HiddenState]:
        ...


class CustomAsyncLLM(AsyncLLM):
    @property
    def model_name(self) -> str:
        return "my-model"

    @validate_call
    async def agenerate(self, input: ChatType, num_generations: int = 1, **kwargs: Any) -> GenerateOutput:
        for _ in range(num_generations):
            ...

    def get_last_hidden_state(self, inputs: List[ChatType]) -> List[HiddenState]:
        ...
```

`generate` and `agenerate` keyword arguments (but `input` and `num_generations`) are considered as `RuntimeParameter`s, so a value can be passed to them via the `parameters` argument of the `Pipeline.run` method.

!!! NOTE
    To have the arguments of the `generate` and `agenerate` coerced to the expected types, the `validate_call` decorator is used, which will automatically coerce the arguments to the expected types, and raise an error if the types are not correct. This is specially useful when providing a value for an argument of `generate` or `agenerate` from the CLI, since the CLI will always provide the arguments as strings.

## Available LLMs

Here's a list with the available LLMs that can be used within the `distilabel` library:

* [AnthropicLLM][distilabel.llms.AnthropicLLM]
* [AnyscaleLLM][distilabel.llms.AnyscaleLLM]
* [AzureOpenAILLM][distilabel.llms.AzureOpenAILLM]
* [CohereLLM][distilabel.llms.CohereLLM]
* [GroqLLM][distilabel.llms.GroqLLM]
* [InferenceEndpointsLLM][distilabel.llms.huggingface.InferenceEndpointsLLM]
* [LiteLLM][distilabel.llms.LiteLLM]
* [LlamaCppLLM][distilabel.llms.LlamaCppLLM]
* [MistralLLM][distilabel.llms.MistralLLM]
* [OllamaLLM][distilabel.llms.OllamaLLM]
* [OpenAILLM][distilabel.llms.OpenAILLM]
* [TogetherLLM][distilabel.llms.TogetherLLM]
* [TransformersLLM][distilabel.llms.huggingface.TransformersLLM]
* [VertexAILLM][distilabel.llms.VertexAILLM]
* [vLLM][distilabel.llms.vLLM]
