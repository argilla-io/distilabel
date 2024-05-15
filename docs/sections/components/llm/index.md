# LLM

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
    pipeline.run(
        parameters={
            text_generation.name: {"llm": {"generation_kwargs": {"temperature": 0.3}}},
        },
    )
```

### Structured Generation

`Distilabel` is integrated with some libraries to generate structured text i.e. to guide the [`LLM`][distilabel.llms.LLM] towards the generation of structured outputs following a JSON schema, a regex, etc.

#### Outlines

`Distilabel` integrates [`outlines`](https://outlines-dev.github.io/outlines/welcome/) within some [`LLM`][distilabel.llms.LLM] subclasses. At the moment, the following LLMs integrated with `outlines` are supported in `distilabel`: [`TransformersLLM`][distilabel.llms.TransformersLLM], [`vLLM`][distilabel.llms.vLLM] or [`LlamaCppLLM`][distilabel.llms.LlamaCppLLM], so that anyone can generate structured outputs in the form of *JSON* or a parseable *regex*.

The [`LLM`][distilabel.llms.LLM] has an argument named `structured_output`[^1] that determines how we can generate structured outputs with it, let's see an example using [`LlamaCppLLM`][distilabel.llms.LlamaCppLLM], where we initially define a `pydantic.BaseModel` schema to guide the generation of the structured output.

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    last_name: str
    id: int
```

And then we provide that schema to the `structured_output` argument of the LLM.

```python
from distilabel.llms import LlamaCppLLM

llm = LlamaCppLLM(
    model_path="./openhermes-2.5-mistral-7b.Q4_K_M.gguf"  # (1),
    n_gpu_layers=-1,
    n_ctx=1024,
    structured_output={"format": "json", "schema": User},
)
llm.load()
```

1. We have previously downloaded a GGUF model i.e. `llama.cpp` compatible, from the Hugging Face Hub using curl[^2], but any model can be used as replacement, as long as the `model_path` argument is updated.

The `format` argument can be one of *json* or *regex*, and the schema represents the schema in case of JSON, which can be informed as a `str` with the `json_schema` (ADD EXAMPLES HERE), or a `pydantic.BaseModel` class with the structure, while if we work with *regex* it represents the pattern we want our text to adhere to. Let's call the LLM with a example to see it in action.

```python
import json

result = llm.generate(
    [[{"role": "user", "content": "Create a user profile for the following marathon"}]],
    max_new_tokens=50
)

json.loads(result[0][0])
# {'name': 'Kathy', 'last_name': 'Smith', 'id': 4539210}
```

We get back a python dictionary (formatted as a string) that we can parse using `json.loads`.

And the following example shows regex case. WHAT ARE WE DOING HERE?

```python
pattern = r"<name>(.*?)</name>.*?<grade>(.*?)</grade>"

llm=LlamaCppLLM(
    model_path=model_path,
    n_gpu_layers=-1,
    n_ctx=1024,
    structured_output={"format": "regex", "schema": pattern},
)
llm.load()

result = llm.generate(
    [[
        {"role": "system", "content": "You are Simpsons' fans who loves assigning grades from A to E, where A is the best and E is the worst."},
        {"role": "user", "content": "What's up with Homer Simpson?"}
    ]],
    max_new_tokens=200
)
```

We can check the output by parsing the content using the same pattern we required from the LLM.

```python
import re
match = re.search(pattern, result[0][0])

if match:
    name = match.group(1)
    grade = match.group(2)
    print(f"Name: {name}, Grade: {grade}")
# Name: Homer Simpson, Grade: C+
```

These were some simple examples, but one can see the options this opens.

!!! NOTE
    A complete pipeline can using [`LlamaCppLLM`][distilabel.llms.LlamaCppLLM] and *JSON* be seen in the [`examples/rpg_characters_llamacpp.py`](https://github.com/argilla-io/distilabel/tree/main/examples/rpg_characters_llamacpp.py) script.


[^1]:
    You can check the variable type by importing it from:

    ```python
    from distilabel.steps.tasks.structured_outputs.outlines import StructuredOutputType
    ```

[^2]:
    Download the model with curl:

    ```bash
    curl -L -o ~/Downloads/openhermes-2.5-mistral-7b.Q4_K_M.gguf https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf
    ```

#### OpenAI JSON

OpenAI offers a [JSON Mode](https://platform.openai.com/docs/guides/text-generation/json-mode) to deal with structured output via their API, let's see how to make use of them. The JSON mode instructs the model to always return a JSON object following the instruction required.

!!! WARNING
    Bear in mind, that in order for this to work, you must instruct the model in some way to generate JSON, either in the `system message` or in the instruction, as can be seen in the [API reference](https://platform.openai.com/docs/guides/text-generation/json-mode).

Contrary to what we have via `outlines`, JSON mode will not guarantee the output matches any specific schema, only that it is valid and parses without errors. More information can be found the OpenAI documentation.

Other than the reference to generating JSON, to ensure the model generates parseable JSON we can pass the argument `response_format="json"`[^3]:

```python
from distilabel.llms import OpenAILLM
llm = OpenAILLM(model="gpt4-turbo", api_key="api.key")
llm.generate(..., response_format="json")
```

[^3]:
    Keep in mind that to interact with this `response_format` argument in a pipeline, you will have to pass it via the `generation_kwargs`:

    ```python
    # Assuming a pipeline is already defined, and we have a task using OpenAILLM called `task_with_openai`:
    pipeline.run(
        parameters={
            "task_with_openai": {
                "llm": {
                    "generation_kwargs": {
                        "response_format": "json"
                    }
                }
            }
        }
    )
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
