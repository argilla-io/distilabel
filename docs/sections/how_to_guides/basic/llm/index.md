# Define LLMs as local or remote models

## Working with LLMs

LLM subclasses are designed to be used within a [Task][distilabel.steps.tasks.Task], but they can also be used standalone.

```python
from distilabel.llms import InferenceEndpointsLLM

llm = InferenceEndpointsLLM(model="meta-llama/Meta-Llama-3.1-70B-Instruct")
llm.load()

llm.generate_outputs(
    inputs=[
        [{"role": "user", "content": "What's the capital of Spain?"}],
    ],
)
# "The capital of Spain is Madrid."
```

!!! NOTE
    Always call the `LLM.load` or `Task.load` method when using LLMs standalone or as part of a `Task`. If using a `Pipeline`, this is done automatically in `Pipeline.run()`.

### Offline Batch Generation

By default, all `LLM`s will generate text in a synchronous manner i.e. send inputs using `generate_outputs` method that will get blocked until outputs are generated. There are some `LLM`s (such as [OpenAILLM][distilabel.llms.openai.OpenAILLM]) that implements what we denote as _offline batch generation_, which allows to send the inputs to the LLM-as-a-service which will generate the outputs asynchronously and give us a job id that we can use later to check the status and retrieve the generated outputs when they are ready. LLM-as-a-service platforms offers this feature as a way to save costs in exchange of waiting for the outputs to be generated.

To use this feature in `distilabel` the only thing we need to do is to set the `use_offline_batch_generation` attribute to `True` when creating the `LLM` instance:

```python
from distilabel.llms import OpenAILLM

llm = OpenAILLM(
    model="gpt-4o",
    use_offline_batch_generation=True,
)

llm.load()

llm.jobs_ids  # (1)
# None

llm.generate_outputs(  # (2)
    inputs=[
        [{"role": "user", "content": "What's the capital of Spain?"}],
    ],
)
# DistilabelOfflineBatchGenerationNotFinishedException: Batch generation with jobs_ids=('batch_OGB4VjKpu2ay9nz3iiFJxt5H',) is not finished

llm.jobs_ids  # (3)
# ('batch_OGB4VjKpu2ay9nz3iiFJxt5H',)


llm.generate_outputs(  # (4)
    inputs=[
        [{"role": "user", "content": "What's the capital of Spain?"}],
    ],
)
# "The capital of Spain is Madrid."
```

1. At first the `jobs_ids` attribute is `None`.
2. The first call to `generate_outputs` will send the inputs to the LLM-as-a-service and return a `DistilabelOfflineBatchGenerationNotFinishedException` since the outputs are not ready yet.
3. After the first call to `generate_outputs` the `jobs_ids` attribute will contain the job ids created for generating the outputs.
4. The second call or subsequent calls to `generate_outputs` will return the outputs if they are ready or raise a `DistilabelOfflineBatchGenerationNotFinishedException` if they are not ready yet.

The `offline_batch_generation_block_until_done` attribute can be used to block the `generate_outputs` method until the outputs are ready polling the platform the specified amount of seconds.

```python
from distilabel.llms import OpenAILLM

llm = OpenAILLM(
    model="gpt-4o",
    use_offline_batch_generation=True,
    offline_batch_generation_block_until_done=5,  # poll for results every 5 seconds
)
llm.load()

llm.generate_outputs(
    inputs=[
        [{"role": "user", "content": "What's the capital of Spain?"}],
    ],
)
# "The capital of Spain is Madrid."
```

### Within a Task

Pass the LLM as an argument to the [`Task`][distilabel.steps.tasks.Task], and the task will handle the rest.

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

LLMs can have runtime parameters, such as `generation_kwargs`, provided via the `Pipeline.run()` method using the `params` argument.

!!! NOTE
    Runtime parameters can differ between LLM subclasses, caused by the different functionalities offered by the LLM providers.

```python
from distilabel.pipeline import Pipeline
from distilabel.llms import OpenAILLM
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration

with Pipeline(name="text-generation-pipeline") as pipeline:
    load_dataset = LoadDataFromDicts(
        name="load_dataset",
        data=[{"instruction": "Write a short story about a dragon that saves a princess from a tower."}],
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

## Creating custom LLMs

To create custom LLMs, subclass either [`LLM`][distilabel.llms.LLM] for synchronous or [`AsyncLLM`][distilabel.llms.AsyncLLM] for asynchronous LLMs. Implement the following methods:

* `model_name`: A property containing the model's name.

* `generate`: A method that takes a list of prompts and returns generated texts.

* `agenerate`: A method that takes a single prompt and returns generated texts. This method is used within the `generate` method of the `AsyncLLM` class.

* (optional) `get_last_hidden_state`: is a method that will take a list of prompts and return a list of hidden states. This method is optional and will be used by some tasks such as the [`GenerateEmbeddings`][distilabel.steps.tasks.GenerateEmbeddings] task.


=== "Custom LLM"

    ```python
    from typing import Any

    from pydantic import validate_call

    from distilabel.llms import LLM
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
    ```

=== "Custom AsyncLLM"

    ```python
    from typing import Any

    from pydantic import validate_call

    from distilabel.llms import AsyncLLM
    from distilabel.llms.typing import GenerateOutput, HiddenState
    from distilabel.steps.tasks.typing import ChatType

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

[Our LLM gallery](../../../../components-gallery/llms/index.md) shows a list of the available LLMs that can be used within the `distilabel` library.
