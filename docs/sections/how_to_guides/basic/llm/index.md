# Executing Tasks with LLMs

## Working with LLMs

LLM subclasses are designed to be used within a [Task][distilabel.steps.tasks.Task], but they can also be used standalone.

```python
from distilabel.models import InferenceEndpointsLLM

llm = InferenceEndpointsLLM(
    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct"
)
llm.load()

llm.generate_outputs(
    inputs=[
        [{"role": "user", "content": "What's the capital of Spain?"}],
    ],
)
# [
#   {
#     "generations": [
#       "The capital of Spain is Madrid."
#     ],
#     "statistics": {
#       "input_tokens": [
#         43
#       ],
#       "output_tokens": [
#         8
#       ]
#     }
#   }
# ]
```

!!! Note
    Always call the `LLM.load` or `Task.load` method when using LLMs standalone or as part of a `Task`. If using a `Pipeline`, this is done automatically in `Pipeline.run()`.

!!! Tip "New in version 1.5.0"
    Since version `1.5.0` the LLM output is a list of dictionaries (one per item in the `inputs`),
    each containing `generations`, that reports the text returned by the `LLM`, and a `statistics` field that will store statistics related to the `LLM` generation. Initially, this will include
    `input_tokens` and `output_tokens` when available, which will be obtained via the API when available, or if a tokenizer is available for the model used, using the tokenizer for the model.
    This data will be moved by the corresponding `Task` during the pipeline processing and moved to `distilabel_metadata` so we can operate on this data if we want, like for example computing the number of tokens per dataset.

    To access to the previous result one just has to access to the generations in the resulting dictionary: `result[0]["generations"]`.

### Offline Batch Generation

By default, all `LLM`s will generate text in a synchronous manner i.e. send inputs using `generate_outputs` method that will get blocked until outputs are generated. There are some `LLM`s (such as [OpenAILLM][distilabel.models.llms.openai.OpenAILLM]) that implements what we denote as _offline batch generation_, which allows to send the inputs to the LLM-as-a-service which will generate the outputs asynchronously and give us a job id that we can use later to check the status and retrieve the generated outputs when they are ready. LLM-as-a-service platforms offers this feature as a way to save costs in exchange of waiting for the outputs to be generated.

To use this feature in `distilabel` the only thing we need to do is to set the `use_offline_batch_generation` attribute to `True` when creating the `LLM` instance:

```python
from distilabel.models import OpenAILLM

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
# [{'generations': ['The capital of Spain is Madrid.'],
#   'statistics': {'input_tokens': [13], 'output_tokens': [7]}}]
```

1. At first the `jobs_ids` attribute is `None`.
2. The first call to `generate_outputs` will send the inputs to the LLM-as-a-service and return a `DistilabelOfflineBatchGenerationNotFinishedException` since the outputs are not ready yet.
3. After the first call to `generate_outputs` the `jobs_ids` attribute will contain the job ids created for generating the outputs.
4. The second call or subsequent calls to `generate_outputs` will return the outputs if they are ready or raise a `DistilabelOfflineBatchGenerationNotFinishedException` if they are not ready yet.

The `offline_batch_generation_block_until_done` attribute can be used to block the `generate_outputs` method until the outputs are ready polling the platform the specified amount of seconds.

```python
from distilabel.models import OpenAILLM

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
# [{'generations': ['The capital of Spain is Madrid.'],
#   'statistics': {'input_tokens': [13], 'output_tokens': [7]}}]
```

### Within a Task

Pass the LLM as an argument to the [`Task`][distilabel.steps.tasks.Task], and the task will handle the rest.

```python
from distilabel.models import OpenAILLM
from distilabel.steps.tasks import TextGeneration

llm = OpenAILLM(model="gpt-4o-mini")
task = TextGeneration(name="text_generation", llm=llm)

task.load()

next(task.process(inputs=[{"instruction": "What's the capital of Spain?"}]))
# [{'instruction': "What's the capital of Spain?",
#   'generation': 'The capital of Spain is Madrid.',
#   'distilabel_metadata': {'raw_output_text_generation': 'The capital of Spain is Madrid.',
#    'raw_input_text_generation': [{'role': 'user',
#      'content': "What's the capital of Spain?"}],
#    'statistics_text_generation': {'input_tokens': 13, 'output_tokens': 7}},
#   'model_name': 'gpt-4o-mini'}]
```

!!! Note
    As mentioned in *Working with LLMs* section, the generation of an LLM is automatically moved to `distilabel_metadata` to avoid interference with the common workflow, so the addition of the `statistics` it's an extra component available for the user, but nothing has to be changed in the
    defined pipelines.

### Runtime Parameters

LLMs can have runtime parameters, such as `generation_kwargs`, provided via the `Pipeline.run()` method using the `params` argument.

!!! Note
    Runtime parameters can differ between LLM subclasses, caused by the different functionalities offered by the LLM providers.

```python
from distilabel.pipeline import Pipeline
from distilabel.models import OpenAILLM
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration

with Pipeline(name="text-generation-pipeline") as pipeline:
    load_dataset = LoadDataFromDicts(
        name="load_dataset",
        data=[{"instruction": "Write a short story about a dragon that saves a princess from a tower."}],
    )

    text_generation = TextGeneration(
        name="text_generation",
        llm=OpenAILLM(model="gpt-4o-mini"),
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

To create custom LLMs, subclass either [`LLM`][distilabel.models.llms.LLM] for synchronous or [`AsyncLLM`][distilabel.models.llms.AsyncLLM] for asynchronous LLMs. Implement the following methods:

* `model_name`: A property containing the model's name.

* `generate`: A method that takes a list of prompts and returns generated texts.

* `agenerate`: A method that takes a single prompt and returns generated texts. This method is used within the `generate` method of the `AsyncLLM` class.

* (optional) `get_last_hidden_state`: is a method that will take a list of prompts and return a list of hidden states. This method is optional and will be used by some tasks such as the [`GenerateEmbeddings`][distilabel.steps.tasks.GenerateEmbeddings] task.


=== "Custom LLM"

    ```python
    from typing import Any

    from pydantic import validate_call

    from distilabel.models import LLM
    from distilabel.typing import GenerateOutput, HiddenState
    from distilabel.typing import ChatType

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

    from distilabel.models import AsyncLLM
    from distilabel.typing import GenerateOutput, HiddenState
    from distilabel.typing import ChatType

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

!!! Note
    To have the arguments of the `generate` and `agenerate` coerced to the expected types, the `validate_call` decorator is used, which will automatically coerce the arguments to the expected types, and raise an error if the types are not correct. This is specially useful when providing a value for an argument of `generate` or `agenerate` from the CLI, since the CLI will always provide the arguments as strings.

!!! Warning
    Additional LLMs created in `distilabel` will have to take into account how the `statistics` are generated to properly include them in the LLM output.

## Available LLMs

[Our LLM gallery](../../../../components-gallery/llms/index.md) shows a list of the available LLMs that can be used within the `distilabel` library.
