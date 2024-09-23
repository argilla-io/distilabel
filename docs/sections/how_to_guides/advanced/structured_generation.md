# Structured data generation

`Distilabel` has integrations with relevant libraries to generate structured text i.e. to guide the [`LLM`][distilabel.llms.LLM] towards the generation of structured outputs following a JSON schema, a regex, etc.

## Outlines

`Distilabel` integrates [`outlines`](https://outlines-dev.github.io/outlines/welcome/) within some [`LLM`][distilabel.llms.LLM] subclasses. At the moment, the following LLMs integrated with `outlines` are supported in `distilabel`: [`TransformersLLM`][distilabel.llms.TransformersLLM], [`vLLM`][distilabel.llms.vLLM] or [`LlamaCppLLM`][distilabel.llms.LlamaCppLLM], so that anyone can generate structured outputs in the form of *JSON* or a parseable *regex*.

The [`LLM`][distilabel.llms.LLM] has an argument named `structured_output`[^1] that determines how we can generate structured outputs with it, let's see an example using [`LlamaCppLLM`][distilabel.llms.LlamaCppLLM].

!!! Note

    For `outlines` integration to work you may need to install the corresponding dependencies:

    ```bash
    pip install distilabel[outlines]
    ```

### JSON

We will start with a JSON example, where we initially define a `pydantic.BaseModel` schema to guide the generation of the structured output.

!!! NOTE
    Take a look at [`StructuredOutputType`][distilabel.steps.tasks.structured_outputs.outlines.StructuredOutputType] to see the expected format
    of the `structured_output` dict variable.

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
    model_path="./openhermes-2.5-mistral-7b.Q4_K_M.gguf"  # (1)
    n_gpu_layers=-1,
    n_ctx=1024,
    structured_output={"format": "json", "schema": User},
)
llm.load()
```

1. We have previously downloaded a GGUF model i.e. `llama.cpp` compatible, from the Hugging Face Hub using curl[^2], but any model can be used as replacement, as long as the `model_path` argument is updated.

And we are ready to pass our instruction as usual:

```python
import json

result = llm.generate(
    [[{"role": "user", "content": "Create a user profile for the following marathon"}]],
    max_new_tokens=50
)

data = json.loads(result[0][0])
data
# {'name': 'Kathy', 'last_name': 'Smith', 'id': 4539210}
User(**data)
# User(name='Kathy', last_name='Smith', id=4539210)
```

We get back a Python dictionary (formatted as a string) that we can parse using `json.loads`, or validate it directly using the `User`, which si a `pydantic.BaseModel` instance.

### Regex

The following example shows an example of text generation whose output adhere to a regular expression:

```python
pattern = r"<name>(.*?)</name>.*?<grade>(.*?)</grade>"  # the same pattern for re.compile

llm=LlamaCppLLM(
    model_path=model_path,
    n_gpu_layers=-1,
    n_ctx=1024,
    structured_output={"format": "regex", "schema": pattern},
)
llm.load()

result = llm.generate(
    [
        [
            {"role": "system", "content": "You are Simpsons' fans who loves assigning grades from A to E, where A is the best and E is the worst."},
            {"role": "user", "content": "What's up with Homer Simpson?"}
        ]
    ],
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

!!! Tip
    A full pipeline example can be seen in the following script:
    [`examples/structured_generation_with_outlines.py`](../../pipeline_samples/examples/llama_cpp_with_outlines.md)

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

## Instructor

For other LLM providers behinds APIs, there's no direct way of accessing the internal logit processor like `outlines` does, but thanks to [`instructor`](https://python.useinstructor.com/) we can generate structured output from LLM providers based on `pydantic.BaseModel` objects. We have integrated `instructor` to deal with the [`AsyncLLM`][distilabel.llms.AsyncLLM].

!!! Note
    For `instructor` integration to work you may need to install the corresponding dependencies:

    ```bash
    pip install distilabel[instructor]
    ```

!!! Note
    Take a look at [`InstructorStructuredOutputType`][distilabel.steps.tasks.typing.InstructorStructuredOutputType] to see the expected format
    of the `structured_output` dict variable.

The following is the same example you can see with `outlines`'s `JSON` section for comparison purposes.

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    last_name: str
    id: int
```

And then we provide that schema to the `structured_output` argument of the LLM:

!!! NOTE
    In this example we are using *Meta Llama 3.1 8B Instruct*, keep in mind not all the models support structured outputs.

```python
from distilabel.llms import MistralLLM

llm = InferenceEndpointsLLM(
    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    tokenizer_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    structured_output={"schema": User}
)
llm.load()
```

And we are ready to pass our instructions as usual:

```python
import json

result = llm.generate(
    [[{"role": "user", "content": "Create a user profile for the following marathon"}]],
    max_new_tokens=256
)

data = json.loads(result[0][0])
data
# {'name': 'John', 'last_name': 'Doe', 'id': 12345}
User(**data)
# User(name='John', last_name='Doe', id=12345)
```

We get back a Python dictionary (formatted as a string) that we can parse using `json.loads`, or validate it directly using the `User`, which is a `pydantic.BaseModel` instance.

!!! Tip
    A full pipeline example can be seen in the following script:
    [`examples/structured_generation_with_instructor.py`](../../pipeline_samples/examples/mistralai_with_instructor.md)

## OpenAI JSON

OpenAI offers a [JSON Mode](https://platform.openai.com/docs/guides/text-generation/json-mode) to deal with structured output via their API, let's see how to make use of them. The JSON mode instructs the model to always return a JSON object following the instruction required.

!!! WARNING
    Bear in mind, for this to work, you must instruct the model in some way to generate JSON, either in the `system message` or in the instruction, as can be seen in the [API reference](https://platform.openai.com/docs/guides/text-generation/json-mode).

Contrary to what we have via `outlines`, JSON mode will not guarantee the output matches any specific schema, only that it is valid and parses without errors. More information can be found in the OpenAI documentation.

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
