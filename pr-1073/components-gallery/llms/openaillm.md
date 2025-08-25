---
hide:
  - navigation
---
# OpenAILLM


OpenAI LLM implementation running the async API client.







### Attributes

- **model**: the model name to use for the LLM e.g. "gpt-3.5-turbo", "gpt-4", etc.  Supported models can be found [here](https://platform.openai.com/docs/guides/text-generation).

- **base_url**: the base URL to use for the OpenAI API requests. Defaults to `None`, which  means that the value set for the environment variable `OPENAI_BASE_URL` will  be used, or "https://api.openai.com/v1" if not set.

- **api_key**: the API key to authenticate the requests to the OpenAI API. Defaults to  `None` which means that the value set for the environment variable `OPENAI_API_KEY`  will be used, or `None` if not set.

- **max_retries**: the maximum number of times to retry the request to the API before  failing. Defaults to `6`.

- **timeout**: the maximum time in seconds to wait for a response from the API. Defaults  to `120`.

- **structured_output**: a dictionary containing the structured output configuration configuration  using `instructor`. You can take a look at the dictionary structure in  `InstructorStructuredOutputType` from `distilabel.steps.tasks.structured_outputs.instructor`.





### Runtime Parameters

- **base_url**: the base URL to use for the OpenAI API requests. Defaults to `None`.

- **api_key**: the API key to authenticate the requests to the OpenAI API. Defaults  to `None`.

- **max_retries**: the maximum number of times to retry the request to the API before  failing. Defaults to `6`.

- **timeout**: the maximum time in seconds to wait for a response from the API. Defaults  to `120`.




### Examples


#### Generate text
```python
from distilabel.models.llms import OpenAILLM

llm = OpenAILLM(model="gpt-4-turbo", api_key="api.key")

llm.load()

output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
```

#### Generate text from a custom endpoint following the OpenAI API
```python
from distilabel.models.llms import OpenAILLM

llm = OpenAILLM(
    model="prometheus-eval/prometheus-7b-v2.0",
    base_url=r"http://localhost:8080/v1"
)

llm.load()

output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
```

#### Generate structured data
```python
from pydantic import BaseModel
from distilabel.models.llms import OpenAILLM

class User(BaseModel):
    name: str
    last_name: str
    id: int

llm = OpenAILLM(
    model="gpt-4-turbo",
    api_key="api.key",
    structured_output={"schema": User}
)

llm.load()

output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Create a user profile for the following marathon"}]])
```

#### Generate with Batch API (offline batch generation)
```python
from distilabel.models.llms import OpenAILLM

load = llm = OpenAILLM(
    model="gpt-3.5-turbo",
    use_offline_batch_generation=True,
    offline_batch_generation_block_until_done=5,  # poll for results every 5 seconds
)

llm.load()

output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
# [['Hello! How can I assist you today?']]
```



