---
hide:
  - navigation
---
# MistralLLM


Mistral LLM implementation running the async API client.







### Attributes

- **model**: the model name to use for the LLM e.g. "mistral-tiny", "mistral-large", etc.

- **endpoint**: the endpoint to use for the Mistral API. Defaults to "https://api.mistral.ai".

- **api_key**: the API key to authenticate the requests to the Mistral API. Defaults to `None` which  means that the value set for the environment variable `OPENAI_API_KEY` will be used, or  `None` if not set.

- **max_retries**: the maximum number of retries to attempt when a request fails. Defaults to `5`.

- **timeout**: the maximum time in seconds to wait for a response. Defaults to `120`.

- **max_concurrent_requests**: the maximum number of concurrent requests to send. Defaults  to `64`.

- **structured_output**: a dictionary containing the structured output configuration configuration  using `instructor`. You can take a look at the dictionary structure in  `InstructorStructuredOutputType` from `distilabel.steps.tasks.structured_outputs.instructor`.

- **_api_key_env_var**: the name of the environment variable to use for the API key. It is meant to  be used internally.

- **_aclient**: the `Mistral` to use for the Mistral API. It is meant to be used internally.  Set in the `load` method.





### Runtime Parameters

- **api_key**: the API key to authenticate the requests to the Mistral API.

- **max_retries**: the maximum number of retries to attempt when a request fails.  Defaults to `5`.

- **timeout**: the maximum time in seconds to wait for a response. Defaults to `120`.

- **max_concurrent_requests**: the maximum number of concurrent requests to send.  Defaults to `64`.




### Examples


#### Generate text
```python
from distilabel.models.llms import MistralLLM

llm = MistralLLM(model="open-mixtral-8x22b")

llm.load()

# Call the model
output = llm.generate(inputs=[[{"role": "user", "content": "Hello world!"}]])

Generate structured data:
```



