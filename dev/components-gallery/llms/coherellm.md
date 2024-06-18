---
hide:
  - navigation
---
# CohereLLM


Cohere API implementation using the async client for concurrent text generation.







### Attributes

- **model**: the name of the model from the Cohere API to use for the generation.

- **base_url**: the base URL to use for the Cohere API requests. Defaults to  `"https://api.cohere.ai/v1"`.

- **api_key**: the API key to authenticate the requests to the Cohere API. Defaults to  the value of the `COHERE_API_KEY` environment variable.

- **timeout**: the maximum time in seconds to wait for a response from the API. Defaults  to `120`.

- **client_name**: the name of the client to use for the API requests. Defaults to  `"distilabel"`.

- **structured_output**: a dictionary containing the structured output configuration configuration  using `instructor`. You can take a look at the dictionary structure in  `InstructorStructuredOutputType` from `distilabel.steps.tasks.structured_outputs.instructor`.

- **_ChatMessage**: the `ChatMessage` class from the `cohere` package.

- **_aclient**: the `AsyncClient` client from the `cohere` package.





### Runtime Parameters

- **base_url**: the base URL to use for the Cohere API requests. Defaults to  `"https://api.cohere.ai/v1"`.

- **api_key**: the API key to authenticate the requests to the Cohere API. Defaults  to the value of the `COHERE_API_KEY` environment variable.

- **timeout**: the maximum time in seconds to wait for a response from the API. Defaults  to `120`.

- **client_name**: the name of the client to use for the API requests. Defaults to  `"distilabel"`.




### Examples


#### Generate text
```python
from distilabel.llms import CohereLLM

llm = CohereLLM(model="CohereForAI/c4ai-command-r-plus")

llm.load()

# Call the model
output = llm.generate(inputs=[[{"role": "user", "content": "Hello world!"}]])

Generate structured data:
```



