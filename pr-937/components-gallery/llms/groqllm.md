---
hide:
  - navigation
---
# GroqLLM


Groq API implementation using the async client for concurrent text generation.







### Attributes

- **model**: the name of the model from the Groq API to use for the generation.

- **base_url**: the base URL to use for the Groq API requests. Defaults to  `"https://api.groq.com"`.

- **api_key**: the API key to authenticate the requests to the Groq API. Defaults to  the value of the `GROQ_API_KEY` environment variable.

- **max_retries**: the maximum number of times to retry the request to the API before  failing. Defaults to `2`.

- **timeout**: the maximum time in seconds to wait for a response from the API. Defaults  to `120`.

- **structured_output**: a dictionary containing the structured output configuration configuration  using `instructor`. You can take a look at the dictionary structure in  `InstructorStructuredOutputType` from `distilabel.steps.tasks.structured_outputs.instructor`.

- **_api_key_env_var**: the name of the environment variable to use for the API key.

- **_aclient**: the `AsyncGroq` client from the `groq` package.





### Runtime Parameters

- **base_url**: the base URL to use for the Groq API requests. Defaults to  `"https://api.groq.com"`.

- **api_key**: the API key to authenticate the requests to the Groq API. Defaults to  the value of the `GROQ_API_KEY` environment variable.

- **max_retries**: the maximum number of times to retry the request to the API before  failing. Defaults to `2`.

- **timeout**: the maximum time in seconds to wait for a response from the API. Defaults  to `120`.




### Examples


#### Generate text
```python
from distilabel.llms import GroqLLM

llm = GroqLLM(model="llama3-70b-8192")

llm.load()

# Call the model
output = llm.generate(inputs=[[{"role": "user", "content": "Hello world!"}]])

Generate structured data:
```



