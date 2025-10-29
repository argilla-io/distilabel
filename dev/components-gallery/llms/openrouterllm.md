---
hide:
  - navigation
---
# OpenRouterLLM


OpenRouter LLM implementation running the async API client of OpenAI.







### Attributes

- **model**: the model name to use for the LLM, e.g., `google/gemma-7b-it`. See the  supported models under the "Models" section  [here](https://openrouter.ai/models?fmt=cards&supported_parameters=tools).

- **base_url**: the base URL to use for the OpenRouter API requests. Defaults to `None`, which  means that the value set for the environment variable `OPENROUTER_BASE_URL` will be used, or  "https://api.endpoints.anyscale.com/v1" if not set.

- **api_key**: the API key to authenticate the requests to the OpenRouter API. Defaults to `None` which  means that the value set for the environment variable `OPENROUTER_API_KEY` will be used, or  `None` if not set.

- **_api_key_env_var**: the name of the environment variable to use for the API key.  It is meant to be used internally.







### Examples


#### Generate text
```python
from distilabel.models.llms import OpenRouterLLM

llm = OpenRouterLLM(model="google/gemma-7b-it", api_key="api.key")

llm.load()

output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
```



