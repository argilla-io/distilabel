---
hide:
  - navigation
---
# TogetherLLM


TogetherLLM LLM implementation running the async API client of OpenAI.







### Attributes

- **model**: the model name to use for the LLM e.g. "mistralai/Mixtral-8x7B-Instruct-v0.1".  Supported models can be found [here](https://api.together.xyz/models).

- **base_url**: the base URL to use for the Together API can be set with `TOGETHER_BASE_URL`.  Defaults to `None` which means that the value set for the environment variable  `TOGETHER_BASE_URL` will be used, or "https://api.together.xyz/v1" if not set.

- **api_key**: the API key to authenticate the requests to the Together API. Defaults to `None`  which means that the value set for the environment variable `TOGETHER_API_KEY` will be  used, or `None` if not set.

- **_api_key_env_var**: the name of the environment variable to use for the API key. It  is meant to be used internally.







### Examples


#### Generate text
```python
from distilabel.llms import AnyscaleLLM

llm = TogetherLLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key="api.key")

llm.load()

output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
```



