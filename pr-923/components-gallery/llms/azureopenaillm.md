---
hide:
  - navigation
---
# AzureOpenAILLM


Azure OpenAI LLM implementation running the async API client.







### Attributes

- **model**: the model name to use for the LLM i.e. the name of the Azure deployment.

- **base_url**: the base URL to use for the Azure OpenAI API can be set with `AZURE_OPENAI_ENDPOINT`.  Defaults to `None` which means that the value set for the environment variable  `AZURE_OPENAI_ENDPOINT` will be used, or `None` if not set.

- **api_key**: the API key to authenticate the requests to the Azure OpenAI API. Defaults to `None`  which means that the value set for the environment variable `AZURE_OPENAI_API_KEY` will be  used, or `None` if not set.

- **api_version**: the API version to use for the Azure OpenAI API. Defaults to `None` which means  that the value set for the environment variable `OPENAI_API_VERSION` will be used, or  `None` if not set.







### Examples


#### Generate text
```python
from distilabel.llms import AzureOpenAILLM

llm = AzureOpenAILLM(model="gpt-4-turbo", api_key="api.key")

llm.load()

output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
```

#### Generate text from a custom endpoint following the OpenAI API
```python
from distilabel.llms import AzureOpenAILLM

llm = AzureOpenAILLM(
    model="prometheus-eval/prometheus-7b-v2.0",
    base_url=r"http://localhost:8080/v1"
)

llm.load()

output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
```

#### Generate structured data
```python
from pydantic import BaseModel
from distilabel.llms import AzureOpenAILLM

class User(BaseModel):
    name: str
    last_name: str
    id: int

llm = AzureOpenAILLM(
    model="gpt-4-turbo",
    api_key="api.key",
    structured_output={"schema": User}
)

llm.load()

output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Create a user profile for the following marathon"}]])
```



