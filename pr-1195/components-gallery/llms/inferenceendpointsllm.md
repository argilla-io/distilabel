---
hide:
  - navigation
---
# InferenceEndpointsLLM


InferenceEndpoints LLM implementation running the async API client.



This LLM will internally use `huggingface_hub.AsyncInferenceClient`.





### Attributes

- **model_id**: the model ID to use for the LLM as available in the Hugging Face Hub, which  will be used to resolve the base URL for the serverless Inference Endpoints API requests.  Defaults to `None`.

- **endpoint_name**: the name of the Inference Endpoint to use for the LLM. Defaults to `None`.

- **endpoint_namespace**: the namespace of the Inference Endpoint to use for the LLM. Defaults to `None`.

- **base_url**: the base URL to use for the Inference Endpoints API requests.

- **api_key**: the API key to authenticate the requests to the Inference Endpoints API.

- **provider**: the name of the provider to use for inference. Defaults to `None`.

- **tokenizer_id**: the tokenizer ID to use for the LLM as available in the Hugging Face Hub.  Defaults to `None`, but defining one is recommended to properly format the prompt.

- **model_display_name**: the model display name to use for the LLM. Defaults to `None`.

- **use_magpie_template**: a flag used to enable/disable applying the Magpie pre-query  template. Defaults to `False`.

- **magpie_pre_query_template**: the pre-query template to be applied to the prompt or  sent to the LLM to generate an instruction or a follow up user message. Valid  values are "llama3", "qwen2" or another pre-query template provided. Defaults  to `None`.

- **structured_output**: a dictionary containing the structured output configuration or  if more fine-grained control is needed, an instance of `OutlinesStructuredOutput`.  Defaults to None.







### Examples


#### Free serverless Inference API, set the input_batch_size of the Task that uses this to avoid Model is overloaded
```python
from distilabel.models.llms.huggingface import InferenceEndpointsLLM

llm = InferenceEndpointsLLM(
    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
)

llm.load()

output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
```

#### Dedicated Inference Endpoints
```python
from distilabel.models.llms.huggingface import InferenceEndpointsLLM

llm = InferenceEndpointsLLM(
    endpoint_name="<ENDPOINT_NAME>",
    api_key="<HF_API_KEY>",
    endpoint_namespace="<USER|ORG>",
)

llm.load()

output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
```

#### Dedicated Inference Endpoints or TGI
```python
from distilabel.models.llms.huggingface import InferenceEndpointsLLM

llm = InferenceEndpointsLLM(
    api_key="<HF_API_KEY>",
    base_url="<BASE_URL>",
)

llm.load()

output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
```

#### Generate structured data
```python
from pydantic import BaseModel
from distilabel.models.llms import InferenceEndpointsLLM

class User(BaseModel):
    name: str
    last_name: str
    id: int

llm = InferenceEndpointsLLM(
    model_id="meta-llama/Meta-Llama-3-70B-Instruct",
    tokenizer_id="meta-llama/Meta-Llama-3-70B-Instruct",
    api_key="api.key",
    structured_output={"format": "json", "schema": User.model_json_schema()}
)

llm.load()

output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Create a user profile for the Tour De France"}]])
```



