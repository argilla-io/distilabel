# InferenceEndpointsLLM


InferenceEndpoints LLM implementation running the async API client.



This LLM will internally use `huggingface_hub.AsyncInferenceClient` or `openai.AsyncOpenAI`
    depending on the `use_openai_client` attribute.



### Attributes

- **model_id**: the model ID to use for the LLM as available in the Hugging Face Hub, which  will be used to resolve the base URL for the serverless Inference Endpoints API requests.  Defaults to `None`.

- **endpoint_name**: the name of the Inference Endpoint to use for the LLM. Defaults to `None`.

- **endpoint_namespace**: the namespace of the Inference Endpoint to use for the LLM. Defaults to `None`.

- **base_url**: the base URL to use for the Inference Endpoints API requests.

- **api_key**: the API key to authenticate the requests to the Inference Endpoints API.

- **tokenizer_id**: the tokenizer ID to use for the LLM as available in the Hugging Face Hub.  Defaults to `None`, but defining one is recommended to properly format the prompt.

- **model_display_name**: the model display name to use for the LLM. Defaults to `None`.

- **use_openai_client**: whether to use the OpenAI client instead of the Hugging Face client.







### Examples


#### Free serverless Inference API
```python
from distilabel.llms.huggingface import InferenceEndpointsLLM

llm = InferenceEndpointsLLM(
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
)

llm.load()

# Synchrounous request
output = llm.generate(inputs=[[{"role": "user", "content": "Hello world!"}]])

# Asynchronous request
output = await llm.agenerate(input=[{"role": "user", "content": "Hello world!"}])
```

#### Dedicated Inference Endpoints
```python
from distilabel.llms.huggingface import InferenceEndpointsLLM

llm = InferenceEndpointsLLM(
    endpoint_name="<ENDPOINT_NAME>",
    api_key="<HF_API_KEY>",
    endpoint_namespace="<USER|ORG>",
)

llm.load()

# Synchrounous request
output = llm.generate(inputs=[[{"role": "user", "content": "Hello world!"}]])

# Asynchronous request
output = await llm.agenerate(input=[{"role": "user", "content": "Hello world!"}])
```

#### Dedicated Inference Endpoints or TGI
```python
from distilabel.llms.huggingface import InferenceEndpointsLLM

llm = InferenceEndpointsLLM(
    api_key="<HF_API_KEY>",
    base_url="<BASE_URL>",
)

llm.load()

# Synchrounous request
output = llm.generate(inputs=[[{"role": "user", "content": "Hello world!"}]])

# Asynchronous request
output = await llm.agenerate(input=[{"role": "user", "content": "Hello world!"}])
```



