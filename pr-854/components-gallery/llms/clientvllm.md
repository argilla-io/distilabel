---
hide:
  - navigation
---
# ClientvLLM


A client for the `vLLM` server served with `python -m vllm.entrypoints.api_server`.







### Attributes

- **base_url**: the base URL of the `vLLM` server. Defaults to `"http://localhost:8000"`.

- **max_retries**: the maximum number of times to retry the request to the API before  failing. Defaults to `6`.

- **timeout**: the maximum time in seconds to wait for a response from the API. Defaults  to `120`.

- **httpx_client_kwargs**: extra kwargs that will be passed to the `httpx.AsyncClient`  created to comunicate with the `vLLM` server. Defaults to `None`.

- **tokenizer**: the Hugging Face Hub repo id or path of the tokenizer that will be used  to apply the chat template and tokenize the inputs before sending it to the  server. Defaults to `None`.

- **tokenizer_revision**: the revision of the tokenizer to load. Defaults to `None`.

- **_aclient**: the `httpx.AsyncClient` used to comunicate with the `vLLM` server. Defaults  to `None`.





### Runtime Parameters

- **base_url**: the base url of the `vLLM` server. Defaults to `"http://localhost:8000"`.

- **max_retries**: the maximum number of times to retry the request to the API before  failing. Defaults to `6`.

- **timeout**: the maximum time in seconds to wait for a response from the API. Defaults  to `120`.

- **httpx_client_kwargs**: extra kwargs that will be passed to the `httpx.AsyncClient`  created to comunicate with the `vLLM` server. Defaults to `None`.




### Examples


#### Generate text
```python
from distilabel.llms import ClientvLLM

llm = ClientvLLM(tokenizer="meta-llama/Meta-Llama-3.1-8B-Instruct")

llm.load()

results = llm.generate(
    inputs=[[{"role": "user", "content": "Hello, how are you?"}]],
    temperature=0.7,
    top_p=1.0,
    max_new_tokens=256,
)
#
```



