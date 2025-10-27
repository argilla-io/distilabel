---
hide:
  - navigation
---
# OllamaLLM


Ollama LLM implementation running the Async API client.







### Attributes

- **model**: the model name to use for the LLM e.g. "notus".

- **host**: the Ollama server host.

- **timeout**: the timeout for the LLM. Defaults to `120`.

- **follow_redirects**: whether to follow redirects. Defaults to `True`.

- **structured_output**: a dictionary containing the structured output configuration or if more  fine-grained control is needed, an instance of `OutlinesStructuredOutput`. Defaults to None.

- **tokenizer_id**: the tokenizer Hugging Face Hub repo id or a path to a directory containing  the tokenizer config files. If not provided, the one associated to the `model`  will be used. Defaults to `None`.

- **use_magpie_template**: a flag used to enable/disable applying the Magpie pre-query  template. Defaults to `False`.

- **magpie_pre_query_template**: the pre-query template to be applied to the prompt or  sent to the LLM to generate an instruction or a follow up user message. Valid  values are "llama3", "qwen2" or another pre-query template provided. Defaults  to `None`.

- **_aclient**: the `AsyncClient` to use for the Ollama API. It is meant to be used internally.  Set in the `load` method.





### Runtime Parameters

- **host**: the Ollama server host.

- **timeout**: the client timeout for the Ollama API. Defaults to `120`.




### Examples


#### Generate text
```python
from distilabel.models.llms import OllamaLLM

llm = OllamaLLM(model="llama3")

llm.load()

# Call the model
output = llm.generate(inputs=[[{"role": "user", "content": "Hello world!"}]])
```



