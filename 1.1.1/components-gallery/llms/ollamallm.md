# OllamaLLM


Ollama LLM implementation running the Async API client.







### Attributes

- **model**: the model name to use for the LLM e.g. "notus".

- **host**: the Ollama server host.

- **timeout**: the timeout for the LLM. Defaults to `120`.

- **_aclient**: the `AsyncClient` to use for the Ollama API. It is meant to be used internally.  Set in the `load` method.





### Runtime Parameters

- **host**: the Ollama server host.

- **timeout**: the client timeout for the Ollama API. Defaults to `120`.





