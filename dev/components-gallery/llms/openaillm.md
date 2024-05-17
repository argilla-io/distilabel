# OpenAILLM


OpenAI LLM implementation running the async API client.







### Attributes

- **model**: the model name to use for the LLM e.g. "gpt-3.5-turbo", "gpt-4", etc.  Supported models can be found [here](https://platform.openai.com/docs/guides/text-generation).

- **base_url**: the base URL to use for the OpenAI API requests. Defaults to `None`, which  means that the value set for the environment variable `OPENAI_BASE_URL` will  be used, or "https://api.openai.com/v1" if not set.

- **api_key**: the API key to authenticate the requests to the OpenAI API. Defaults to  `None` which means that the value set for the environment variable `OPENAI_API_KEY`  will be used, or `None` if not set.

- **max_retries**: the maximum number of times to retry the request to the API before  failing. Defaults to `6`.

- **timeout**: the maximum time in seconds to wait for a response from the API. Defaults  to `120`.

- **structured_output**: a dictionary containing the structured output configuration or if more  fine-grained control is needed, an instance of `OutlinesStructuredOutput`. Defaults to None.





### Runtime Parameters

- **base_url**: the base URL to use for the OpenAI API requests. Defaults to `None`.

- **api_key**: the API key to authenticate the requests to the OpenAI API. Defaults  to `None`.

- **max_retries**: the maximum number of times to retry the request to the API before  failing. Defaults to `6`.

- **timeout**: the maximum time in seconds to wait for a response from the API. Defaults  to `120`.





