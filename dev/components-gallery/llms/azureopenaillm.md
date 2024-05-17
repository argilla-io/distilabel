# AzureOpenAILLM


Azure OpenAI LLM implementation running the async API client.







### Attributes

- **model**: the model name to use for the LLM i.e. the name of the Azure deployment.

- **base_url**: the base URL to use for the Azure OpenAI API can be set with `AZURE_OPENAI_ENDPOINT`.  Defaults to `None` which means that the value set for the environment variable  `AZURE_OPENAI_ENDPOINT` will be used, or `None` if not set.

- **api_key**: the API key to authenticate the requests to the Azure OpenAI API. Defaults to `None`  which means that the value set for the environment variable `AZURE_OPENAI_API_KEY` will be  used, or `None` if not set.

- **api_version**: the API version to use for the Azure OpenAI API. Defaults to `None` which means  that the value set for the environment variable `OPENAI_API_VERSION` will be used, or  `None` if not set.








