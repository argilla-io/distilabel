---
hide:
  - navigation
---
# OpenAIImageLM


OpenAI image generation implementation running the async API client.







### Attributes

- **model**: the model name to use for the LLM e.g. "dall-e-3", etc.  Supported models can be found [here](https://platform.openai.com/docs/guides/images).

- **base_url**: the base URL to use for the OpenAI API requests. Defaults to `None`, which  means that the value set for the environment variable `OPENAI_BASE_URL` will  be used, or "https://api.openai.com/v1" if not set.

- **api_key**: the API key to authenticate the requests to the OpenAI API. Defaults to  `None` which means that the value set for the environment variable `OPENAI_API_KEY`  will be used, or `None` if not set.

- **max_retries**: the maximum number of times to retry the request to the API before  failing. Defaults to `6`.

- **timeout**: the maximum time in seconds to wait for a response from the API. Defaults  to `120`.







### Examples


#### Generate images from text prompts
```python
from distilabel.models.vlms import OpenAIImageLM

ilm = OpenAIImageLM(model="dall-e-3", api_key="api.key")

ilm.load()

output = ilm.generate_outputs(
    inputs=["a white siamese cat"],
    size="1024x1024",
    quality="standard",
    style="natural",
)
# [{"images": ["iVBORw0KGgoAAAANSUhEUgA..."]}]
```



