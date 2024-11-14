---
hide:
  - navigation
---
# InferenceEndpointsImageLM


OpenAI image generation implementation running the async API client.







### Attributes

- **model_id**: the model ID to use for the LLM as available in the Hugging Face Hub, which  will be used to resolve the base URL for the serverless Inference Endpoints API requests.  Defaults to `None`.

- **endpoint_name**: the name of the Inference Endpoint to use for the LLM. Defaults to `None`.

- **endpoint_namespace**: the namespace of the Inference Endpoint to use for the LLM. Defaults to `None`.

- **base_url**: the base URL to use for the Inference Endpoints API requests.

- **api_key**: the API key to authenticate the requests to the Inference Endpoints API.







### Examples


#### Generate images from text prompts
```python
from distilabel.models.ilms import InferenceEndpointsImageLM

ilm = InferenceEndpointsImageLM(model_id="black-forest-labs/FLUX.1-schnell", api_key="api.key")
ilm.load()

output = ilm.generate_outputs(
    inputs=["a white siamese cat"],
)
# {"images": ["iVBORw0KGgoAAAANSUhEUgA..."]}
```



