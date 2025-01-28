---
hide:
  - navigation
---
# VertexAILLM


VertexAI LLM implementation running the async API clients for Gemini.



- Gemini API: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini

    To use the `VertexAILLM` is necessary to have configured the Google Cloud authentication
    using one of these methods:

    - Setting `GOOGLE_CLOUD_CREDENTIALS` environment variable
    - Using `gcloud auth application-default login` command
    - Using `vertexai.init` function from the `google-cloud-aiplatform` library





### Attributes

- **model**: the model name to use for the LLM e.g. "gemini-1.0-pro". [Supported models](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models).

- **_aclient**: the `GenerativeModel` to use for the Vertex AI Gemini API. It is meant  to be used internally. Set in the `load` method.







### Examples


#### Generate text
```python
from distilabel.models.llms import VertexAILLM

llm = VertexAILLM(model="gemini-1.5-pro")

llm.load()

# Call the model
output = llm.generate(inputs=[[{"role": "user", "content": "Hello world!"}]])
```



