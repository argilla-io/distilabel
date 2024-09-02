---
hide:
  - navigation
---
# vLLMEmbeddings


`vllm` library implementation for embedding generation.







### Attributes

- **model**: the model Hugging Face Hub repo id or a path to a directory containing the  model weights and configuration files.

- **dtype**: the data type to use for the model. Defaults to `auto`.

- **trust_remote_code**: whether to trust the remote code when loading the model. Defaults  to `False`.

- **quantization**: the quantization mode to use for the model. Defaults to `None`.

- **revision**: the revision of the model to load. Defaults to `None`.

- **enforce_eager**: whether to enforce eager execution. Defaults to `True`.

- **seed**: the seed to use for the random number generator. Defaults to `0`.

- **extra_kwargs**: additional dictionary of keyword arguments that will be passed to the  `LLM` class of `vllm` library. Defaults to `{}`.

- **_model**: the `vLLM` model instance. This attribute is meant to be used internally  and should not be accessed directly. It will be set in the `load` method.







### Examples


#### Generating sentence embeddings
```python
from distilabel.embeddings import vLLMEmbeddings

embeddings = vLLMEmbeddings(model="intfloat/e5-mistral-7b-instruct")

embeddings.load()

results = embeddings.encode(inputs=["distilabel is awesome!", "and Argilla!"])
# [
#   [-0.05447685346007347, -0.01623094454407692, ...],
#   [4.4889533455716446e-05, 0.044016145169734955, ...],
# ]
```




### References

- [Offline inference embeddings](https://docs.vllm.ai/en/latest/getting_started/examples/offline_inference_embedding.html)

