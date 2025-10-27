---
hide:
  - navigation
---
# SGLangEmbeddings


`sglang` library implementation for embedding generation.







### Attributes

- **model**: the model Hugging Face Hub repo id or a path to a directory containing the  model weights and configuration files.

- **dtype**: the data type to use for the model. Defaults to `auto`.

- **trust_remote_code**: whether to trust the remote code when loading the model. Defaults  to `False`.

- **quantization**: the quantization mode to use for the model. Defaults to `None`.

- **revision**: the revision of the model to load. Defaults to `None`.

- **seed**: the seed to use for the random number generator. Defaults to `0`.

- **extra_kwargs**: additional dictionary of keyword arguments that will be passed to the  `Engine` class of `sglang` library. Defaults to `{}`.

- **_model**: the `SGLang` model instance. This attribute is meant to be used internally  and should not be accessed directly. It will be set in the `load` method.







### Examples


#### Generating sentence embeddings
```python
if __name__ == "__main__":

    from distilabel.models import SGLangEmbeddings
    embeddings = SGLangEmbeddings(model="intfloat/e5-mistral-7b-instruct")
    embeddings.load()
    results = embeddings.encode(inputs=["distilabel is awesome!", "and Argilla!"])
    print(results)
    # [
    #   [0.0203704833984375, -0.0060882568359375, ...],
    #   [0.02398681640625, 0.0177001953125 ...],
    # ]
```



