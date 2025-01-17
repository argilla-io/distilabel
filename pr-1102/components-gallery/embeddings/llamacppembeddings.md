---
hide:
  - navigation
---
# LlamaCppEmbeddings


`LlamaCpp` library implementation for embedding generation.







### Attributes

- **model_name**: contains the name of the GGUF quantized model, compatible with the  installed version of the `llama.cpp` Python bindings.

- **model_path**: contains the path to the GGUF quantized model, compatible with the  installed version of the `llama.cpp` Python bindings.

- **repo_id**: the Hugging Face Hub repository id.

- **verbose**: whether to print verbose output. Defaults to `False`.

- **n_gpu_layers**: number of layers to run on the GPU. Defaults to `-1` (use the GPU if available).

- **disable_cuda_device_placement**: whether to disable CUDA device placement. Defaults to `True`.

- **normalize_embeddings**: whether to normalize the embeddings. Defaults to `False`.

- **seed**: RNG seed, -1 for random

- **n_ctx**: Text context, 0 = from model

- **n_batch**: Prompt processing maximum batch size

- **extra_kwargs**: additional dictionary of keyword arguments that will be passed to the  `Llama` class of `llama_cpp` library. Defaults to `{}`.





### Runtime Parameters

- **n_gpu_layers**: the number of layers to use for the GPU. Defaults to `-1`.

- **verbose**: whether to print verbose output. Defaults to `False`.

- **normalize_embeddings**: whether to normalize the embeddings. Defaults to `False`.

- **extra_kwargs**: additional dictionary of keyword arguments that will be passed to the  `Llama` class of `llama_cpp` library. Defaults to `{}`.




### Examples


#### Generate sentence embeddings using a local model
```python
from pathlib import Path
from distilabel.models.embeddings import LlamaCppEmbeddings

# You can follow along this example downloading the following model running the following
# command in the terminal, that will download the model to the `Downloads` folder:
# curl -L -o ~/Downloads/all-MiniLM-L6-v2-Q2_K.gguf https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q2_K.gguf

model_path = "Downloads/"
model = "all-MiniLM-L6-v2-Q2_K.gguf"
embeddings = LlamaCppEmbeddings(
    model=model,
    model_path=str(Path.home() / model_path),
)

embeddings.load()

results = embeddings.encode(inputs=["distilabel is awesome!", "and Argilla!"])
print(results)
embeddings.unload()
```

#### Generate sentence embeddings using a HuggingFace Hub model
```python
from distilabel.models.embeddings import LlamaCppEmbeddings
# You need to set environment variable to download private model to the local machine

repo_id = "second-state/All-MiniLM-L6-v2-Embedding-GGUF"
model = "all-MiniLM-L6-v2-Q2_K.gguf"
embeddings = LlamaCppEmbeddings(model=model,repo_id=repo_id)

embeddings.load()

results = embeddings.encode(inputs=["distilabel is awesome!", "and Argilla!"])
print(results)
embeddings.unload()
# [
#   [-0.05447685346007347, -0.01623094454407692, ...],
#   [4.4889533455716446e-05, 0.044016145169734955, ...],
# ]
```

#### Generate sentence embeddings with cpu
```python
from pathlib import Path
from distilabel.models.embeddings import LlamaCppEmbeddings

# You can follow along this example downloading the following model running the following
# command in the terminal, that will download the model to the `Downloads` folder:
# curl -L -o ~/Downloads/all-MiniLM-L6-v2-Q2_K.gguf https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q2_K.gguf

model_path = "Downloads/"
model = "all-MiniLM-L6-v2-Q2_K.gguf"
embeddings = LlamaCppEmbeddings(
    model=model,
    model_path=str(Path.home() / model_path),
    n_gpu_layers=0,
    disable_cuda_device_placement=True,
)

embeddings.load()

results = embeddings.encode(inputs=["distilabel is awesome!", "and Argilla!"])
print(results)
embeddings.unload()
# [
#   [-0.05447685346007347, -0.01623094454407692, ...],
#   [4.4889533455716446e-05, 0.044016145169734955, ...],
# ]
```




### References

- [Offline inference embeddings](https://llama-cpp-python.readthedocs.io/en/stable/#embeddings)

