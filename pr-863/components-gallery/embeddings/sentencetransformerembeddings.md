---
hide:
  - navigation
---
# SentenceTransformerEmbeddings


`sentence-transformers` library implementation for embedding generation.







### Attributes

- **model**: the model Hugging Face Hub repo id or a path to a directory containing the  model weights and configuration files.

- **device**: the name of the device used to load the model e.g. "cuda", "mps", etc.  Defaults to `None`.

- **prompts**: a dictionary containing prompts to be used with the model. Defaults to  `None`.

- **default_prompt_name**: the default prompt (in `prompts`) that will be applied to the  inputs. If not provided, then no prompt will be used. Defaults to `None`.

- **trust_remote_code**: whether to allow fetching and executing remote code fetched  from the repository in the Hub. Defaults to `False`.

- **revision**: if `model` refers to a Hugging Face Hub repository, then the revision  (e.g. a branch name or a commit id) to use. Defaults to `"main"`.

- **token**: the Hugging Face Hub token that will be used to authenticate to the Hugging  Face Hub. If not provided, the `HF_TOKEN` environment or `huggingface_hub` package  local configuration will be used. Defaults to `None`.

- **truncate_dim**: the dimension to truncate the sentence embeddings. Defaults to `None`.

- **model_kwargs**: extra kwargs that will be passed to the Hugging Face `transformers`  model class. Defaults to `None`.

- **tokenizer_kwargs**: extra kwargs that will be passed to the Hugging Face `transformers`  tokenizer class. Defaults to `None`.

- **config_kwargs**: extra kwargs that will be passed to the Hugging Face `transformers`  configuration class. Defaults to `None`.

- **precision**: the dtype that will have the resulting embeddings. Defaults to `"float32"`.

- **normalize_embeddings**: whether to normalize the embeddings so they have a length  of 1. Defaults to `None`.







### Examples


#### Generating sentence embeddings
```python
from distilabel.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model="mixedbread-ai/mxbai-embed-large-v1")

embeddings.load()

results = embeddings.encode(inputs=["distilabel is awesome!", "and Argilla!"])
# [
#   [-0.05447685346007347, -0.01623094454407692, ...],
#   [4.4889533455716446e-05, 0.044016145169734955, ...],
# ]
```



