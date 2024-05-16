# TransformersLLM


Hugging Face `transformers` library LLM implementation using the text generation



pipeline.



### Attributes

- **model**: the model Hugging Face Hub repo id or a path to a directory containing the  model weights and configuration files.

- **revision**: if `model` refers to a Hugging Face Hub repository, then the revision  (e.g. a branch name or a commit id) to use. Defaults to `"main"`.

- **torch_dtype**: the torch dtype to use for the model e.g. "float16", "float32", etc.  Defaults to `"auto"`.

- **trust_remote_code**: whether to trust or not remote (code in the Hugging Face Hub  repository) code to load the model. Defaults to `False`.

- **model_kwargs**: additional dictionary of keyword arguments that will be passed to  the `from_pretrained` method of the model.

- **tokenizer**: the tokenizer Hugging Face Hub repo id or a path to a directory containing  the tokenizer config files. If not provided, the one associated to the `model`  will be used. Defaults to `None`.

- **use_fast**: whether to use a fast tokenizer or not. Defaults to `True`.

- **chat_template**: a chat template that will be used to build the prompts before  sending them to the model. If not provided, the chat template defined in the  tokenizer config will be used. If not provided and the tokenizer doesn't have  a chat template, then ChatML template will be used. Defaults to `None`.

- **device**: the name or index of the device where the model will be loaded. Defaults  to `None`.

- **device_map**: a dictionary mapping each layer of the model to a device, or a mode  like `"sequential"` or `"auto"`. Defaults to `None`.

- **token**: the Hugging Face Hub token that will be used to authenticate to the Hugging  Face Hub. If not provided, the `HF_TOKEN` environment or `huggingface_hub` package  local configuration will be used. Defaults to `None`.








