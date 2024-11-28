---
hide:
  - navigation
---
# LlamaCppLLM


llama.cpp LLM implementation running the Python bindings for the C++ code.







### Attributes

- **model_path**: contains the path to the GGUF quantized model, compatible with the  installed version of the `llama.cpp` Python bindings.

- **n_gpu_layers**: the number of layers to use for the GPU. Defaults to `-1`, meaning that  the available GPU device will be used.

- **chat_format**: the chat format to use for the model. Defaults to `None`, which means the  Llama format will be used.

- **n_ctx**: the context size to use for the model. Defaults to `512`.

- **n_batch**: the prompt processing maximum batch size to use for the model. Defaults to `512`.

- **seed**: random seed to use for the generation. Defaults to `4294967295`.

- **verbose**: whether to print verbose output. Defaults to `False`.

- **structured_output**: a dictionary containing the structured output configuration or if more  fine-grained control is needed, an instance of `OutlinesStructuredOutput`. Defaults to None.

- **extra_kwargs**: additional dictionary of keyword arguments that will be passed to the  `Llama` class of `llama_cpp` library. Defaults to `{}`.

- **_model**: the Llama model instance. This attribute is meant to be used internally and  should not be accessed directly. It will be set in the `load` method.





### Runtime Parameters

- **model_path**: the path to the GGUF quantized model.

- **n_gpu_layers**: the number of layers to use for the GPU. Defaults to `-1`.

- **chat_format**: the chat format to use for the model. Defaults to `None`.

- **verbose**: whether to print verbose output. Defaults to `False`.

- **extra_kwargs**: additional dictionary of keyword arguments that will be passed to the  `Llama` class of `llama_cpp` library. Defaults to `{}`.




### Examples


#### Generate text
```python
from pathlib import Path
from distilabel.models.llms import LlamaCppLLM

# You can follow along this example downloading the following model running the following
# command in the terminal, that will download the model to the `Downloads` folder:
# curl -L -o ~/Downloads/openhermes-2.5-mistral-7b.Q4_K_M.gguf https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf

model_path = "Downloads/openhermes-2.5-mistral-7b.Q4_K_M.gguf"

llm = LlamaCppLLM(
    model_path=str(Path.home() / model_path),
    n_gpu_layers=-1,  # To use the GPU if available
    n_ctx=1024,       # Set the context size
)

llm.load()

# Call the model
output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
```

#### Generate structured data
```python
from pathlib import Path
from distilabel.models.llms import LlamaCppLLM

model_path = "Downloads/openhermes-2.5-mistral-7b.Q4_K_M.gguf"

class User(BaseModel):
    name: str
    last_name: str
    id: int

llm = LlamaCppLLM(
    model_path=str(Path.home() / model_path),  # type: ignore
    n_gpu_layers=-1,
    n_ctx=1024,
    structured_output={"format": "json", "schema": Character},
)

llm.load()

# Call the model
output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Create a user profile for the following marathon"}]])
```




### References

- [llama.cpp](https://github.com/ggerganov/llama.cpp)

- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

