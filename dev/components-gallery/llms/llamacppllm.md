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






### References

- [llama.cpp](https://github.com/ggerganov/llama.cpp)

- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

