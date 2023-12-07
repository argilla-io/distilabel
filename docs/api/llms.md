# LLMs

The following LLMs are implemented:

- [OpenAILLM][distilabel.llm.openai.OpenAILLM]: 

- [LlammaCppLLM][distilabel.llm.llama_cpp.LlamaCppLLM]: 

    Useful when you need to run your LLMs locally, keep located your weights and run them.

- [vLLM][distilabel.llm.vllm.vLLM]: 

- Huggingface LLMs

    - [TransformersLLM][distilabel.llm.huggingface.transformers.TransformersLLM]: 

    - [InferenceEndpointsLLM][distilabel.llm.huggingface.inference_endpoints.InferenceEndpointsLLM]: 

        Useful if you want to use the huggingface infraestructure to deploy your LLM easily.

---

```python
from distilabel.llm import InferenceEndpointsLLM


```

