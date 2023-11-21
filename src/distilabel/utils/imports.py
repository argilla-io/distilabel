import importlib.util


def _check_module_is_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


_OPENAI_AVAILABLE = _check_module_is_available("openai")
_LLAMA_CPP_AVAILABLE = _check_module_is_available("llama_cpp")
_VLLM_AVAILABLE = _check_module_is_available("vllm")
_HUGGINGFACE_HUB_AVAILABLE = _check_module_is_available("huggingface_hub")
_TRANSFORMERS_AVAILABLE = _check_module_is_available("transformers")
