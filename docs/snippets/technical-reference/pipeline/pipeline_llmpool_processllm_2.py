from distilabel.llm import LLM, ProcessLLM
from distilabel.tasks import Task, TextGenerationTask


def load_notus(task: Task) -> LLM:
    import os
    from distilabel.llm import vLLM
    from vllm import LLM

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    return vLLM(
        vllm=LLM(model="argilla/notus-7b-v1"),
        task=task,
        max_new_tokens=512,
        temperature=0.7,
        prompt_format="notus",
    )


def load_zephyr(task: Task) -> LLM:
    import os
    from distilabel.llm import vLLM
    from vllm import LLM

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    return vLLM(
        vllm=LLM(model="HuggingFaceH4/zephyr-7b-beta"),
        task=task,
        max_new_tokens=512,
        temperature=0.7,
        prompt_format="notus",
    )


def load_starling(task: Task) -> LLM:
    import os
    from distilabel.llm import vLLM
    from vllm import LLM

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    return vLLM(
        vllm=LLM(model="berkeley-nest/Starling-LM-7B-alpha"),
        task=task,
        max_new_tokens=512,
        temperature=0.7,
        prompt_format="notus",
    )


def load_neural_chat(task: Task) -> LLM:
    import os
    from distilabel.llm import vLLM
    from vllm import LLM

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    return vLLM(
        vllm=LLM(model="Intel/neural-chat-7b-v3-3"),
        task=task,
        max_new_tokens=512,
        temperature=0.7,
        prompt_format="notus",
    )


notus = ProcessLLM(task=TextGenerationTask(), load_llm_fn=load_notus)
zephyr = ProcessLLM(task=TextGenerationTask(), load_llm_fn=load_zephyr)
starling = ProcessLLM(task=TextGenerationTask(), load_llm_fn=load_starling)
neural_chat = ProcessLLM(task=TextGenerationTask(), load_llm_fn=load_neural_chat)
