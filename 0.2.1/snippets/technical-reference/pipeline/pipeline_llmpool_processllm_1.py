from distilabel.llm import LLM, ProcessLLM
from distilabel.tasks import Task, TextGenerationTask


def load_notus(task: Task) -> LLM:  # (1)
    import os
    from distilabel.llm import vLLM
    from vllm import LLM

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # (2)

    return vLLM(
        vllm=LLM(model="argilla/notus-7b-v1"),
        task=task,
        max_new_tokens=512,
        temperature=0.7,
        prompt_format="notus",
    )


llm = ProcessLLM(task=TextGenerationTask(), load_llm_fn=load_notus)
