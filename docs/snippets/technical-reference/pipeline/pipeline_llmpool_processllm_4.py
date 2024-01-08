from distilabel.llm import LLM, ProcessLLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import UltraFeedbackTask


def load_gpt_4(task: UltraFeedbackTask) -> LLM:
    from distilabel.llm import OpenAILLM

    return OpenAILLM(
        model="gpt-4-1106-preview",
        task=task,
        max_new_tokens=512,
        num_threads=4,
    )


pipeline = Pipeline(
    generator=pool,
    labeller=ProcessLLM(task=UltraFeedbackTask(), load_llm_fn=load_gpt_4),  # (1)
)
