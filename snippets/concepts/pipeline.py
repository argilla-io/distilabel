from datasets import load_dataset
from distilabel.llm import LlamaCppLLM, OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import TextGenerationTask, UltraJudgeTask
from llama_cpp import Llama

dataset = load_dataset("argilla/distilabel-docs", split="train")
dataset = dataset.remove_columns(
    [
        column
        for column in dataset.column_names
        if column not in ["input", "generations"]
    ]
)

pipeline = Pipeline(
    generator=LlamaCppLLM(
        model=Llama(
            model_path="./llama-2-7b-chat.Q4_0.gguf",
            verbose=False,
            n_ctx=1024,
        ),
        task=TextGenerationTask(),
        max_new_tokens=512,
        prompt_format="llama2",
    ),
    labeller=OpenAILLM(
        model="gpt-3.5-turbo",
        task=UltraJudgeTask(),
        prompt_format="openai",
        max_new_tokens=1024,
        num_threads=1,
        temperature=0.0,
    ),
)


dataset = pipeline.generate(dataset, num_generations=2, batch_size=5)
