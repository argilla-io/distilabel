import os
from datasets import load_dataset
from distilabel.pipeline import Pipeline
from distilabel.tasks import TextGenerationTask, UltraFeedbackTask
from distilabel.llm import vLLM, ProcessLLM, LLMPool
from vllm import LLM


def load_notus(task):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="TheBloke/notus-7B-v1-AWQ", quantization="awq")
    return vLLM(vllm=llm, task=task, max_new_tokens=512)


def load_zephyr(task):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    llm = LLM(model="TheBloke/zephyr-7b-beta-AWQ", quantization="awq")
    return vLLM(vllm=llm, task=task, max_new_tokens=512)

def load_openai(task):
    from distilabel.llm import OpenAILLM

    return OpenAILLM(
        model="gpt-3.5-turbo",
        task=task,
        openai_api_key="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",
        max_new_tokens=512,
    )


if __name__ == "__main__":
    dataset = (
        load_dataset("HuggingFaceH4/instruction-dataset", split="test")
        .remove_columns(["completion", "meta"])
        .rename_column("prompt", "input")
    )


    pipeline = Pipeline(
        generator=LLMPool(
            [
                ProcessLLM(task=TextGenerationTask(), load_llm_fn=load_notus),
                ProcessLLM(task=TextGenerationTask(), load_llm_fn=load_zephyr),
            ]
        ),
        labeller=ProcessLLM(
            task=UltraFeedbackTask.for_instruction_following(),
            load_llm_fn=load_openai
        )
    )

    dataset = pipeline.generate(
        dataset=dataset,
        num_generations=3,
        batch_size=5,
    )

    dataset.push_to_hub("gabrielmbmb/test-dataset")
