import time
from typing import Any, Dict

from datasets import load_dataset
from distilabel.llm.huggingface.inference_endpoints import InferenceEndpointsLLM
from distilabel.llm.openai_ import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks.base import Task
from distilabel.tasks.preference.judgelm import JudgeLMTask
from distilabel.tasks.prompt import Prompt
from transformers import AutoTokenizer


class TextGenerationTask(Task):
    system_prompt: str = (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible,"
        " while being safe. Your answers should not include any harmful, unethical, racist, sexist,"
        " toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased"
        " and positive in nature.\nIf a question does not make any sense, or is not factually coherent,"
        " explain why instead of answering something not correct. If you don't know the answer to a"
        " question, please don't share false information."
    )

    def generate_prompt(self, input: str) -> Prompt:
        return Prompt(system_prompt=self.system_prompt, formatted_prompt=input)

    def parse_output(self, output: str) -> dict[str, str]:
        return {"generations": output}

    @property
    def input_args_names(self) -> list[str]:
        return ["input"]

    @property
    def output_args_names(self) -> list[str]:
        return ["generations"]


tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
task = TextGenerationTask()


def remove_sample_if_token_limit_reached(example: Dict[str, Any]) -> bool:
    prompt = task.generate_prompt(input=example["instruction"])
    prompt_chat = tokenizer.apply_chat_template(
        prompt.format_as("openai"), tokenize=True
    )
    return len(prompt_chat) < 1024


dataset = load_dataset("openbmb/UltraFeedback", split="train[:500]")
print(f"Original Dataset length: {len(dataset)}")  # type: ignore
dataset = dataset.filter(lambda example: remove_sample_if_token_limit_reached(example))

print(f"Filtered Dataset length: {len(dataset)}")  # type: ignore
dataset = (
    dataset.shuffle()
    .remove_columns(
        [
            "source",
            "models",
            "completions",
            "correct_answers",
            "incorrect_answers",
        ]
    )
    .rename_column("instruction", "input")
)

pipeline = Pipeline(
    generator=InferenceEndpointsLLM(
        endpoint_name="<ZEPHYR_BETA_INFERENCE_endpoint_name>",
        task=TextGenerationTask(),
        max_new_tokens=1024,  # input_length=1024, max_length=2048
        num_threads=4,
        temperature=1.0,
        prompt_formatting_fn=lambda prompt: f"<|system|>\n{prompt.system_prompt}</s>\n<|user|>\n{prompt.formatted_prompt}</s>\n<|assistant|>\n",
    ),
    labeller=OpenAILLM(
        model="gpt-3.5-turbo",
        task=JudgeLMTask(),
        max_new_tokens=512,
        num_threads=4,
        openai_api_key="<OPENAI_API_KEY>",
        temperature=0.0,
    ),
)

start = time.time()
disti_dataset = pipeline.generate(
    dataset,  # type: ignore
    num_generations=2,
    batch_size=8,
    enable_checkpoints=True,
    display_progress_bar=True,
)
end = time.time()
print("Elapsed", end - start)

dataset.push_to_hub(  # type: ignore
    "<HUGGING_FACE_HUB_USER>/zephyr-7b-beta-judgelm",
    split="train",  # type: ignore
    private=False,
)
