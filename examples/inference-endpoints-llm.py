from typing import Dict

from distilabel.llm.huggingface.inference_endpoints import InferenceEndpointsLLM
from distilabel.tasks.base import Task


class Llama2TextGenerationTask(Task):
    def generate_prompt(self, question: str) -> str:
        return (
            "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]".format(
                system_prompt=self.system_prompt, prompt=question
            )
        )

    def parse_output(self, output: str) -> Dict[str, str]:
        return {"answer": output.split("[ANSWER]")[1].split("[/ANSWER]")[0].strip()}

    def input_args_names(self) -> list[str]:
        return ["question"]

    def output_args_names(self) -> list[str]:
        return ["answer"]


task = Llama2TextGenerationTask(
    system_prompt=(
        "You are a helpful and honest assistant and you have beed asked to"
        " answer faithfully, and with a direct response. You will be provided"
        " with a question and you need to answer using the format `[ANSWER]<ANSWER>[/ANSWER]`"
        " e.g. if the answer is `magic` then you should return `[ANSWER]magic[/ANSWER]`."
    ),
)
llm = InferenceEndpointsLLM(
    endpoint_url="<HUGGING_FACE_INFERENCE_ENDPOINT_URL>",
    task=task,
    token="<HUGGING_FACE_TOKEN>",
)
print(llm.generate([{"question": "What's the capital of Spain?"}]))
# Output: [{"answer": "The capital of Spain is Madrid."}]
