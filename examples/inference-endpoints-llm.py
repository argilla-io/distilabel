from typing import Dict

from distilabel.llm.huggingface.inference_endpoints import InferenceEndpointsLLM
from distilabel.tasks.text_generation.llama import Llama2TextGenerationTask


class Llama2QuestionAnsweringTask(Llama2TextGenerationTask):
    def parse_output(self, output: str) -> Dict[str, str]:
        return {"answer": output.split("[ANSWER]")[1].split("[/ANSWER]")[0].strip()}

    def input_args_names(self) -> list[str]:
        return ["question"]

    def output_args_names(self) -> list[str]:
        return ["answer"]


llm = InferenceEndpointsLLM(
    endpoint_name="<HUGGING_FACE_INFERENCE_ENDPOINT_NAME>",
    token="<HUGGING_FACE_TOKEN>",
    task=Llama2QuestionAnsweringTask(),
)
print(llm.generate([{"question": "What's the capital of Spain?"}]))
# Output: [{"answer": "The capital of Spain is Madrid."}]
