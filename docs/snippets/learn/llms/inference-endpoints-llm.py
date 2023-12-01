import os

from distilabel.llm import InferenceEndpointsLLM
from distilabel.tasks import Llama2TextGenerationTask, Prompt


class Llama2QuestionAnsweringTask(Llama2TextGenerationTask):
    def generate_prompt(self, question: str) -> str:
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=question,
        ).format_as("llama2")  # type: ignore

    def parse_output(self, output: str) -> dict[str, str]:
        return {"answer": output.strip()}

    def input_args_names(self) -> list[str]:
        return ["question"]

    def output_args_names(self) -> list[str]:
        return ["answer"]


llm = InferenceEndpointsLLM(
    endpoint_name=os.getenv("HF_INFERENCE_ENDPOINT_NAME"),  # type: ignore
    endpoint_namespace=os.getenv("HF_NAMESPACE"),  # type: ignore
    token=os.getenv("HF_TOKEN") or None,
    task=Llama2QuestionAnsweringTask(),
)
print(llm.generate([{"question": "What's the capital of Spain?"}]))
# Output: [
#   [{
#       'model_name': 'HuggingFaceH4/zephyr-7b-beta',
#       'prompt_used': "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.<</SYS>>\n\nWhat's the capital of Spain? [/INST]",
#       'raw_output': "\n<<ASSistant>>\nThe capital of Spain is Madrid. Other major cities in Spain include Barcelona, Valencia, Seville, and Bilbao. Madrid is the largest city in Spain and serves as the country's political, economic, and cultural center. It is home to many famous landmarks, such as the Royal Palace, the Prado Museum, and the Plaza Mayor.",
#       'parsed_output': {
#           'answer': "<<ASSistant>>\nThe capital of Spain is Madrid. Other major cities in Spain include Barcelona, Valencia, Seville, and Bilbao. Madrid is the largest city in Spain and serves as the country's political, economic, and cultural center. It is home to many famous landmarks, such as the Royal Palace, the Prado Museum, and the Plaza Mayor.
#       },
#   }]
# ]
