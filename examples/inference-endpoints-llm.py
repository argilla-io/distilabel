from ultralabel.llm.huggingface import InferenceEndpointsLLM
from ultralabel.prompts.base import PromptTemplate


class Llama2Prompt(PromptTemplate):
    def generate_prompt(self, question: str) -> str:
        return (
            "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]".format(
                system_prompt=self.system_prompt, prompt=question
            )
        )

    def _parse_output(self, output: str) -> str:
        return output.split("[ANSWER]")[1].split("[/ANSWER]")[0].strip()

    def input_args_names(self) -> list[str]:
        return ["question"]

    def output_args_names(self) -> list[str]:
        return ["answer"]


prompt_template = Llama2Prompt(
    system_prompt=(
        "You are a helpful and honest assistant and you have beed asked to"
        " answer faithfully, and with a direct response. You will be provided"
        " with a question and you need to answer using the format `[ANSWER]<ANSWER>[/ANSWER]`"
        " e.g. if the answer is `magic` then you should return `[ANSWER]magic[/ANSWER]`."
    ),
)
llm = InferenceEndpointsLLM(
    endpoint_url="<HUGGING_FACE_INFERENCE_ENDPOINT_URL>",
    prompt_template=prompt_template,
    token="<HUGGING_FACE_TOKEN>",
)
print(llm.generate([{"question": "What's the capital of Spain?"}]))
# Output: ["The capital of Spain is Madrid."]
