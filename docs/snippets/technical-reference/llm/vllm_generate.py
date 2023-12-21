from distilabel.tasks import TextGenerationTask
from distilabel.llm import vLLM
from vllm import LLM

llm = vLLM(
    vllm=LLM(model="argilla/notus-7b-v1"),
    task=TextGenerationTask(),
    max_new_tokens=512,
    temperature=0.3,
    prompt_format="notus",
)
result_vllm = llm.generate([{"input": "What's a large language model?"}])
# >>> print(result[0][0]["parsed_output"]["generations"])
# A large language model is a type of artificial intelligence (AI) system that is designed
# to understand and interpret human language. It is called "large" because it uses a vast
# amount of data, typically billions of words or more, to learn and make predictions about
# language. Large language models are ...
