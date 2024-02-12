import os

from distilabel.llm import JSONOpenAILLM
from distilabel.tasks import TextGenerationTask

openaillm = JSONOpenAILLM(
    model="gpt-3.5-turbo-1106",  # json response is a limited feature
    task=TextGenerationTask(),
    prompt_format="openai",
    max_new_tokens=256,
    api_key=os.getenv("OPENAI_API_KEY", None),
    temperature=0.3,
)
result = openaillm.generate(
    [{"input": "write a json object with a key 'city' and value 'Madrid'"}]
)
# >>> print(result[0][0]["parsed_output"]["generations"])
# {"answer": "Madrid"}
