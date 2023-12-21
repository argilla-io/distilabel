import os

from distilabel.llm import OpenAILLM
from distilabel.tasks import OpenAITextGenerationTask

openaillm = OpenAILLM(
    model="gpt-3.5-turbo",
    task=OpenAITextGenerationTask(),
    max_new_tokens=256,
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0.3,
)
result = openaillm.generate([{"input": "What is OpenAI?"}])
# >>> print(result[0][0]["parsed_output"]["generations"])
# OpenAI is an artificial intelligence research laboratory and company. It was founded
# with the goal of ensuring that artificial general intelligence (AGI) benefits all of
# humanity. OpenAI conducts cutting-edge research in various fields of AI ...
