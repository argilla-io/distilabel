import os

from distilabel.llm import OpenAILLM
from distilabel.tasks import OpenAITextGenerationTask

openaillm = OpenAILLM(
    model="gpt-3.5-turbo",
    task=OpenAITextGenerationTask(),
    max_new_tokens=256,
    num_threads=2,
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0.3,
)
result_openai = openaillm.generate([{"input": "What is OpenAI?"}])
# >>> result_openai
# [<Future at 0x2970ea560 state=running>]
# >>> result_openai[0].result()[0][0]["parsed_output"]["generations"]
# 'OpenAI is an artificial intelligence research organization that aims to ensure that artificial general intelligence (AGI) benefits all of humanity. AGI refers to highly autonomous systems that outperform humans at most economically valuable work. OpenAI conducts research, develops AI technologies, and promotes the responsible and safe use of AI. They also work on projects to make AI more accessible and beneficial to society. OpenAI is committed to transparency, cooperation, and avoiding uses of AI that could harm humanity or concentrate power in the wrong hands.'
