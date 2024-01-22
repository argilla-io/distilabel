from distilabel.llm import OllamaLLM
from distilabel.tasks import TextGenerationTask

llm = OllamaLLM(
    model="notus",  # should be deployed via `ollama notus:7b-v1-q5_K_M`
    task=TextGenerationTask(),
    prompt_format="openai",
)
result = llm.generate([{"input": "What's a large language model?"}])
# >>> print(result[0][0]["parsed_output"]["generations"])
# A large language model is a type of artificial intelligence (AI) system that has been trained
# on a vast amount of text data to generate human-like language. These models are capable of
# understanding and generating complex sentences, and can be used for tasks such as language
# translation, text summarization, and natural language generation. They are typically very ...
