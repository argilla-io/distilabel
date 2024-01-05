import os

from distilabel.llm import AnyscaleLLM
from distilabel.tasks import TextGenerationTask

anyscale_llm = AnyscaleLLM(
    model="HuggingFaceH4/zephyr-7b-beta",
    task=TextGenerationTask(),
    prompt_format="openai",
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
)
result = anyscale_llm.generate([{"input": "What is Anyscale?"}])
# >>> print(result[0][0]["parsed_output"]["generations"])
# 'Anyscale is a machine learning (ML) software company that provides tools and platforms
# for scalable distributed ML workflows. Their offerings enable data scientists and engineers
# to easily and efficiently deploy ML models at scale, both on-premise and on the cloud.
# Anyscale's core technology, Ray, is an open-source framework for distributed Python computation 
# that provides a unified interface for distributed computing, resource management, and task scheduling.
# With Anyscale's solutions, businesses can accelerate their ML development and deployment cycles and drive
# greater value from their ML investments.'