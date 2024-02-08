import os

from distilabel.llm import MistralAILLM
from distilabel.tasks import TextGenerationTask

mistralai_llm = MistralAILLM(
    model="mistral-tiny",
    task=TextGenerationTask(),
    api_key=os.environ.get("MISTRALAI_API_KEY"),
)
result = mistralai_llm.generate([{"input": "What is Anyscale?"}])
# >>> print(result[0][0]["parsed_output"]["generations"])
# I'd be happy to help answer your question, but it's important to note
# that the "best" French cheese can be subjective as it depends on personal
# taste preferences. Some popular and highly regarded French cheeses include
# Roquefort for its strong, tangy flavor and distinct blue veins; Camembert
# for its earthy, mushroomy taste and soft, runny texture; and Brie for its
# creamy, buttery, and slightly sweet taste. I'd recommend trying different
# types to find the one you enjoy the most.