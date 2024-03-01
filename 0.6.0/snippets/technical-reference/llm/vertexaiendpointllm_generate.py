from distilabel.llm import VertexAIEndpointLLM
from distilabel.tasks import TextGenerationTask

llm = VertexAIEndpointLLM(
    task=TextGenerationTask(),
    endpoint_id="3466410517680095232",
    project="experiments-404412",
    location="us-central1",
    generation_kwargs={
        "temperature": 1.0,
        "max_tokens": 128,
        "top_p": 1.0,
        "top_k": 10,
    },
)

results = llm.generate(
    inputs=[
        {"input": "Write a short summary about the Gemini astrological sign"},
    ],
)
# >>> print(results[0][0]["parsed_output"]["generations"])
# Geminis are known for their curiosity, adaptability, and love of knowledge. They are
# also known for their tendency to be indecisive, impulsive and prone to arguing. They
# are ruled by the planet Mercury, which is associated with communication, quick thinking,
# and change.
