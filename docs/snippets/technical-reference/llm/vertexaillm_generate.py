from distilabel.llm import VertexAILLM
from distilabel.tasks import TextGenerationTask

llm = VertexAILLM(
    task=TextGenerationTask(), model="gemini-pro", max_new_tokens=512, temperature=0.3
)

results = llm.generate(
    inputs=[
        {"input": "Write a short summary about the Gemini astrological sign"},
    ],
)
# >>> print(results[0][0]["parsed_output"]["generations"])
# Gemini, the third astrological sign in the zodiac, is associated with the element of
# air and is ruled by the planet Mercury. People born under the Gemini sign are often
# characterized as being intelligent, curious, and communicative. They are known for their
# quick wit, adaptability, and versatility. Geminis are often drawn to learning and enjoy
# exploring new ideas and concepts. They are also known for their social nature and ability
# to connect with others easily. However, Geminis can also be seen as indecisive, restless,
# and superficial at times. They may struggle with commitment and may have difficulty focusing
# on one thing for too long. Overall, Geminis are known for their intelligence, curiosity,
# and social nature.
