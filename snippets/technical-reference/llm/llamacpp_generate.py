from distilabel.llm import LlamaCppLLM
from distilabel.tasks import TextGenerationTask
from llama_cpp import Llama

# Instantiate our LLM with them:
llm = LlamaCppLLM(
    model=Llama(model_path="./notus-7b-v1.q4_k_m.gguf", n_gpu_layers=-1),
    task=TextGenerationTask(),
    max_new_tokens=128,
    temperature=0.3,
    prompt_format="notus",
)

result = llm.generate([{"input": "What is the capital of Spain?"}])
# >>> print(result[0][0]["parsed_output"]["generations"])
# The capital of Spain is Madrid. It is located in the center of the country and
# is known for its vibrant culture, beautiful architecture, and delicious food.
# Madrid is home to many famous landmarks such as the Prado Museum, Retiro Park,
# and the Royal Palace of Madrid. I hope this information helps!
