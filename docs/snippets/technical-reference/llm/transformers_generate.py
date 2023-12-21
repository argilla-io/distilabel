from distilabel.llm import TransformersLLM
from distilabel.tasks import TextGenerationTask
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the models from huggingface hub:
tokenizer = AutoTokenizer.from_pretrained("argilla/notus-7b-v1")
model = AutoModelForCausalLM.from_pretrained("argilla/notus-7b-v1", device_map="auto")

# Instantiate our LLM with them:
llm = TransformersLLM(
    model=model,
    tokenizer=tokenizer,
    task=TextGenerationTask(),
    max_new_tokens=128,
    temperature=0.3,
    prompt_format="notus",
)

result = llm.generate([{"input": "What's a large language model?"}])
# >>> print(result[0][0]["parsed_output"]["generations"])
# A large language model is a type of machine learning algorithm that is designed to analyze
# and understand large amounts of text data. It is called "large" because it requires a
# vast amount of data to train and improve its accuracy. These models are ...
