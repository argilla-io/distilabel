from distilabel.llm import TransformersLLM
from distilabel.tasks import TextGenerationTask
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the models from huggingface hub:
tokenizer = AutoTokenizer.from_pretrained("argilla/notus-7b-v1")
model = AutoModelForCausalLM.from_pretrained("argilla/notus-7b-v1")

# Instantiate our LLM with them:
llm = TransformersLLM(
    model=model,
    tokenizer=tokenizer,
    task=TextGenerationTask(),
    max_new_tokens=128,
    temperature=0.3,
    prompt_format="notus",
)
