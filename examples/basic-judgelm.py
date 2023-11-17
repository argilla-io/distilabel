import os

from distilabel.pipeline import Pipeline
from distilabel.llm.openai_ import OpenAILLM
from distilabel.tasks.preference.judgelm import JudgeLMTask
from datasets import load_dataset

os.environ["OPENAI_API_KEY"] = "sk..."
dataset = load_dataset("gabrielmbmb/ultrafeedback-prompts-judgelm-gpt35", split="train")

dataset = (
    dataset.shuffle()
    .remove_columns([
        'generation_model', 
        'generation_prompt', 
        'raw_generation_responses', 
        'labelling_model', 
        'labelling_prompt', 
        'raw_labelling_response', 
        'ratings', 
        'rationale'
    ])
    .select(range(10))
    #.rename_column("instruction", "input")
)

labeller = OpenAILLM(
        model="gpt-3.5-turbo",
        task=JudgeLMTask(),
        max_new_tokens=512,
        num_threads=4,
        temperature=1.0,
)

pipeline = Pipeline(
  labeller=labeller
)

labelled = pipeline.generate(
    dataset,  # type: ignore
    num_generations=2,
    batch_size=8,
    enable_checkpoints=True,
    display_progress_bar=True,
)

rg_dataset = labelled.to_argilla()
import argilla as rg

rg.init(
    api_url="<rg api url>",
    api_key="<rg api key>"
)

rg_dataset.push_to_argilla(name="example", workspace="argilla")