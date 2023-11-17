import os

from distilabel.pipeline import Pipeline
from distilabel.llm.openai_ import OpenAILLM
from distilabel.tasks.preference.judgelm import JudgeLMTask
from distilabel.tasks.preference.ultrafeedback import UltraFeedbackTask
from distilabel.tasks.preference.ultrajudge import UltraJudgeTask

import argilla as rg

from datasets import load_dataset

from distilabel.tasks.preference.ultrajudge import UltraJudgeTask

os.environ["OPENAI_API_KEY"] = "sk-"
rg.init(
    api_url="...",
    api_key="argilla.apikey"
)

dataset = load_dataset("gabrielmbmb/ultrafeedback-prompts-judgelm-gpt35", split="train")

dataset = (
    dataset#.shuffle()
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
    .select(range(1))

)

labeller = OpenAILLM(
        model="gpt-3.5-turbo",
        task=UltraFeedbackTask.for_text_quality(),
        max_new_tokens=1024,
        num_threads=16,
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

print("Preparing dataset for UltraFeedback")
rg_dataset = labelled.to_argilla()
rg.FeedbackDataset.from_argilla(name="disti-ufeedback", workspace="argilla").delete()
rg_dataset.push_to_argilla(name="disti-ufeedback", workspace="argilla")

# judgelm
labeller = OpenAILLM(
        model="gpt-3.5-turbo",
        task=JudgeLMTask(),
        max_new_tokens=1024,
        num_threads=16,
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

print("Preparing dataset for JudgeLM")
rg_dataset = labelled.to_argilla()
rg.FeedbackDataset.from_argilla(name="disti-judgelm", workspace="argilla").delete()
rg_dataset.push_to_argilla(name="disti-judgelm", workspace="argilla")

# ultrajudge
labeller = OpenAILLM(
        model="gpt-3.5-turbo",
        task=UltraJudgeTask(),
        max_new_tokens=1024,
        num_threads=16,
        temperature=0.0,
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
print("Preparing dataset for UltraJudge")
rg_dataset = labelled.to_argilla()
rg.FeedbackDataset.from_argilla(name="disti-ultrajudge-std", workspace="argilla").delete()
rg_dataset.push_to_argilla(name="disti-ultrajudge-std", workspace="argilla")