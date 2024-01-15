from distilabel.llm import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import TextGenerationTask

from datasets import Dataset

token = ""
model = "aws-notus-7b-v1-3184"
namespace = "argilla"

llm = InferenceEndpointsLLM(
    endpoint_name_or_model_id=model,  # type: ignore
    endpoint_namespace=namespace,  # type: ignore
    token=token,
    task=TextGenerationTask(),
)

pipeline = Pipeline(generator=llm)

inputs = ["Generate whatever"] * 50
dataset = Dataset.from_dict({"input": inputs})

generated_instructions = pipeline.generate(
    dataset=dataset, num_generations=10, display_progress_bar=True
)

print(generated_instructions[0])