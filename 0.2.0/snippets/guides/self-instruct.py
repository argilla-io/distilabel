from distilabel.llm import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import SelfInstructTask

self_instruct = SelfInstructTask(
    application_description="An AI application to generate tables in markdown format.",
    num_instructions=5,
)

generator = OpenAILLM(task=self_instruct)

pipeline = Pipeline(generator=generator)

dataset = pipeline.generate(dataset=dataset, num_generations=4, batch_size=2)
