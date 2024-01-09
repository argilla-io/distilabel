from datasets import Dataset

dataset = Dataset.from_dict(
    {"input": ["Create an easy dinner recipe with few ingredients"]}
)
dataset_generated = pipe_generation.generate(dataset, num_generations=2)
