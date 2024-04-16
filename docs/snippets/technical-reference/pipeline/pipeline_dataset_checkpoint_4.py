from distilabel.dataset import DatasetCheckpoint

dataset_checkpoint = DatasetCheckpoint(
    strategy="hf-hub",
    save_frequency=1,
    extra_kwargs={
        "repo_id": "username/dataset-name"
    }
)

new_ds = pipe.generate(
    dataset=dataset,
    num_generations=1,
    checkpoint_strategy=dataset_checkpoint,
)