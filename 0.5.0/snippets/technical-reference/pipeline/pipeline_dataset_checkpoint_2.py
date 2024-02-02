new_ds = pipe.generate(
    dataset=dataset,
    num_generations=1,
    checkpoint_strategy=dataset_checkpoint,
)