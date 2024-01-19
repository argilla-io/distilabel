from datasets import load_dataset

dataset = (
    load_dataset("HuggingFaceH4/instruction-dataset", split="test[:50]")
    .remove_columns(["completion", "meta"])
    .rename_column("prompt", "input")
)

dataset = pipeline.generate(
    dataset=dataset,
    num_generations=2,
    batch_size=5,
    display_progress_bar=True,
)

dataset.to_argilla().push_to_argilla(name="preference-dataset", workspace="admin")
