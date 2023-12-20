from datasets import load_dataset

instruction_dataset = (
    load_dataset("HuggingFaceH4/instruction-dataset", split="test[:3]")
    .remove_columns(["completion", "meta"])
    .rename_column("prompt", "input")
)

pipe_dataset = pipe.generate(
    instruction_dataset,
    num_generations=2,
    batch_size=1,
    enable_checkpoints=True,
    display_progress_bar=True,
)
