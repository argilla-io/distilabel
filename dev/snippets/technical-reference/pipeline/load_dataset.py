from distilabel.dataset import load_dataset

dataset: "CustomDataset" = load_dataset("argilla/distilabel-sample-evol-instruct", split="train")
print(dataset.task)
# EvolInstructTask(system_prompt="", task_description=...)