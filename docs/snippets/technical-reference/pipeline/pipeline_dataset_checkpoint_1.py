from pathlib import Path
from distilabel.dataset import DatasetCheckpoint

# Assuming we want to save the dataset every 10% of the records generated.

freq = len(dataset) // 10
dataset_checkpoint = DatasetCheckpoint(path=Path.cwd() / "checkpoint_folder", save_frequency=freq)
