from datasets import load_dataset
from distilabel.tasks import JudgeLMTask
from distilabel.utils import prepare_dataset

dataset = load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split="train")
dataset.task = JudgeLMTask()
dataset_binarized_random = prepare_dataset(dataset, strategy="random", keep_ties=True)
# >>> len(dataset)
# 12859
# >>> len(dataset_binarized_random)
# 12817
dataset_binarized_random = prepare_dataset(dataset, strategy="random", keep_ties=False)
# >>> len(dataset_binarized_random)
# 8850