import argparse
import random
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
from datasets import load_dataset, load_from_disk, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from distilabel.llm import OpenAILLM
from distilabel.tasks.base import Task
from distilabel.pipeline import pipeline
from distilabel.tasks import UltraFeedbackTask, JudgeLMTask, UltraJudgeTask


def format_columns_lmsys(example: Dict[str, Union[int, List[Dict[str, str]]]]) -> Dict[str, Union[str, List[str], int]]:
    """
    Format columns for the LMSys dataset.

    Args:
        example (dict): Input example.

    Returns:
        dict: Formatted example.
    """
    if example["turn"] == 1:
        A = example["conversation_a"][1]["content"]
        B = example["conversation_b"][1]["content"]
        prompt = example["conversation_a"][0]["content"]
        generation = [A, B]
    else:
        A = example["conversation_a"][3]["content"]
        B = example["conversation_b"][3]["content"]
        prompt = example["conversation_a"][2]["content"]
        generation = [A, B]
    # dataset has label "tie", that will have label=2
    gold = int(example["winner"] == "model_b") if example["winner"] != "tie" else 2
    example["input"] = prompt
    example["generations"] = generation
    example["gold"] = gold
    return example


def format_columns_hhh(examples: Dataset) -> Dataset:
    """
    Format columns for the HHH dataset.

    Args:
        examples (Dataset): Input dataset.

    Returns:
        Dataset: Formatted dataset.
    """
    generations = []
    gold_labels = []
    for target in examples["targets"]:
        generations.append(target["choices"])
        # Conventional multiple-choice accuracy can be achieved by assigning the correct target
        # a score of 1, and all incorrect targets a score of 0.
        gold_labels.append(target["labels"].index(1))
    examples["generations"] = generations
    examples["gold"] = gold_labels
    return examples


def get_dataset(dataset_name: str) -> Tuple[Dataset, List[int]]:
    """
    Load and format the specified dataset.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        Tuple[Dataset, List[int]]: Formatted dataset and gold labels.
    """
    if dataset_name == "MT_Bench":
        dataset = load_dataset("lmsys/mt_bench_human_judgments", split="human")
        dataset = dataset.select(random.sample(range(len(dataset)), 100))       
        dataset = dataset.map(format_columns_lmsys)
        gold = dataset["gold"]
        cols_to_remove = ["question_id", "model_a", "model_b", "winner", "judge", "turn", "conversation_a", "conversation_b"]
        dataset = dataset.remove_columns(cols_to_remove)
        dataset = dataset.select(range(100))
    elif dataset_name == "HHH_Alignment":
        dataset = load_dataset("HuggingFaceH4/hhh_alignment", "other", split="test")
        dataset = dataset.map(format_columns_hhh, batched=True)
        gold = dataset["gold"]
        dataset = dataset.remove_columns(["targets", "gold"])
    else:
        raise KeyError("Choose either MT_Bench or HHH_Alignment dataset")
    return dataset, gold


def generate_preds(dataset: Dataset, task_type: Task) -> Dataset:
    """
    Generate predictions using the specified task and dataset.

    Args:
        dataset (Dataset): Input dataset.
        task_type (Task): Type of task.

    Returns:
        Dataset: Dataset with generated predictions.
    """
    generator = OpenAILLM(
        task=task, 
        prompt_format="openai",
        max_new_tokens=1024,
        openai_api_key="sk-YOUR-KEY",
        temperature = 1.0,
        top_p = 1.0,
    )
    pipe = pipeline("preference", "text-quality", labeller=generator)
    dataset = pipe.generate(dataset, num_generations=1, display_progress_bar=True)
    return dataset


def calc_metrics(dataset: Union[Dataset, str], gold: List[int], binary: bool=True) -> Dict[str, float]:
    """
    Calculate and print evaluation metrics.

    Args:
        dataset (Union[Dataset, str]): Input dataset or path to the saved dataset.
        gold (List[int]): Gold labels.
        binary (bool): Whether the task is binary or not.

    Returns:
        Dict[str, float]: Evaluation metrics.
    """
    # NOTE: HHH dataset always has the first response as target, so recall, precision and f1 scores are 0
    if isinstance(dataset, str):
        dataset = load_from_disk(dataset)

    preds = dataset["rating"]
    if not binary:
        preds_indices = [int(pred[0] > pred[1]) if pred[0] != pred[1] else 2
                            for pred in preds]
    else:
        preds_indices = np.array(preds).argmax(axis=1).tolist()

    gold = gold[:len(preds_indices)]
    metrics = {
        "accuracy": accuracy_score(gold, preds_indices),
        "recall": recall_score(gold, preds_indices, average="weighted"),
        "precision": precision_score(gold, preds_indices, average="weighted"),
        "f1 score": f1_score(gold, preds_indices, average="weighted")
    }

    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.03f}")
    return metrics

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate predictions and calculate metrics.")
    parser.add_argument("--dataset_name", type=str, default="MT_Bench", help="Name of the dataset (MT_Bench or HHH_Alignment).")
    parser.add_argument("--task_name", type=str, default="UltraJudge", help="Name of the task (UltraFeedback, JudgeLM, or UltraJudge).")
    parser.add_argument("--save_path", type=str, default="mt_bench_judgelm", help="Path to save the generated dataset.")
    args = parser.parse_args()

    tasks = {"UltraFeedback": UltraFeedbackTask.for_text_quality(), "JudgeLM": JudgeLMTask(), "UltraJudge": UltraJudgeTask()}
    task = tasks[args.task_name]
    binary = (args.dataset_name == "HHH_Alignment")
    dataset, gold = get_dataset(args.dataset_name)
    preds_dataset = generate_preds(dataset, task)
    preds_dataset["gold"] = gold
    metrics = calc_metrics(preds_dataset, gold, binary=binary)
    if args.save_path:
        preds_dataset.save_to_disk(args.save_path)