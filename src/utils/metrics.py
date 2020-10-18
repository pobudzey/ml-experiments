import pandas as pd
import csv

from sklearn.metrics import confusion_matrix
from typing import List
from pathlib import Path

def generate_metrics(pred: List[int], true: List[int], size: int) -> List[List[int]]:
    metrics = []
    class_count = [0 for x in range(size)]

    # Precision, recall and f1 for each class
    for letter in range(size):
        tp = fp = fn = 0
        precision = recall = f1 = 0.0
        for i, j in zip(pred, true):
            if i == letter and j == letter:
                tp += 1
                class_count[letter] += 1
            if i == letter and j != letter:
                fp += 1
            if i != letter and j == letter:
                fn += 1
                class_count[letter] += 1
        if tp + fp != 0:
            precision = round(tp / (tp + fp), 3)
        if tp + fn != 0:
            recall = round(tp / (tp + fn), 3)
        if precision + recall != 0:
            f1 = round((2 * precision * recall) / (precision + recall), 3)
        metrics.append([precision, recall, f1])
    
    # Accuracy
    total_correct = 0
    for i, j in zip(pred, true):
        if i == j:
            total_correct += 1
    accuracy = round(total_correct / len(pred), 3)

    # Macro-average f1
    macro_average_f1 = 0
    for i in metrics:
        macro_average_f1 += i[2]
    macro_average_f1 = round(macro_average_f1 / len(metrics), 3)

    # Weighted-average f1
    weighted_average_f1 = 0
    for weight, i in zip(class_count, metrics):
        weighted_average_f1 += weight * i[2]
    weighted_average_f1 = round(weighted_average_f1 / len(pred), 3)

    metrics.append([accuracy])
    metrics.append([macro_average_f1])
    metrics.append([weighted_average_f1])
    return metrics

def compute(pred: pd.DataFrame, true: pd.Series, filename: str) -> None:
    alphabet_size = len(confusion_matrix(pred, true))
    metrics = generate_metrics(pd.Series(pred.iloc[:, 0]).tolist(), true.tolist(), alphabet_size)
    csv_path = Path.cwd().parent / 'output' / 'csv' / filename
    if csv_path.exists():
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(metrics)







