import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
import string

from pathlib import Path
from typing import Tuple
from sklearn.metrics import confusion_matrix

def load_data(dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path_to_dataset = Path.cwd() / 'datasets' / dataset_name
    if path_to_dataset.exists():
        df = pd.read_csv(path_to_dataset, header=None)
        return (df.iloc[:, 0:1024], df.iloc[:, 1024])

def generate_csv(df: pd.DataFrame, filename: str) -> None:
    path_to_output = Path.cwd().parent / 'output' / 'csv'
    path_to_output.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_to_output / filename, header=False)

def generate_cm(pred: pd.DataFrame, true: pd.DataFrame, filename: str) -> None:
    path_to_output = Path.cwd().parent / 'output' / 'plots'
    path_to_output.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(pred, true)
    if len(cm) == 26:
        labels = list(string.ascii_uppercase)
    else:
        labels = ['\u03C0', '\u03B1', '\u03B2', '\u03C3', '\u03B3', '\u03B4', '\u03BB', '\u03C9', '\u03BC', '\u03BE']
    plot.figure(figsize=(18,10))
    sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels)
    plot.savefig(path_to_output / filename)