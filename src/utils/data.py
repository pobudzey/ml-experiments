import pandas as pd

from pathlib import Path
from typing import Tuple

def load_data(dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path_to_dataset = Path.cwd() / 'datasets' / dataset_name
    if path_to_dataset.exists():
        df = pd.read_csv(path_to_dataset, header=None)
        return (df.iloc[:, 0:1024], df.iloc[:, 1024])

def generate_csv(df: pd.DataFrame, name: str) -> None:
    path_to_output = Path.cwd() / 'output'
    path_to_output.mkdir(exist_ok=True)
    df.to_csv(path_to_output / name, header=False)
