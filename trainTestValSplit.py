import pandas as pd

from definitions import ROOT_DIR
from os.path import isfile, join
from os import listdir
from sklearn.model_selection import train_test_split


def getCSVs(_dataset_path: str) -> dict[str, pd.DataFrame]:
    paths_CSV: set[str] = set(
        [join(_dataset_path, f) for f in listdir(_dataset_path) if isfile(join(_dataset_path, f))])

    return {path_CSV: pd.read_csv(path_CSV, index_col=False) for path_CSV in paths_CSV}


def getTrainTestValSplit(_dataset_path: str, _train_frac: float = 0.7, _test_frac: float = 0.2,
                         _val_frac: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    CSVs: dict[str, pd.DataFrame] = getCSVs(_dataset_path=_dataset_path)
    assert _train_frac + _test_frac + _val_frac <= 1.0

    train_filenames, test_filenames = train_test_split(list(CSVs.keys()), test_size=_test_frac, train_size=_train_frac,
                                                       random_state=7342)
    val_filenames = set(CSVs.keys()) - set(train_filenames) - set(test_filenames)

    train_CSVs: list[pd.DataFrame] = [CSVs[df] for df in train_filenames]
    test_CSVs: list[pd.DataFrame] = [CSVs[df] for df in test_filenames]
    val_CSVs: list[pd.DataFrame] = [CSVs[df] for df in val_filenames]

    train_concat: pd.DataFrame = pd.concat(train_CSVs, ignore_index=True)
    test_concat: pd.DataFrame = pd.concat(test_CSVs, ignore_index=True)
    val_concat: pd.DataFrame = pd.concat(val_CSVs, ignore_index=True)

    train_concat: pd.DataFrame = train_concat.drop(columns=[_ for _ in train_concat.columns if 'Unnamed' in _])

    del CSVs, train_CSVs, test_CSVs, val_CSVs  # Release memory

    return train_concat, test_concat, val_concat


if __name__ == '__main__':
    getTrainTestValSplit(_dataset_path=f'{ROOT_DIR}/dataset/ExtraSensory/ExtraSensory.per_uuid_features_labels')
