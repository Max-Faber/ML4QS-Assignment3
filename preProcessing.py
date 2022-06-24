import pandas as pd

from definitions import ROOT_DIR, main_activity_label_classes, columns_to_keep
from trainTestValSplit import getTrainTestValSplit
from copy import deepcopy


def addClassLabels(_df: pd.DataFrame) -> pd.DataFrame:
    _df['target'] = main_activity_label_classes['label:UNKNOWN']
    remaining_label_classes: dict[str, int] = deepcopy(main_activity_label_classes)
    del remaining_label_classes['label:UNKNOWN']

    for col_name, class_label in list(remaining_label_classes.items()):
        _df[col_name] = _df[col_name].fillna(0)
        _df.loc[_df[col_name] == 1, 'target'] = class_label
    return _df


def removeLabels(_df: pd.DataFrame) -> pd.DataFrame:
    return _df.drop(columns=[col for col in _df.columns if 'label:' in col])


def preProcess(_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    _df: pd.DataFrame = addClassLabels(_df=_df)
    _df: pd.DataFrame = removeLabels(_df=_df)
    _df: pd.DataFrame = _df.drop(columns=['timestamp'])
    _df: pd.DataFrame = _df[list(set(columns_to_keep).union({'target'}))]
    y: pd.Series = _df['target']
    x: pd.DataFrame = _df.drop(columns=['target'])
    del _df
    return x, y


if __name__ == '__main__':
    train, test, val = getTrainTestValSplit(
        _dataset_path=f'{ROOT_DIR}/dataset/ExtraSensory/Processed'
    )

    train_x, train_y = preProcess(_df=train)
    test_x, test_y = preProcess(_df=test)
    val_x, val_y = preProcess(_df=val)
