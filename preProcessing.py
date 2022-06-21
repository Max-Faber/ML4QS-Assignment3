import pandas as pd

from definitions import ROOT_DIR
from trainTestValSplit import getTrainTestValSplit

def preProcess(_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return _df

if __name__ == '__main__':
    # lying down, sitting, standing in place, standing and moving, walking, running, bicycling.
    main_activity_labels: list[str] = ['label:LYING_DOWN', 'label:SITTING', 'label:OR_standing', 'label:FIX_walking', 'label:FIX_running', '']
    train, test, val = getTrainTestValSplit(_dataset_path=f'{ROOT_DIR}/dataset/ExtraSensory/ExtraSensory.per_uuid_features_labels')
    train = preProcess(_df=train)
    test = preProcess(_df=test)
    val = preProcess(_df=val)
    # df_uuid_test.insert(1, 'datetime', pd.to_datetime(df_uuid_test['timestamp'], unit='s'))
    # pass
