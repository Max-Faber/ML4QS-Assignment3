import pandas as pd

from definitions import ROOT_DIR

if __name__ == '__main__':
    df_uuid_test: pd.DataFrame = pd.read_csv(f'{ROOT_DIR}/datasets/ExtraSensory/ExtraSensory.per_uuid_features_labels/0A986513-7828-4D53-AA1F-E02D6DF9561B.features_labels.csv')
    df_uuid_test.insert(1, 'datetime', pd.to_datetime(df_uuid_test['timestamp'], unit='s'))
    pass
