import pandas as pd
from os import listdir
from preProcessing import preProcess

def load_data(dataset_path):
    files = listdir(dataset_path)
    combined = []

    for file in files:
        user = pd.read_csv(dataset_path + file)
        name = file.split('.')[0]
        user['name'] = name
        combined.append(user)

    combined = pd.concat(combined, axis=0, ignore_index=True)
    combined = preProcess(combined, add_cols=['name'])

    del user, files

    #train_concat: pd.DataFrame = train_concat.drop(columns=[_ for _ in train_concat.columns if 'Unnamed' in _])

    return combined

