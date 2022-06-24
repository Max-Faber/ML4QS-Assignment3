import os
import pandas as pd

from trainTestValSplit import getCSVs
from preProcessing import preProcess
from definitions import ROOT_DIR, main_activity_label_classes


def fixMissingLabels() -> None:
    baseDir: str = f'{ROOT_DIR}/dataset/ExtraSensory'

    dirProcessed: str = 'Processed'
    dirProcessedOutliers: str = 'Processed_outliers'
    dirProcessedOutliersWithLabels: str = 'Processed_outliers_with_labels'

    pathProcessed: str = f'{baseDir}/{dirProcessed}'
    pathProcessedOutliers: str = f'{baseDir}/{dirProcessedOutliers}'
    pathProcessedOutliersWithLabels: str = f'{baseDir}/{dirProcessedOutliersWithLabels}'

    os.makedirs(pathProcessedOutliersWithLabels, exist_ok=True)

    processedCSVs: dict[str, pd.DataFrame] = getCSVs(_dataset_path=pathProcessed)
    processedOutliersCSVs: dict[str, pd.DataFrame] = getCSVs(_dataset_path=pathProcessedOutliers)

    target_labels: list[str] = list(main_activity_label_classes.keys())
    target_labels_without_unknown: list[str] = list(set(target_labels) - {'label:UNKNOWN'})

    for path, df in processedCSVs.items():
        df_x, series_y = preProcess(_df=df, add_cols=target_labels_without_unknown)
        df_x['label:UNKNOWN'] = series_y == main_activity_label_classes['label:UNKNOWN']
        df_x['label:UNKNOWN'] = df_x['label:UNKNOWN'].astype(int)
        dfProcessedOutliers: pd.DataFrame = processedOutliersCSVs[path.replace(dirProcessed, dirProcessedOutliers)]
        dfProcessedOutliers[target_labels] = df_x[target_labels]
        exportPath: str = path.replace(dirProcessed, dirProcessedOutliersWithLabels)
        dfProcessedOutliers.to_csv(exportPath, index=False)


if __name__ == '__main__':
    fixMissingLabels()
