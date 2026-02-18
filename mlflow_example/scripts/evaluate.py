import os
import json
import numpy as np
import pandas as pd
import mlflow
from joblib import load
from sklearn.metrics import get_scorer, classification_report, confusion_matrix, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH
from utils import get_logger, load_params

STAGE_NAME = 'evaluate'


def evaluate():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')

    logger.info('Загружаем обученную модель')
    if not os.path.exists(MODEL_FILEPATH):
        raise FileNotFoundError(
            'Не нашли файл с моделью. Убедитесь, что был запущен шаг с обучением'
        )
    model = load(MODEL_FILEPATH)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    logger.info('Начали считать метрики на тесте')
    metrics = {}
    for metric_name in params['metrics']:
        scorer = get_scorer(metric_name)
        score = scorer(model, X_test, y_test)
        metrics[metric_name] = score
    metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    metrics['pr_auc'] = average_precision_score(y_test, y_proba)
    mlflow.log_metrics(metrics)
    logger.info(f'Значения метрик - {metrics}')

    logger.info('Логируем артефакты')
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    with open('classification_report.json', 'w') as f:
        json.dump(report, f)
    mlflow.log_artifact('classification_report.json')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    logger.info('Успешно!')


if __name__ == '__main__':
    evaluate()
