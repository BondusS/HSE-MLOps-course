import pandas as pd
import mlflow
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
from utils import get_logger, load_params

STAGE_NAME = 'train'


def train():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')

    logger.info('Создаём модель')
    model_type = params.pop('model_type')
    params['random_state'] = RANDOM_STATE
    logger.info(f'    Тип модели: {model_type}')
    logger.info(f'    Параметры модели: {params}')

    if model_type == 'LogisticRegression':
        model = LogisticRegression(**params)
    elif model_type == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier(**params)
    elif model_type == 'RandomForestClassifier':
        model = RandomForestClassifier(**params)
    elif model_type == 'GradientBoostingClassifier':
        model = GradientBoostingClassifier(**params)
    else:
        raise ValueError(f'Неизвестный тип модели: {model_type}')

    logger.info('Обучаем модель')
    model.fit(X_train, y_train)

    logger.info('Сохраняем и логируем модель')
    dump(model, MODEL_FILEPATH)
    mlflow.sklearn.log_model(model, 'model')
    logger.info('Успешно!')


if __name__ == '__main__':
    train()
