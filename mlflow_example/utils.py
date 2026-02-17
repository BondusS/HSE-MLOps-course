import logging
import os
import yaml
import warnings

from sklearn.exceptions import DataConversionWarning

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(name)s : %(message)s')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DataConversionWarning)

PARAMS_FILEPATH_PATTERN = '/app/params/{stage_name}.yaml'


def load_params(stage_name: str) -> dict:
    params_filepath = PARAMS_FILEPATH_PATTERN.format(stage_name=stage_name)
    if not os.path.exists(params_filepath):
        raise FileNotFoundError(
            f'Параметров для шага {stage_name} не существует! Проверьте имя шага'
        )
    with open(params_filepath, 'r') as file:
        params = yaml.safe_load(file)
    return params['params']


def get_logger(
    logger_name: str | None = None,
    level: int = 20,
) -> logging.Logger:
    logger = logging.getLogger(name=logger_name)
    logger.setLevel(level)
    return logger
