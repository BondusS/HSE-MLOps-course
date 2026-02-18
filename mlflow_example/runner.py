import yaml
import mlflow
from scripts.process_data import process_data
from scripts.train import train
from scripts.evaluate import evaluate
from constants import MLFLOW_URL

def run_experiment(process_data_params, train_params):
    mlflow.set_tracking_uri(MLFLOW_URL)
    mlflow.set_experiment('homework_Bondarenko')
    with mlflow.start_run():
        mlflow.log_params(process_data_params)
        mlflow.log_params(train_params)

        with open('params/process_data.yaml', 'w') as f:
            yaml.dump({'params': process_data_params}, f)
        with open('params/train.yaml', 'w') as f:
            yaml.dump({'params': train_params}, f)

        process_data()
        train()
        evaluate()

if __name__ == '__main__':
    # Experiment 1: Logistic Regression with different features
    features = [
        ['race', 'sex', 'native.country', 'occupation', 'education', 'capital.gain'],
        ['race', 'sex', 'native.country', 'occupation', 'education'],
        ['race', 'sex', 'native.country', 'occupation'],
    ]
    for feature_set in features:
        process_data_params = {'features': feature_set, 'train_size': -1}
        train_params = {'model_type': 'LogisticRegression', 'penalty': 'l2', 'C': 0.9, 'solver': 'lbfgs', 'max_iter': 1000}
        run_experiment(process_data_params, train_params)

    # Experiment 2: Decision Tree with different max_depth
    for depth in [3, 5, 7, 10]:
        process_data_params = {'features': features[0], 'train_size': -1}
        train_params = {'model_type': 'DecisionTreeClassifier', 'max_depth': depth}
        run_experiment(process_data_params, train_params)

    # Experiment 3: Random Forest with different n_estimators
    for n_estimators in [50, 100, 150, 200]:
        process_data_params = {'features': features[0], 'train_size': -1}
        train_params = {'model_type': 'RandomForestClassifier', 'n_estimators': n_estimators}
        run_experiment(process_data_params, train_params)

    # Experiment 4: Gradient Boosting with different learning rates
    for lr in [0.01, 0.1, 0.2, 0.3]:
        process_data_params = {'features': features[0], 'train_size': -1}
        train_params = {'model_type': 'GradientBoostingClassifier', 'learning_rate': lr}
        run_experiment(process_data_params, train_params)

    # Experiment 5: Logistic Regression with different train sizes
    for train_size in [100, 500, 1000, 2000, -1]:
        process_data_params = {'features': features[0], 'train_size': train_size}
        train_params = {'model_type': 'LogisticRegression', 'penalty': 'l2', 'C': 0.9, 'solver': 'lbfgs', 'max_iter': 1000}
        run_experiment(process_data_params, train_params)
