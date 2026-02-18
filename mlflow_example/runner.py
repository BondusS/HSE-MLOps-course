import yaml
import mlflow
from scripts.process_data import process_data
from scripts.train import train
from scripts.evaluate import evaluate
from constants import MLFLOW_URL

def run_experiment(run_name, process_data_params, train_params):
    mlflow.set_tracking_uri(MLFLOW_URL)
    mlflow.set_experiment('homework_Bondarenko')
    with mlflow.start_run(run_name=run_name):
        print(f"--- Starting run: {run_name} ---")
        mlflow.log_params({**process_data_params, **train_params})

        with open('params/process_data.yaml', 'w') as f:
            yaml.dump({'params': process_data_params}, f)
        with open('params/train.yaml', 'w') as f:
            yaml.dump({'params': train_params}, f)

        process_data()
        train()
        evaluate()

if __name__ == '__main__':
    # Define a base feature set to avoid repetition (DRY principle)
    base_features = ['race', 'sex', 'native.country', 'occupation', 'education', 'capital.gain']

    # Experiment 1: Logistic Regression with different features
    feature_sets = [
        base_features,
        ['race', 'sex', 'native.country', 'occupation', 'education'],
        ['race', 'sex', 'native.country', 'occupation'],
    ]
    for i, feature_set in enumerate(feature_sets):
        process_data_params = {'features': feature_set, 'train_size': -1}
        train_params = {'model_type': 'LogisticRegression', 'penalty': 'l2', 'C': 0.9, 'solver': 'lbfgs', 'max_iter': 1000}
        run_name = f"log_reg_features_{len(feature_set)}_cols"
        run_experiment(run_name, process_data_params, train_params)

    # Experiment 2: Decision Tree with different max_depth
    for depth in [3, 5, 7, 10]:
        process_data_params = {'features': base_features, 'train_size': -1}
        train_params = {'model_type': 'DecisionTreeClassifier', 'max_depth': depth}
        run_name = f"decision_tree_depth_{depth}"
        run_experiment(run_name, process_data_params, train_params)

    # Experiment 3: Random Forest with different n_estimators
    for n_estimators in [50, 100, 150, 200]:
        process_data_params = {'features': base_features, 'train_size': -1}
        train_params = {'model_type': 'RandomForestClassifier', 'n_estimators': n_estimators}
        run_name = f"random_forest_estimators_{n_estimators}"
        run_experiment(run_name, process_data_params, train_params)

    # Experiment 4: Gradient Boosting with different learning rates
    for lr in [0.01, 0.1, 0.2, 0.3]:
        process_data_params = {'features': base_features, 'train_size': -1}
        train_params = {'model_type': 'GradientBoostingClassifier', 'learning_rate': lr}
        run_name = f"grad_boosting_lr_{lr:.2f}"
        run_experiment(run_name, process_data_params, train_params)

    # Experiment 5: Logistic Regression with different train sizes
    for train_size in [100, 500, 1000, 2000, -1]:
        process_data_params = {'features': base_features, 'train_size': train_size}
        train_params = {'model_type': 'LogisticRegression', 'penalty': 'l2', 'C': 0.9, 'solver': 'lbfgs', 'max_iter': 1000}
        run_name = f"log_reg_train_size_{'full' if train_size == -1 else train_size}"
        run_experiment(run_name, process_data_params, train_params)
