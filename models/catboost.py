import mlflow
import mlflow.catboost
import optuna
import catboost
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

mlflow.set_experiment("With_class_balanced_CV_Accuracy")


def run_optimization(X, y, cat_features, n_splits=5):
    def objective(trial: optuna.Trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "eval_metric": "Accuracy",
            "loss_function": "Logloss",
            "auto_class_weights": "Balanced",
            "verbose": 0,
        }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        f1_scores, auc_scores, precision_scores, recall_scores, accuracy_scores = (
            [],
            [],
            [],
            [],
            [],
        )

        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_param("cat_features", ",".join(cat_features))

            for train_idx, valid_idx in skf.split(X, y):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

                model = catboost.CatBoostClassifier(**params)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_valid, y_valid),
                    cat_features=cat_features,
                    early_stopping_rounds=50,
                )

                y_pred = model.predict(X_valid)
                y_proba = model.predict_proba(X_valid)[:, 1]

                f1_scores.append(f1_score(y_valid, y_pred))
                auc_scores.append(roc_auc_score(y_valid, y_proba))
                precision_scores.append(precision_score(y_valid, y_pred))
                recall_scores.append(recall_score(y_valid, y_pred))
                accuracy_scores.append(accuracy_score(y_valid, y_pred))

            mlflow.log_metric("mean_f1_score", np.mean(f1_scores))
            mlflow.log_metric("mean_roc_auc_score", np.mean(auc_scores))
            mlflow.log_metric("mean_precision", np.mean(precision_scores))
            mlflow.log_metric("mean_recall", np.mean(recall_scores))
            mlflow.log_metric("mean_accuracy", np.mean(accuracy_scores))

            mlflow.catboost.log_model(model, "model")

        return np.mean(accuracy_scores)  # optimize accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    return study
