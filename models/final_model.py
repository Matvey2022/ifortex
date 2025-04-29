import mlflow
import catboost
import mlflow.catboost
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)


mlflow.set_experiment("final_training")


def run_final_training(X_train, y_train, X_test, y_test, cat_features):
    best_params = {
        "iterations": 554,
        "learning_rate": 0.25486612921338103,
        "depth": 5,
        "l2_leaf_reg": 1.2920326786341383,
        "subsample": 0.6876616820894427,
        "eval_metric": "Accuracy",
        "loss_function": "Logloss",
        "auto_class_weights": "Balanced",
    }
    with mlflow.start_run():
        mlflow.log_params(best_params)
        model = catboost.CatBoostClassifier(**best_params)
        model.fit(
            X_train,
            y_train,
            verbose=0,
            cat_features=cat_features,
        )

        y_pred = model.predict(X_test)
        preds_proba = model.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, preds_proba)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc_score", auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.catboost.log_model(model, "model")
    return model, f1, auc, precision, recall, accuracy
