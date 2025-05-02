import os
import sys
import numpy as np
import random
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score,
                             confusion_matrix, precision_recall_curve, roc_curve)
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import joblib
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

FEATURE_DIR = sys.argv[1]
BASE_PATH = os.path.join("features", FEATURE_DIR)

def load_data(feature):
    X = []
    y = []
    for label in [0, 1]:
        folder = os.path.join(BASE_PATH, str(label))
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            if feature == "mel_spectrogram":
                img = Image.open(path).convert("L").resize((128, 128))
                arr = np.array(img).flatten()
            else:
                arr = np.load(path).flatten()
            X.append(arr)
            y.append(label)
    return np.array(X), np.array(y)

X, y = load_data(FEATURE_DIR)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, stratify=y_temp, random_state=SEED)

param_grid = dict(
    XGB__max_depth = [10, 11, 12],
    XGB__gamma = np.random.uniform(0,1,3),
    XGB__n_estimators = [42, 43, 44, 45],
    XGB__tree_method = ["hist"],
    XGB__learning_rate = [0.1, 0.15, 0.2, 0.3, 0.4],
    XGB__seed = [42]
)

def objective(trial):
    max_depth = trial.suggest_categorical("max_depth", param_grid["XGB__max_depth"])
    gamma = trial.suggest_float("gamma", 0.0, 1.0)
    n_estimators = trial.suggest_categorical("n_estimators", param_grid["XGB__n_estimators"])
    learning_rate = trial.suggest_categorical("learning_rate", param_grid["XGB__learning_rate"])

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    f1_scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val_inner = X_train[train_idx], X_train[val_idx]
        y_tr, y_val_inner = y_train[train_idx], y_train[val_idx]

        class_weights = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

        model = XGBClassifier(
            max_depth=max_depth,
            gamma=gamma,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            tree_method="hist",
            seed=SEED,
            eval_metric="logloss",
            scale_pos_weight=class_weight_dict[1]/class_weight_dict[0]
        )

        model.fit(X_tr, y_tr)
        y_pred_proba = model.predict_proba(X_val_inner)[:, 1]
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_val_inner, y_pred_proba)
        f1_vals = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
        optimal_idx = np.argmax(f1_vals)
        optimal_threshold = thresholds[optimal_idx] if thresholds.size > 0 else 0.5

        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        f1_scores.append(f1_score(y_val_inner, y_pred))

    return np.mean(f1_scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)


best_params = study.best_trial.params
final_model = XGBClassifier(
    max_depth=best_params["max_depth"],
    gamma=best_params["gamma"],
    n_estimators=best_params["n_estimators"],
    learning_rate=best_params["learning_rate"],
    tree_method="hist",
    seed=SEED,
    eval_metric="logloss"
)

X_final_train = np.concatenate([X_train, X_val], axis=0)
y_final_train = np.concatenate([y_train, y_val], axis=0)
class_weights = compute_class_weight("balanced", classes=np.unique(y_final_train), y=y_final_train)
final_model.set_params(scale_pos_weight=class_weights[1]/class_weights[0])
final_model.fit(X_final_train, y_final_train)

model_path = f"xgb_{FEATURE_DIR}.model"
joblib.dump(final_model, model_path)
print(f"Model saved to {model_path}")

y_val_proba = final_model.predict_proba(X_val)[:, 1]
precision_vals, recall_vals, thresholds = precision_recall_curve(y_val, y_val_proba)
f1_vals = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
optimal_idx = np.argmax(f1_vals)
optimal_threshold = thresholds[optimal_idx] if thresholds.size > 0 else 0.5

y_test_proba = final_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

auc_score = roc_auc_score(y_test, y_test_proba)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
cm = confusion_matrix(y_test, y_test_pred)
p_misclass = cm[1, 0] / cm[1].sum() * 100 if cm[1].sum() != 0 else 0
n_misclass = cm[0, 1] / cm[0].sum() * 100 if cm[0].sum() != 0 else 0

print(f"\nFinal Test Set Evaluation for {FEATURE_DIR}:")
print(f"AUC: {auc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
print(f"pMisclass (Positive Misclassification Rate): {p_misclass:.2f}%")
print(f"nMisclass (Negative Misclassification Rate): {n_misclass:.2f}%")

fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall_vals, precision_vals, label="Precision-Recall Curve")
plt.axhline(y=np.mean(y_test), linestyle='--', color='gray', label="No Skill Line")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()

plt.tight_layout()
plt.savefig(f"xgb_{FEATURE_DIR}_plots.png")
plt.close()