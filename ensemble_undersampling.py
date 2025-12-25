import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from preprocessor import column_processor, get_train_test_data, preprocess_data
import joblib
from joblib import Parallel, delayed
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def train_ensemble(X_train, y_train, X_val, y_val, preprocess):
    # Base pipeline structure for individual tuning (NO SMOTE)
    def create_tuning_pipeline(clf):
        return Pipeline(steps=[
            ("preprocess", preprocess),
            ("clf", clf),
        ])

    # 1. Tune SVM
    print("\nTuning SVM...")
    svm_pipe = create_tuning_pipeline(SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42))
    param_grid_svm = {'clf__C': [0.01, 0.1, 0.5, 1, 2, 5, 10]}
    gs_svm = GridSearchCV(svm_pipe, param_grid_svm, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    gs_svm.fit(X_train, y_train)
    best_svm = gs_svm.best_estimator_.named_steps['clf']
    print(f"Best SVM Params: {gs_svm.best_params_}")

    # 2. Tune Decision Tree
    print("\nTuning Decision Tree...")
    dt_pipe = create_tuning_pipeline(DecisionTreeClassifier(class_weight="balanced", random_state=42))
    param_grid_dt = {
        'clf__max_depth': range(3, 8),
        'clf__min_samples_leaf': range(5, 11),
    }
    gs_dt = GridSearchCV(dt_pipe, param_grid_dt, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    gs_dt.fit(X_train, y_train)
    best_dt = gs_dt.best_estimator_.named_steps['clf']
    print(f"Best DT Params: {gs_dt.best_params_}")

    # 3. Tune Random Forest
    print("\nTuning Random Forest...")
    rf_pipe = create_tuning_pipeline(RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1))
    param_grid_rf = {
        'clf__n_estimators': range(100, 201, 20),
        'clf__max_depth': range(3, 8),
    }
    gs_rf = GridSearchCV(rf_pipe, param_grid_rf, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    gs_rf.fit(X_train, y_train)
    best_rf = gs_rf.best_estimator_.named_steps['clf']
    print(f"Best RF Params: {gs_rf.best_params_}")

    print("\nConstructing Final Ensemble...")
    bagging_dt = BaggingClassifier(estimator=best_dt, n_estimators=50, random_state=42, n_jobs=1)
    boosted_dt = AdaBoostClassifier(estimator=best_dt, n_estimators=50, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[
            ("svm", best_svm),
            ("bagged_dt", bagging_dt),
            ("boosted_dt", boosted_dt),
            ("rf", best_rf),
        ],
        voting="soft",
        n_jobs=1
    )

    ensemble_model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("vote", voting_clf),
    ])

    ensemble_model.fit(X_train, y_train)
    return ensemble_model

def validation_grid_search(ensemble_model, X_val, y_val):
    y_val_proba = ensemble_model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 19)
    best_thr = 0.5
    best_f2 = -1
    best_precision = 0
    recall_threshold = 0.6

    print("\nThreshold Tuning (Val Set):")
    for thr in thresholds:
        y_pred = (y_val_proba >= thr).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average="binary", zero_division=0
        )
        beta = 2.0
        f2 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-12)

        print(f"thr={thr:.2f}  P={precision:.3f}  R={recall:.3f}  F2={f2:.3f}")

        if precision > best_precision and recall > recall_threshold:
            best_precision = precision
            best_f2 = f2
            best_thr = thr

    print(f"\nBest threshold by F2 on val: {best_thr:.2f}, F2={best_f2:.3f}")
    return best_thr

def test_model(model, X_test, y_test, threshold):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    print(f"Test Precision: {precision:.3f}, Test Recall: {recall:.3f}")

def main():
    print("Loading data...")
    train_df, val_df, test_df = get_train_test_data()
    X_train, y_train = preprocess_data(train_df)
    X_val, y_val = preprocess_data(val_df)
    X_test, y_test = preprocess_data(test_df)
    col_processor = column_processor()
    print("Training...")
    model = train_ensemble(X_train, y_train, X_val, y_val, col_processor)
    print("Validation...")
    threshold = validation_grid_search(model, X_val, y_val)
    print("Testing...")
    test_model(model, X_test, y_test, threshold)

if __name__ == "__main__":
    main()
