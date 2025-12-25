import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline   # NOTE: from imblearn.pipeline

from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from preprocessor import column_processor, get_train_test_data, preprocess_data

from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np


def train_ensemble(X_train, y_train, X_val, y_val, preprocess):
    # 1. SVM
    # C=1.0 is the default; smaller C (e.g., 0.1) creates a "softer" margin (more regularization)
    svm = SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced", random_state=42)

    # 2 & 3. Base models for Ensemble (to be tuned via GridSearch)
    dt = DecisionTreeClassifier(class_weight="balanced", random_state=42)
    rf = RandomForestClassifier(class_weight="balanced", random_state=42)

    # 4. Bagging and Boosting
    # Bagging reduces variance; Boosting (AdaBoost) reduces bias
    bagging_dt = BaggingClassifier(estimator=dt, n_estimators=50, random_state=42)
    boosted_dt = AdaBoostClassifier(estimator=dt, n_estimators=50, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[
            ("svm", svm),
            ("bagged_dt", bagging_dt),
            ("boosted_dt", boosted_dt),
            ("rf", rf),
        ],
        voting="soft"
    )

    # Full Pipeline
    ensemble_model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("smote", SMOTE(random_state=42)),
        ("vote", voting_clf),
    ])

    # 5. Grid Search for DT and RF within the Pipeline
    # Syntax: nameOfStep__nameOfEstimator__parameter
    param_grid = {
        # Decision Tree Tuning (inside the bagging/boosting wrappers)
        'vote__bagged_dt__estimator__max_depth': range(3, 8),
        'vote__bagged_dt__estimator__min_samples_leaf': range(5, 11),
        
        # Random Forest Tuning
        'vote__rf__n_estimators': range(100, 201, 20),
        'vote__rf__max_depth': range(3, 8),
        
        # SVM Tuning
        'vote__svm__C': [0.01, 0.1, 0.5, 1, 2, 5, 10]
    }

    grid_search = GridSearchCV(ensemble_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search

def validation_grid_search(ensemble_model, X_val, y_val):

    # Get validation probabilities for positive class
    y_val_proba = ensemble_model.predict_proba(X_val)[:, 1]

    # Grid search over thresholds to emphasize recall (F2)
    thresholds = np.linspace(0.05, 0.95, 19)
    desired_recall = 0.80
    candidate_thr = None
    best_precision = -1

    for thr in thresholds:
        y_pred = (y_val_proba >= thr).astype(int)
        precision, recall, _, _ = precision_recall_fscore_support(
            y_val, y_pred, average="binary", zero_division=0
        )
        if recall >= desired_recall and precision > best_precision:
            best_precision = precision
            candidate_thr = thr

    print(f"Thr with recall ≥ {desired_recall}: {candidate_thr}, precision={best_precision}")

    return candidate_thr

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


    
    