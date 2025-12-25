import pandas as pd
from sklearn.model_selection import train_test_split
from config import EHR_DATA_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH


def main():
    # Load your dataset
    df = pd.read_csv(EHR_DATA_PATH)
    readmitted_column = "readmitted_30d"


    # Separate features and target
    X = df.drop(readmitted_column, axis=1)
    y = df[readmitted_column]

    # First: create a test set (unseen, ~20%) with stratification
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.20,      # 100 out of 500 approx
        stratify=y,
        random_state=42
    )

    # Second: split remaining into train + validation (70/10)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.125,     # 0.125 of 400 = 50 → for val
        stratify=y_temp,
        random_state=42
    )

    # # Print class distributions to verify correctness
    # print("Train:", y_train.value_counts(normalize=True))
    # print("Val:  ", y_val.value_


    # -----------------------------
    # Save splits to CSV
    # -----------------------------
    train_data = pd.concat([X_train, y_train], axis=1)
    val_data   = pd.concat([X_val,   y_val],   axis=1)
    test_data  = pd.concat([X_test,  y_test],  axis=1)

    train_data.to_csv(TRAIN_PATH, index=False)
    val_data.to_csv(VAL_PATH, index=False)
    test_data.to_csv(TEST_PATH, index=False)

    print(f"Files saved: {TRAIN_PATH}, {VAL_PATH}, {TEST_PATH}")


if __name__ == "__main__":
    main()
    