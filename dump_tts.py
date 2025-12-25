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

    # Perform Undersampling to achieve 50/50 split on readmitted_30d
    count_class_1 = y.sum()
    df_class_0 = df[df[readmitted_column] == 0]
    df_class_1 = df[df[readmitted_column] == 1]

    df_class_0_under = df_class_0.sample(count_class_1, random_state=42)
    df_balanced = pd.concat([df_class_0_under, df_class_1], axis=0).sample(frac=1, random_state=42)

    X_balanced = df_balanced.drop(readmitted_column, axis=1)
    y_balanced = df_balanced[readmitted_column]

    print(f"Balanced Dataset Distribution:\n{y_balanced.value_counts()}")

    # First: create a test set (~25%) with stratification
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_balanced, y_balanced,
        test_size=0.25,
        stratify=y_balanced,
        random_state=42
    )

    # Second: split remaining into train + validation (50/25)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.333,     # 0.333 of 0.75 is ~0.25 of total → for val
        stratify=y_temp,
        random_state=42
    )

    # Print class distributions to verify correctness
    print("\nSplit Distributions (counts):")
    print("Train:\n", y_train.value_counts())
    print("Val:  \n", y_val.value_counts())
    print("Test: \n", y_test.value_counts())

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
    